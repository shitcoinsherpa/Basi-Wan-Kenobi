"""Gating-experiment harness: vllm's vendored llama.cpp MMQ Q4_K vs our baseline.

Per memory/research_marlin_q4k_port_2026-06-02.md, before investing 13-15 days in a
from-scratch native Q4_K mma kernel, we want to know whether the existing vllm-vendored
MMQ kernel already wins over our `dequantize + F.linear` baseline at our real operating
points (M=7800 for p480_17f, M=32400 for p720_33f). llama.cpp's heuristic switches to
cuBLAS at M>256, which is circumstantial evidence MMQ loses at large M — but we want a
measurement, not a guess.

Sweep: M = {17, 1000, 4000, 7800, 32400}, N=5120, K=5120, dtype=bf16, device=cuda.

For each (M, kernel) measure: median wall ms across 20 iters, max-abs vs baseline,
peak alloc bytes.

Decision rule:
  - vllm MMQ >= 1.3x baseline at M=7800 -> ship the wrapper (1-2 day integration)
  - vllm MMQ < 1.1x baseline at M=7800 -> from-scratch native Q4_K Marlin port is justified
  - Borderline (1.1x..1.3x) -> bench Q6_K and the Wan-actual tensor shapes before deciding

Usage:
  python -m gguf_vendor.bench_vllm_mmq_vs_baseline \
      --gguf /path/to/Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf \
      --tensor blocks.0.self_attn.q.weight \
      --m-sizes 17,1000,4000,7800,32400 \
      --out bench_results/mmq_vs_baseline.json

vllm-not-installed handling:
  If `vllm` import fails, the script logs a clear pip install line and skips the MMQ
  kernel, still emitting baseline + v1.1 results so the run is partially useful.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import warnings
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

# Locate q4k_triton_gemm module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gguf_vendor.q4k_triton_gemm import (  # noqa: E402
    Q4_K_QTYPE,
    load_gguf_tensor,
    q4k_gemm,
    reference_q4k_linear,
    slice_q4k_weight,
    _select_launch_config,
)
from gguf_vendor.dequant import dequantize  # noqa: E402


def _try_import_vllm():
    """Attempt to locate vllm's GGUF Q4_K matmul entry point.

    vllm exposes the vendored llama.cpp kernels via `vllm._custom_ops` (older builds)
    or `vllm._C` (newer C-extension layout). The relevant ops:
      - ggml_mul_mat_a8(W_packed, x_q8_1, qtype, n_rows) -> output (fp16/bf16)
      - ggml_quantize_q8_1(x_fp16_or_bf16) -> Q8_1 packed tensor

    Q8_1 quantization of activations is required by MMQ — the kernel does NOT accept
    fp16/bf16 activations directly. The caller must pre-quantize.

    Returns dict with {ok: bool, mul_mat: callable | None, quantize: callable | None,
                       reason: str}.
    """
    info = {"ok": False, "mul_mat": None, "quantize": None, "reason": ""}
    try:
        try:
            from vllm import _custom_ops as ops  # type: ignore
        except Exception:
            from vllm import _C as ops  # type: ignore
    except Exception as e:
        info["reason"] = f"vllm import failed ({type(e).__name__}: {e}). pip install vllm"
        return info

    mul_mat = getattr(ops, "ggml_mul_mat_a8", None) or getattr(ops, "gguf_mul_mat_a8", None)
    quantize = getattr(ops, "ggml_quantize_q8_1", None) or getattr(ops, "gguf_quantize_q8_1", None)

    # vllm ≥0.20 internalises Q8_1 activation quantisation inside ggml_mul_mat_a8
    # — the public signature became (W, X_fp, qtype, row) → out. The explicit
    # ggml_quantize_q8_1 entry was removed. Accept that layout.
    if mul_mat is None:
        info["reason"] = (
            f"vllm imported, but ggml_mul_mat_a8 not found on the ops module. "
            f"Available attrs starting with 'ggml' or 'gguf': "
            f"{[a for a in dir(ops) if a.startswith(('ggml', 'gguf'))]}"
        )
        return info

    info["ok"] = True
    info["mul_mat"] = mul_mat
    info["quantize"] = quantize  # may be None; see _vllm_mmq
    info["reason"] = "ok" if quantize is not None else "ok (mul_mat-only; Q8_1 internalised)"
    return info


def _bench_one(
    label: str,
    fn: Callable[[], torch.Tensor],
    device: torch.device,
    warmup: int = 5,
    iters: int = 20,
) -> dict:
    """Run a kernel and return median ms, peak alloc, and the last output (for diffing)."""
    try:
        for _ in range(warmup):
            out = fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        times = []
        for _ in range(iters):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            out = fn()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append((time.perf_counter() - t0) * 1000)

        peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        median_ms = statistics.median(times)
        p95_ms = sorted(times)[int(len(times) * 0.95)] if times else 0.0
        return {
            "label": label,
            "ok": True,
            "median_ms": median_ms,
            "p95_ms": p95_ms,
            "peak_alloc": peak,
            "iters": iters,
            "warmup": warmup,
            "out": out,
        }
    except Exception as e:
        return {"label": label, "ok": False, "error": f"{type(e).__name__}: {e}", "out": None}


def _baseline_dequant_linear(x, raw_weight, logical_shape):
    w = dequantize(raw_weight, Q4_K_QTYPE, logical_shape, dtype=x.dtype)
    return F.linear(x, w)


def _v1_fused(x, raw_weight, logical_shape):
    return q4k_gemm(x, raw_weight, logical_shape, out_dtype=x.dtype)


def _vllm_mmq(x, raw_weight, logical_shape, vllm_info):
    """Call vllm's MMQ. Newer vllm (≥0.20) accepts fp16/bf16 X directly and
    Q8_1-quantises internally; older vllm exposed ggml_quantize_q8_1 separately."""
    quantize = vllm_info["quantize"]
    mul_mat = vllm_info["mul_mat"]
    n_rows = logical_shape[0]
    x_in = quantize(x) if quantize is not None else x
    try:
        out = mul_mat(raw_weight, x_in, int(Q4_K_QTYPE), n_rows)
    except TypeError:
        # Some versions: mul_mat(W, x, qtype) without n_rows
        out = mul_mat(raw_weight, x_in, int(Q4_K_QTYPE))
    if out.dtype != x.dtype:
        out = out.to(x.dtype)
    return out


def run_sweep(args) -> dict:
    device = torch.device(args.device)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    raw_weight, logical_shape = load_gguf_tensor(args.gguf, args.tensor)
    raw_weight, logical_shape = slice_q4k_weight(
        raw_weight, logical_shape, n_rows=args.n, k_cols=args.k
    )
    raw_weight = raw_weight if device.type == "cpu" else raw_weight.to(device)
    n_size, k_size = logical_shape
    print(f"tensor={args.tensor} logical_shape={logical_shape} dtype={dtype} device={device}")

    vllm_info = _try_import_vllm()
    if vllm_info["ok"]:
        print(f"[vllm] ok: mul_mat={vllm_info['mul_mat']}, quantize={vllm_info['quantize']}")
    else:
        print(f"[vllm] SKIPPED: {vllm_info['reason']}")

    m_sizes = [int(m) for m in args.m_sizes.split(",")]
    results: list[dict] = []

    for m in m_sizes:
        print(f"\n=== M={m} N={n_size} K={k_size} dtype={dtype} ===")
        x = torch.randn(m, k_size, dtype=dtype, device=device)
        block_m, block_n, num_warps, num_stages = _select_launch_config(m, n_size)
        print(f"  v1.1 launch: BLOCK_M={block_m} BLOCK_N={block_n} num_warps={num_warps} num_stages={num_stages}")

        # Reference (baseline) — always run
        baseline = _bench_one(
            "baseline_dequant_linear",
            lambda: _baseline_dequant_linear(x, raw_weight, logical_shape),
            device,
            warmup=args.warmup,
            iters=args.iters,
        )

        # Our v1.1 fused
        v1 = _bench_one(
            "v1.1_fused",
            lambda: _v1_fused(x, raw_weight, logical_shape),
            device,
            warmup=args.warmup,
            iters=args.iters,
        )

        # vllm MMQ
        if vllm_info["ok"] and "vllm" in args.kernels.split(","):
            mmq = _bench_one(
                "vllm_mmq",
                lambda: _vllm_mmq(x, raw_weight, logical_shape, vllm_info),
                device,
                warmup=args.warmup,
                iters=args.iters,
            )
        else:
            mmq = {"label": "vllm_mmq", "ok": False, "error": vllm_info["reason"], "out": None}

        # Correctness vs baseline
        def _max_abs(a, b):
            if a is None or b is None or not (a.shape == b.shape):
                return None
            return float((a.to(torch.float32) - b.to(torch.float32)).abs().max().item())

        baseline_out = baseline.get("out")
        v1_max_abs = _max_abs(v1.get("out"), baseline_out) if v1["ok"] and baseline["ok"] else None
        mmq_max_abs = _max_abs(mmq.get("out"), baseline_out) if mmq["ok"] and baseline["ok"] else None

        # Strip "out" from results before saving
        for r in (baseline, v1, mmq):
            r.pop("out", None)
        v1["max_abs_vs_baseline"] = v1_max_abs
        mmq["max_abs_vs_baseline"] = mmq_max_abs

        # Speedups
        if baseline["ok"]:
            base_ms = baseline["median_ms"]
            for r in (v1, mmq):
                if r["ok"]:
                    r["speedup_vs_baseline"] = base_ms / r["median_ms"]

        row = {
            "m": m,
            "n": n_size,
            "k": k_size,
            "dtype": str(dtype),
            "launch": {
                "block_m": block_m,
                "block_n": block_n,
                "num_warps": num_warps,
                "num_stages": num_stages,
            },
            "kernels": {
                "baseline": baseline,
                "v1.1_fused": v1,
                "vllm_mmq": mmq,
            },
        }
        results.append(row)

        # Print summary
        for r in (baseline, v1, mmq):
            if r["ok"]:
                speedup = r.get("speedup_vs_baseline", 1.0)
                ma = r.get("max_abs_vs_baseline")
                ma_str = f" max_abs={ma:.4g}" if ma is not None else ""
                print(
                    f"  {r['label']:25s} median={r['median_ms']:7.3f}ms "
                    f"p95={r['p95_ms']:7.3f}ms peak={r['peak_alloc']/1e6:6.1f}MB "
                    f"speedup={speedup:.2f}x{ma_str}"
                )
            else:
                print(f"  {r['label']:25s} FAILED: {r.get('error')}")

    summary = {
        "tensor": args.tensor,
        "n_size": n_size,
        "k_size": k_size,
        "dtype": str(dtype),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "vllm_available": vllm_info["ok"],
        "vllm_skip_reason": vllm_info["reason"] if not vllm_info["ok"] else None,
        "results": results,
    }
    return summary


def _print_verdict(summary: dict) -> None:
    """Apply the decision rule from research_marlin_q4k_port_2026-06-02.md."""
    print("\n=== VERDICT (per memory/research_marlin_q4k_port_2026-06-02.md) ===")
    rows_7800 = [r for r in summary["results"] if r["m"] == 7800]
    if not rows_7800:
        print("  M=7800 row not present; verdict requires that operating point. Re-run with --m-sizes including 7800.")
        return
    row = rows_7800[0]
    mmq = row["kernels"]["vllm_mmq"]
    base = row["kernels"]["baseline"]
    if not (mmq.get("ok") and base.get("ok")):
        print(f"  insufficient data at M=7800: mmq ok={mmq.get('ok')} baseline ok={base.get('ok')}")
        return
    sp = mmq["speedup_vs_baseline"]
    if sp >= 1.3:
        verdict = "SHIP vllm MMQ wrapper (1-2 day integration). MMQ wins decisively at M=7800."
    elif sp < 1.1:
        verdict = (
            "MMQ loses at M=7800. From-scratch native Q4_K Marlin port (Option B, 13-15 days) "
            "is justified by the gap."
        )
    else:
        verdict = (
            f"BORDERLINE (speedup={sp:.2f}x). Bench Q6_K and Wan-actual tensor shapes before committing."
        )
    print(f"  M=7800 MMQ speedup vs baseline: {sp:.2f}x")
    print(f"  Verdict: {verdict}")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", required=True, help="Path to a Wan2.2-A14B GGUF file (HighNoise or LowNoise)")
    p.add_argument("--tensor", default="blocks.0.self_attn.q.weight")
    p.add_argument("--m-sizes", default="17,1000,4000,7800,32400", help="Comma-separated M values")
    p.add_argument("--n", type=int, default=5120, help="N rows to slice")
    p.add_argument("--k", type=int, default=5120, help="K cols to slice (multiple of 256)")
    p.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="bf16")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--kernels", default="baseline,v1.1,vllm", help="Comma-separated kernel names to run")
    p.add_argument("--out", default="bench_results/mmq_vs_baseline.json")
    args = p.parse_args(argv)

    if args.device != "cpu" and not torch.cuda.is_available():
        print("ERROR: --device cuda but no CUDA device available", file=sys.stderr)
        return 1

    summary = run_sweep(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[saved] {out_path}")
    _print_verdict(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
