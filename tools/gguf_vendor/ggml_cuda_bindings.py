"""CUDA matmul bindings for GGUF Q4_K/Q6_K via a small ggml bridge.

This is the v2 path after the Triton prototype:

1. Keep the GGUF packed weight in its original Q4_K/Q6_K layout.
2. Hand the packed bytes to a small C++ bridge library built against
   llama.cpp/ggml shared libraries with CUDA enabled.
3. Let ggml's production CUDA backend execute `ggml_mul_mat(weight, x)`
   while activations and outputs stay in device memory via D2D copies.

The Python side is intentionally conservative:
- If the bridge library is unavailable, ABI-mismatched, or the current
  tensor/device combination is unsupported, it returns `None`.
- Callers then fall back to the existing dequantize + `F.linear` path.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import gguf
import torch
import torch.nn.functional as F

try:
    from .dequant import dequantize
    from .loader import get_orig_shape
    from .ops import GGMLTensor
except ImportError:  # pragma: no cover - supports direct script execution
    from dequant import dequantize
    from loader import get_orig_shape
    from ops import GGMLTensor


Q4_K_QTYPE = gguf.GGMLQuantizationType.Q4_K
Q6_K_QTYPE = gguf.GGMLQuantizationType.Q6_K
SUPPORTED_QTYPES = {Q4_K_QTYPE, Q6_K_QTYPE}

BRIDGE_ABI_VERSION = 1
BRIDGE_CACHE_ATTR = "_faster_wan_ggml_cuda_bridge_cache"
MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_LIB_DIR = MODULE_DIR / "libs"
DEFAULT_BRIDGE_BASENAME = "libfaster_wan_ggml_cuda_bridge.so"
_BRIDGE_LOAD_ATTEMPTED = False
_BRIDGE_LOAD_ERROR: str | None = None
_BRIDGE_INSTANCE: "_BridgeLib | None" = None
_WARNED_MESSAGES: set[str] = set()


def _warn_once(message: str) -> None:
    if message in _WARNED_MESSAGES:
        return
    _WARNED_MESSAGES.add(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def _truthy_env(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def llama_cpp_env_enabled() -> bool:
    return _truthy_env("BASIWAN_Q4_LLAMACPP")


def _logical_shape(
    logical_shape: Sequence[int] | torch.Size | None,
) -> tuple[int, int]:
    if logical_shape is None or len(logical_shape) != 2:
        raise ValueError(f"expected a 2D logical shape, got {logical_shape!r}")
    return (int(logical_shape[0]), int(logical_shape[1]))


def _normalized_qtype(weight: torch.Tensor) -> gguf.GGMLQuantizationType | None:
    qtype = getattr(weight, "tensor_type", None)
    if qtype in SUPPORTED_QTYPES:
        return qtype
    qname = getattr(qtype, "name", None)
    if qname == "Q4_K":
        return Q4_K_QTYPE
    if qname == "Q6_K":
        return Q6_K_QTYPE
    return None


def _packed_weight_shape(
    qtype: gguf.GGMLQuantizationType,
    logical_shape: Sequence[int] | torch.Size,
) -> tuple[int, int]:
    n_rows, k_cols = _logical_shape(logical_shape)
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    if k_cols % block_size != 0:
        raise ValueError(
            f"{getattr(qtype, 'name', qtype)!r} requires K multiple of {block_size}, got {k_cols}"
        )
    packed_cols = (k_cols // block_size) * type_size
    return (n_rows, packed_cols)


def _validate_quant_weight(
    raw_weight: torch.Tensor,
    qtype: gguf.GGMLQuantizationType,
    logical_shape: Sequence[int] | torch.Size,
) -> tuple[int, int]:
    if raw_weight.ndim != 2:
        raise ValueError(f"expected packed weight to be 2D, got {tuple(raw_weight.shape)}")
    if raw_weight.dtype != torch.uint8:
        raise ValueError(f"expected packed weight dtype=torch.uint8, got {raw_weight.dtype}")
    expected = _packed_weight_shape(qtype, logical_shape)
    if tuple(raw_weight.shape) != expected:
        raise ValueError(
            f"packed weight shape mismatch: expected {expected}, got {tuple(raw_weight.shape)}"
        )
    return _logical_shape(logical_shape)


def _candidate_bridge_paths() -> list[Path]:
    env_path = os.getenv("BASIWAN_GGML_BRIDGE")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            DEFAULT_LIB_DIR / DEFAULT_BRIDGE_BASENAME,
            MODULE_DIR / "ggml_cuda_bridge" / "build" / DEFAULT_BRIDGE_BASENAME,
        ]
    )
    return candidates


def _candidate_lib_dirs(bridge_path: Path | None) -> list[Path]:
    dirs: list[Path] = []
    env_dir = os.getenv("BASIWAN_GGML_LIB_DIR")
    if env_dir:
        dirs.append(Path(env_dir))
    if bridge_path is not None:
        dirs.append(bridge_path.parent)
    dirs.append(DEFAULT_LIB_DIR)
    return dirs


class _BridgeLib:
    def __init__(self) -> None:
        bridge_path = self._resolve_bridge_path()
        self._preload_ggml_libs(bridge_path)
        mode = getattr(os, "RTLD_GLOBAL", 0)
        self._lib = ctypes.CDLL(str(bridge_path), mode=mode)

        self._lib.faster_wan_ggml_bridge_abi_version.argtypes = []
        self._lib.faster_wan_ggml_bridge_abi_version.restype = ctypes.c_int
        abi = int(self._lib.faster_wan_ggml_bridge_abi_version())
        if abi != BRIDGE_ABI_VERSION:
            raise RuntimeError(
                f"bridge ABI mismatch: expected {BRIDGE_ABI_VERSION}, got {abi}"
            )

        self._lib.faster_wan_ggml_last_error.argtypes = []
        self._lib.faster_wan_ggml_last_error.restype = ctypes.c_char_p

        self._lib.faster_wan_ggml_state_create.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self._lib.faster_wan_ggml_state_create.restype = ctypes.c_void_p

        self._lib.faster_wan_ggml_state_destroy.argtypes = [ctypes.c_void_p]
        self._lib.faster_wan_ggml_state_destroy.restype = None

        self._lib.faster_wan_ggml_mul_mat.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self._lib.faster_wan_ggml_mul_mat.restype = ctypes.c_int

    @staticmethod
    def _resolve_bridge_path() -> Path:
        for candidate in _candidate_bridge_paths():
            if candidate.is_file():
                return candidate
        searched = ", ".join(str(p) for p in _candidate_bridge_paths())
        raise FileNotFoundError(
            "ggml bridge library not found; searched: "
            f"{searched}. Build it with tools/gguf_vendor/ggml_cuda_bridge/CMakeLists.txt "
            "and set BASIWAN_GGML_BRIDGE to the resulting .so."
        )

    @staticmethod
    def _preload_ggml_libs(bridge_path: Path) -> None:
        mode = getattr(os, "RTLD_GLOBAL", 0)
        names = (
            "libggml-base.so",
            "libggml.so",
            "libggml-cpu.so",
            "libggml-cuda.so",
        )
        for lib_dir in _candidate_lib_dirs(bridge_path):
            for name in names:
                lib_path = lib_dir / name
                if lib_path.is_file():
                    ctypes.CDLL(str(lib_path), mode=mode)

    def last_error(self) -> str:
        raw = self._lib.faster_wan_ggml_last_error()
        if not raw:
            return "ggml bridge returned an unspecified error"
        return raw.decode("utf-8", errors="replace")

    def state_create(
        self,
        *,
        cuda_device: int,
        qtype: gguf.GGMLQuantizationType,
        logical_shape: tuple[int, int],
        raw_weight: torch.Tensor,
        raw_weight_on_device: bool,
    ) -> int:
        n_rows, k_cols = logical_shape
        ptr = int(
            self._lib.faster_wan_ggml_state_create(
                int(cuda_device),
                int(qtype.value),
                int(n_rows),
                int(k_cols),
                ctypes.c_void_p(int(raw_weight.data_ptr())),
                ctypes.c_size_t(int(raw_weight.numel() * raw_weight.element_size())),
                int(raw_weight_on_device),
            )
        )
        if ptr == 0:
            raise RuntimeError(self.last_error())
        return ptr

    def state_destroy(self, ptr: int) -> None:
        if ptr:
            self._lib.faster_wan_ggml_state_destroy(ctypes.c_void_p(ptr))

    def mul_mat(
        self,
        *,
        state_ptr: int,
        x: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        status = int(
            self._lib.faster_wan_ggml_mul_mat(
                ctypes.c_void_p(state_ptr),
                ctypes.c_void_p(int(x.data_ptr())),
                1,
                int(x.shape[0]),
                ctypes.c_void_p(int(out.data_ptr())),
                1,
            )
        )
        if status != 1:
            raise RuntimeError(self.last_error())


@dataclass
class _BridgeState:
    lib: _BridgeLib
    ptr: int
    qtype: gguf.GGMLQuantizationType
    logical_shape: tuple[int, int]
    device_index: int

    def close(self) -> None:
        if self.ptr:
            self.lib.state_destroy(self.ptr)
            self.ptr = 0

    def __del__(self) -> None:  # pragma: no cover - destructor timing is non-deterministic
        self.close()


def _get_bridge() -> _BridgeLib | None:
    global _BRIDGE_LOAD_ATTEMPTED, _BRIDGE_LOAD_ERROR, _BRIDGE_INSTANCE
    if _BRIDGE_LOAD_ATTEMPTED:
        return _BRIDGE_INSTANCE

    _BRIDGE_LOAD_ATTEMPTED = True
    try:
        _BRIDGE_INSTANCE = _BridgeLib()
    except Exception as exc:  # pragma: no cover - exercised on machines without bridge/libs
        _BRIDGE_LOAD_ERROR = str(exc)
        _BRIDGE_INSTANCE = None
    return _BRIDGE_INSTANCE


def bridge_load_error() -> str | None:
    _get_bridge()
    return _BRIDGE_LOAD_ERROR


def clear_ggml_cuda_state(weight: torch.Tensor | None) -> None:
    if weight is None:
        return
    cache = getattr(weight, BRIDGE_CACHE_ATTR, None)
    if cache is None:
        return
    for state in list(cache.values()):
        if isinstance(state, _BridgeState):
            state.close()
    delattr(weight, BRIDGE_CACHE_ATTR)


def _get_cached_state(
    weight: torch.Tensor,
    *,
    device: torch.device,
    qtype: gguf.GGMLQuantizationType,
    logical_shape: tuple[int, int],
) -> _BridgeState:
    bridge = _get_bridge()
    if bridge is None:
        raise RuntimeError(bridge_load_error() or "ggml bridge unavailable")

    cache = getattr(weight, BRIDGE_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(weight, BRIDGE_CACHE_ATTR, cache)

    key = int(device.index or 0)
    cached = cache.get(key)
    if (
        isinstance(cached, _BridgeState)
        and cached.ptr
        and cached.qtype == qtype
        and cached.logical_shape == logical_shape
    ):
        return cached

    if isinstance(cached, _BridgeState):
        cached.close()

    raw_weight = getattr(weight, "data", weight)
    if not isinstance(raw_weight, torch.Tensor):
        raise TypeError(f"expected Tensor-backed GGML weight, got {type(raw_weight)!r}")

    raw_weight = raw_weight.contiguous()
    raw_weight_on_device = (
        raw_weight.device.type == "cuda" and int(raw_weight.device.index or 0) == key
    )
    if raw_weight.device.type == "cuda" and not raw_weight_on_device:
        raw_weight = raw_weight.cpu().contiguous()

    ptr = bridge.state_create(
        cuda_device=key,
        qtype=qtype,
        logical_shape=logical_shape,
        raw_weight=raw_weight,
        raw_weight_on_device=raw_weight_on_device,
    )
    state = _BridgeState(
        lib=bridge,
        ptr=ptr,
        qtype=qtype,
        logical_shape=logical_shape,
        device_index=key,
    )
    cache[key] = state
    return state


def ggml_cuda_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if x.ndim < 2 or x.device.type != "cuda":
        return None
    if not torch.cuda.is_available():
        return None

    qtype = _normalized_qtype(weight)
    if qtype not in SUPPORTED_QTYPES:
        return None

    logical_shape = _logical_shape(getattr(weight, "tensor_shape", weight.shape))
    raw_weight = getattr(weight, "data", weight)
    _validate_quant_weight(raw_weight, qtype, logical_shape)

    x2d = x.reshape(-1, x.shape[-1])
    n_rows, k_cols = logical_shape
    if int(x2d.shape[1]) != k_cols:
        return None

    try:
        state = _get_cached_state(
            weight,
            device=x2d.device,
            qtype=qtype,
            logical_shape=logical_shape,
        )
        x32 = x2d.to(dtype=torch.float32).contiguous()
        out32 = torch.empty((x32.shape[0], n_rows), device=x32.device, dtype=torch.float32)
        state.lib.mul_mat(state_ptr=state.ptr, x=x32, out=out32)
    except Exception as exc:
        _warn_once(f"ggml CUDA fused matmul disabled for this process: {exc}")
        clear_ggml_cuda_state(weight)
        return None

    out = out32.to(dtype=x.dtype)
    if bias is not None:
        out = out + bias.to(device=out.device, dtype=out.dtype)
    return out.reshape(*x.shape[:-1], n_rows)


def load_gguf_tensor(
    gguf_path: str | os.PathLike[str],
    tensor_name: str,
) -> tuple[GGMLTensor, tuple[int, int]]:
    reader = gguf.GGUFReader(str(gguf_path))
    for tensor in reader.tensors:
        if tensor.name != tensor_name:
            continue
        logical_shape = get_orig_shape(reader, tensor.name)
        if logical_shape is None:
            logical_shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            raw = torch.from_numpy(tensor.data)
        wrapped = GGMLTensor(raw, tensor_type=tensor.tensor_type, tensor_shape=logical_shape)
        return wrapped, _logical_shape(logical_shape)
    raise KeyError(f"tensor not found in GGUF: {tensor_name}")


def slice_quant_weight(
    weight: torch.Tensor,
    logical_shape: Sequence[int] | torch.Size,
    *,
    n_rows: int | None = None,
    k_cols: int | None = None,
) -> tuple[GGMLTensor, tuple[int, int]]:
    qtype = _normalized_qtype(weight)
    if qtype is None:
        raise ValueError(f"tensor is not Q4_K/Q6_K: {getattr(weight, 'tensor_type', None)!r}")
    raw_weight = getattr(weight, "data", weight)
    n, k = _logical_shape(logical_shape)
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    n_rows = n if n_rows is None else int(n_rows)
    k_cols = k if k_cols is None else int(k_cols)
    if n_rows <= 0 or n_rows > n:
        raise ValueError(f"n_rows must be in [1, {n}], got {n_rows}")
    if k_cols <= 0 or k_cols > k or k_cols % block_size != 0:
        raise ValueError(
            f"k_cols must be in [1, {k}] and multiple of {block_size}, got {k_cols}"
        )
    packed_cols = (k_cols // block_size) * type_size
    sliced = GGMLTensor(
        raw_weight[:n_rows, :packed_cols].contiguous(),
        tensor_type=qtype,
        tensor_shape=(n_rows, k_cols),
    )
    return sliced, (n_rows, k_cols)


def reference_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    logical_shape: Sequence[int] | torch.Size,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    qtype = _normalized_qtype(weight)
    if qtype is None:
        raise ValueError("reference_linear expects a Q4_K/Q6_K weight")
    raw_weight = getattr(weight, "data", weight)
    ref_weight = dequantize(raw_weight, qtype, logical_shape, dtype=x.dtype)
    if ref_weight.device != x.device:
        ref_weight = ref_weight.to(x.device)
    return F.linear(x, ref_weight, bias)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _bench(fn, *, device: torch.device, warmup: int, iters: int) -> tuple[float, int | None]:
    for _ in range(warmup):
        fn()
    _sync(device)

    peak = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    elapsed = (time.perf_counter() - t0) / max(iters, 1)

    if device.type == "cuda":
        peak = int(torch.cuda.max_memory_allocated(device))
    return elapsed, peak


def run_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", required=True, help="Path to a GGUF file")
    parser.add_argument(
        "--tensor",
        default="blocks.0.cross_attn.k.weight",
        help="Tensor name inside the GGUF file",
    )
    parser.add_argument("--m", type=int, default=17, help="Number of activation rows")
    parser.add_argument("--n-rows", type=int, default=None, help="Optional row slice")
    parser.add_argument("--k-cols", type=int, default=None, help="Optional K slice")
    parser.add_argument(
        "--dtype",
        choices=("fp16", "bf16", "fp32"),
        default="bf16",
        help="Activation dtype used for the reference path",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Target device for correctness/benchmark",
    )
    args = parser.parse_args(argv)

    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.dtype]
    device = torch.device(args.device)

    weight, logical_shape = load_gguf_tensor(args.gguf, args.tensor)
    if args.n_rows is not None or args.k_cols is not None:
        weight, logical_shape = slice_quant_weight(
            weight,
            logical_shape,
            n_rows=args.n_rows,
            k_cols=args.k_cols,
        )

    qtype = _normalized_qtype(weight)
    if qtype is None:
        raise ValueError(f"tensor is not Q4_K/Q6_K: {args.tensor}")

    n_rows, k_cols = logical_shape
    x = torch.randn(args.m, k_cols, dtype=dtype, device=device)
    bias = torch.randn(n_rows, dtype=dtype, device=device)

    ref = reference_linear(x, weight, logical_shape, bias)
    fused = ggml_cuda_linear(x, weight, bias)
    if fused is None:
        print(f"tensor: {args.tensor}")
        print(f"qtype: {getattr(qtype, 'name', qtype)}")
        print(f"logical_shape: {logical_shape}")
        print(f"device: {device}")
        print(f"bridge_error: {bridge_load_error() or 'bridge unavailable or non-CUDA device'}")
        print("cuda_benchmark: skipped")
        return 0

    diff = (fused - ref).to(torch.float32)
    max_abs = float(diff.abs().max().item())
    rms = float(diff.square().mean().sqrt().item())

    print(f"tensor: {args.tensor}")
    print(f"qtype: {getattr(qtype, 'name', qtype)}")
    print(f"logical_shape: {logical_shape}")
    print(f"device: {device}")
    print(f"dtype: {dtype}")
    print(f"max_abs_diff: {max_abs:.8f}")
    print(f"rms_diff: {rms:.8f}")

    if device.type != "cuda":
        print("cuda_benchmark: skipped (CUDA not available on this machine)")
        clear_ggml_cuda_state(weight)
        return 0

    ref_ms, ref_peak = _bench(
        lambda: reference_linear(x, weight, logical_shape, bias),
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    fused_ms, fused_peak = _bench(
        lambda: ggml_cuda_linear(x, weight, bias),
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )

    print(f"baseline_ms: {ref_ms * 1000.0:.3f}")
    print(f"baseline_peak_bytes: {ref_peak}")
    print(f"ggml_ms: {fused_ms * 1000.0:.3f}")
    print(f"ggml_peak_bytes: {fused_peak}")
    clear_ggml_cuda_state(weight)
    return 0


__all__ = [
    "SUPPORTED_QTYPES",
    "bridge_load_error",
    "clear_ggml_cuda_state",
    "ggml_cuda_linear",
    "llama_cpp_env_enabled",
    "load_gguf_tensor",
    "reference_linear",
    "run_cli",
    "slice_quant_weight",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(run_cli())
