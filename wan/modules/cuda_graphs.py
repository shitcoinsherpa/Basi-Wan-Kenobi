from __future__ import annotations

import os
from dataclasses import dataclass

import torch

# torchao 0.13's FP8 `_scaled_mm` (cublasLt path) fails with
# CUBLAS_STATUS_EXECUTION_FAILED on RTX 4090 (sm_89) when invoked on a non-default
# CUDA stream unless cuBLASLt has an explicitly-sized workspace. Bumping the
# workspace size via env var before any cuBLAS init fixes it (probe:
# 16 MiB is sufficient at the Lightning shape; we go to 32 to be safe).
# Setting the env after torch imports is a no-op for the existing context, so we
# call torch._C._cuda_clearCublasWorkspaces() too to force re-init at our size.
os.environ.setdefault("CUBLASLT_WORKSPACE_SIZE", str(16 * 1024 * 1024))
if not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
try:
    torch._C._cuda_clearCublasWorkspaces()  # type: ignore[attr-defined]
except Exception:
    pass


def model_has_float8_weights(model) -> bool:
    """Detect torchao Float8Tensor weights in `model`.

    torchao FP8 + CUDA Graphs is a known-incompatible combination upstream
    (pytorch/ao#567, open since July 2024). Capture works but replay aborts
    silently because the Float8Tensor python-level __torch_function__ dispatch
    isn't graph-replayable. Callers should skip capture when this returns True
    and wait for the upstream fix (or use BF16 weights instead).
    """
    try:
        from torchao.quantization import Float8Tensor as _F8T
    except ImportError:
        try:
            from torchao.float8 import Float8Tensor as _F8T  # torchao <0.13 fallback
        except ImportError:
            return False
    for p in model.parameters():
        if isinstance(p, _F8T) or isinstance(getattr(p, "data", None), _F8T):
            return True
        # Newer torchao wraps Float8Tensor inside an outer subclass — check by class name too.
        if "Float8" in type(p).__name__ or "Float8" in type(getattr(p, "data", p)).__name__:
            return True
    return False


@dataclass(slots=True)
class WanForwardCudaGraph:
    """Captured fixed-shape WanModel.forward replay state."""

    graph: torch.cuda.CUDAGraph
    static_latent: torch.Tensor
    static_timestep: torch.Tensor
    static_outputs: list[torch.Tensor]

    @classmethod
    def capture(
        cls,
        model,
        *,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        context: list[torch.Tensor],
        seq_len: int,
        amp_dtype: torch.dtype,
        warmup_iters: int = 1,
    ) -> "WanForwardCudaGraph":
        if latent.device.type != "cuda" or timestep.device.type != "cuda":
            raise ValueError("CUDA graph capture requires CUDA latent and timestep tensors.")
        if timestep.shape != (1,):
            raise ValueError(f"expected timestep shape (1,), got {tuple(timestep.shape)}")

        device = latent.device
        static_latent = torch.empty_like(latent)
        static_timestep = torch.empty_like(timestep)
        static_latent.copy_(latent)
        static_timestep.copy_(timestep)
        static_inputs = [static_latent]

        def forward_once() -> list[torch.Tensor]:
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype):
                return model(static_inputs, t=static_timestep, context=context, seq_len=seq_len)

        warmup_stream = torch.cuda.Stream(device=device)
        current_stream = torch.cuda.current_stream(device=device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream):
            for _ in range(max(1, warmup_iters)):
                forward_once()
        current_stream.wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_outputs = forward_once()
        return cls(
            graph=graph,
            static_latent=static_latent,
            static_timestep=static_timestep,
            static_outputs=static_outputs,
        )

    def replay(self, latent: torch.Tensor, timestep: torch.Tensor) -> list[torch.Tensor]:
        if latent.shape != self.static_latent.shape:
            raise ValueError(
                f"latent shape changed from {tuple(self.static_latent.shape)} "
                f"to {tuple(latent.shape)}"
            )
        if timestep.shape != self.static_timestep.shape:
            raise ValueError(
                f"timestep shape changed from {tuple(self.static_timestep.shape)} "
                f"to {tuple(timestep.shape)}"
            )
        self.static_latent.copy_(latent, non_blocking=True)
        self.static_timestep.copy_(timestep, non_blocking=True)
        self.graph.replay()
        return self.static_outputs
