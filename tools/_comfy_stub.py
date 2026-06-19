"""Minimal `comfy` stub for STANDALONE (no-GPU, no-ComfyUI) CPU tests of the
gguf_vendor loader. The real runtime uses the real ComfyUI under Pinokio; this
only satisfies ops.py's import-time needs (comfy.ops.manual_cast base classes +
a few helper fns) so gguf_sd_loader — which builds plain GGMLTensor(torch.Tensor)
objects and never exercises the comfy layer classes — can be imported and run in
a bare venv. Inject via: import _comfy_stub; _comfy_stub.install(). TEST-ONLY."""
import sys, types
import torch


def install():
    if "comfy" in sys.modules:
        return
    comfy = types.ModuleType("comfy")
    ops = types.ModuleType("comfy.ops")
    lora = types.ModuleType("comfy.lora")
    mm = types.ModuleType("comfy.model_management")

    class manual_cast:                      # base for GGMLOps
        class Linear(torch.nn.Linear): ...
        class Conv2d(torch.nn.Conv2d): ...
        class Embedding(torch.nn.Embedding): ...
        class LayerNorm(torch.nn.LayerNorm): ...
        class GroupNorm(torch.nn.GroupNorm): ...

    def cast_to(t, dtype=None, device=None, non_blocking=False, copy=False):
        return t.to(device=device, dtype=dtype) if (device or dtype) else t

    ops.manual_cast = manual_cast
    ops.cast_to = cast_to
    lora.calculate_weight = lambda patches, weight, key, *a, **k: weight
    mm.device_supports_non_blocking = lambda device: False
    comfy.ops, comfy.lora, comfy.model_management = ops, lora, mm
    sys.modules.update({"comfy": comfy, "comfy.ops": ops,
                        "comfy.lora": lora, "comfy.model_management": mm})
