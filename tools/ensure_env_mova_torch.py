"""Idempotent torch install for env_mova — run by install_mova.js with env_mova's own python.

The plain `pip install torch==2.7.0 ... --index-url cu128` step re-resolved and RE-DOWNLOADED the
3.3GB cu128 wheel on every Pinokio Update (pip doesn't reliably treat the installed local-version
build as 'satisfied' under --index-url). This checks first and ONLY pip-installs when torch /
torchvision / torchaudio are actually missing or the wrong build — so a re-run is a no-op.

Usage:  env_mova/python tools/ensure_env_mova_torch.py [<index-url>]
  <index-url> present  -> NVIDIA build is required (verify torch.version.cuda is set)
  <index-url> omitted  -> CPU / default build is fine
"""
import os
import sys
import subprocess

os.environ["PIP_REQUIRE_VIRTUALENV"] = "0"  # conda env isn't a venv; pip's check would block us
WANT = "2.7.0"
index = sys.argv[1].strip() if len(sys.argv) > 1 and sys.argv[1].strip() else None
need_cuda = bool(index) and "cpu" not in index

ok = False
try:
    import torch
    import torchvision  # noqa: F401
    import torchaudio   # noqa: F401
    cuda = getattr(torch.version, "cuda", None)
    ok = torch.__version__.startswith(WANT) and (cuda is not None if need_cuda else True)
    if ok:
        print(f"[basiwan] env_mova torch already present: {torch.__version__} "
              f"(cuda={cuda}) -- skipping download", flush=True)
    else:
        print(f"[basiwan] env_mova torch is {torch.__version__} (cuda={cuda}); "
              f"want {WANT}{' +cuda' if need_cuda else ''} -- reinstalling", flush=True)
except Exception as e:
    print(f"[basiwan] env_mova torch not importable ({type(e).__name__}); installing", flush=True)

if ok:
    sys.exit(0)

cmd = [sys.executable, "-m", "pip", "install",
       "torch==2.7.0", "torchvision==0.22.0", "torchaudio==2.7.0"]
if index:
    cmd += ["--index-url", index]
print("[basiwan] " + " ".join(cmd), flush=True)
sys.exit(subprocess.call(cmd))
