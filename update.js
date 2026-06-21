// BASIWAN — Pinokio update script. git pull, then re-run the IDEMPOTENT install
// so new/changed deps land WITHOUT wiping the venv.
//
// We deliberately do NOT `fs.rm env` here. install.js's steps are all no-ops
// when already satisfied — critically torch.js's pinned `uv pip install
// torch==2.7.0` is AUDITED, not re-downloaded — so an update no longer re-pulls
// the ~2.5GB CUDA torch wheel every time. `uv pip install -r requirements.txt`
// installs only what changed; the musubi clone/checkout and ensure_weights are
// already idempotent/resumable. For a true from-scratch rebuild, use Reset
// (reset.js), which is the place that wipes the venv.
//
// musubi-tuner stays PINNED to the SHA install.js checks out; the pin is
// preserved across updates because install.js owns the version selection.
module.exports = {
  run: [
    { method: "shell.run", params: { message: "git pull --ff-only" } },
    { method: "script.start", params: { uri: "install.js" } },
  ],
};
