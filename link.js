// BASIWAN — "Save Disk Space". ONE safe, no-pollution dedup operation:
//
//   fs.link {venv:"env"} — symlinks identical pip packages (torch, diffusers, etc.) from this
//   app's venv into Pinokio's shared /drive/drives/pip store, so multiple apps share a single
//   physical copy. Pure disk win, zero cross-app side effects.
//
// We deliberately do NOT cross-mount basiwan's checkpoints/ into the shared ecosystem "checkpoints"
// drive (the old peers list did this). Basiwan's private model zoo — MOVA-360p (~78GB), the Wan
// experts, the GGUF pairs, S2V — would otherwise pollute the model lists of comfy / forge / fooocus /
// automatic1111, since those apps scan the same shared checkpoints folder. Instead, cross-app reuse
// of the ONE genuinely-shared model (SDXL base) is handled cleanly at DOWNLOAD time by
// tools/ensure_weights.py: it hardlinks an existing peer copy (or one under $BASIWAN_SHARED_DIR)
// rather than re-downloading, with no pollution of any other app. (env_mova is intentionally left
// out of venv dedup — it's a conda env kept isolated by design; see install_mova.js.)
module.exports = {
  run: [
    { method: "fs.link", params: { venv: "env" } },
  ],
};
