// BASIWAN — Pinokio reset script. Canonical cocktailpeanut pattern:
// wipe the env directory, then re-invoke install.js to rebuild from scratch.
// Used when a partial install has corrupted the venv state.
module.exports = {
  run: [
    { method: "fs.rm", params: { path: "env" } },
    { method: "fs.rm", params: { path: "ext/musubi-tuner" } },
    { method: "script.start", params: { uri: "install.js" } },
  ],
};
