// BASIWAN — Pinokio update script. Canonical cocktailpeanut pattern:
// git pull → wipe env → re-run torch.js → reinstall deps. This is the
// safest update path; reuses the install logic for dep resolution.
//
// Note: musubi-tuner stays PINNED to the SHA install.js checks out. The
// pin is preserved across updates because install.js owns the version
// selection. Bumping requires a deliberate edit to install.js.
module.exports = {
  run: [
    { method: "shell.run", params: { message: "git pull --ff-only" } },
    { method: "fs.rm", params: { path: "env" } },
    { method: "script.start", params: { uri: "install.js" } },
  ],
};
