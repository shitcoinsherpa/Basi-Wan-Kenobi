// BASIWAN — Pinokio manifest. Matches the cocktailpeanut house style
// (fluxgym / cogstudio canonical shape): hardware-constraint description,
// schema version "3.7", full state-machine menu including Reset+confirm
// and Save Disk Space dedup.
module.exports = {
  version: "3.7",
  title: "basiwan",
  description: "[NVIDIA Only] Dead simple web UI for training Wan2.2 video LoRAs (From 12GB VRAM)",
  icon: "icon.png",
  menu: async (kernel, info) => {
    const installed = info.exists("env");
    const installing = info.running("install.js");
    const starting = info.running("start.js");
    const updating = info.running("update.js");
    const resetting = info.running("reset.js");
    const local = info.local("start.js");

    if (installing) return [{ icon: "fa-solid fa-spinner fa-spin", text: "Installing" }];
    if (updating)   return [{ icon: "fa-solid fa-spinner fa-spin", text: "Updating" }];
    if (resetting)  return [{ icon: "fa-solid fa-spinner fa-spin", text: "Resetting" }];

    const items = [];

    if (starting) {
      if (local && local.url) {
        // [2026-06-09] popout: true was opening the URL in the OS
        // browser via Electron shell.openExternal (per Pinokio
        // browser-popout-surface.js). Omit it so the menu item loads
        // local.url in Pinokio's embedded iframe instead — matches the
        // pinokiofactory/wan + pinokiofactory/forge canonical. Pinokio
        // strips X-Frame-Options + frame-ancestors from the response
        // headers (full.js:3212-3250) so Gradio's CSP can't block the
        // embed. default: true makes "Open Web UI" the auto-launched
        // tab when Start finishes.
        items.push({
          default: true,
          icon: "fa-solid fa-rocket",
          text: "Open Web UI",
          href: local.url,
        });
      }
      items.push({
        icon: "fa-solid fa-terminal",
        text: "Terminal",
        href: "start.js",
      });
    } else if (installed) {
      items.push({
        icon: "fa-solid fa-power-off",
        text: "Start",
        href: "start.js",
      });
    } else {
      items.push({
        icon: "fa-solid fa-plug",
        text: "Install",
        href: "install.js",
      });
    }

    if (installed) {
      items.push({
        icon: "fa-solid fa-folder-open",
        text: "Outputs",
        href: "outputs?fs",
      });
      items.push({
        icon: "fa-solid fa-rotate",
        text: "Update",
        href: "update.js",
      });
      items.push({
        icon: "fa-solid fa-link",
        text: "<div><strong>Save Disk Space</strong><div>Deduplicates redundant library files</div></div>",
        href: "link.js",
      });
      items.push({
        icon: "fa-regular fa-circle-xmark",
        text: "Reset",
        href: "reset.js",
        confirm: "Are you sure you wish to reset the app?",
      });
    }

    return items;
  },
};
