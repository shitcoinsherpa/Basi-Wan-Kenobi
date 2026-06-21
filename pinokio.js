// BASIWAN — Pinokio manifest. Matches the cocktailpeanut house style
// (fluxgym / cogstudio canonical shape): hardware-constraint description,
// schema version "3.7", full state-machine menu including Reset+confirm
// and Save Disk Space dedup.
//
// PREREQUISITES (`pre`): Pinokio shows this list as a guided Prerequisites
// screen BEFORE the install screen (PINOKIO.md §"Display prerequisite apps").
// `requires: { bundle: "ai" }` in install.js already pulls Pinokio's bundled
// toolchain — which on Windows INCLUDES Visual Studio — so for most users the
// MSVC C++ compiler (cl.exe) is already present. But the Wan GGUF path compiles
// a small CUDA kernel on first generation (tools/build_kernel.py) and that needs
// cl.exe; on machines where the bundled VS is absent/partial it fails (non-fatal:
// it retries each gen). So we surface VS C++ Build Tools as a Windows-only `pre`
// item — a one-click guided install before app install for anyone the bundle
// didn't cover. Linux/macOS need no such prereq (gcc/clang from the bundle).
const _WIN = process.platform === "win32";
module.exports = {
  version: "3.7",
  title: "BASI WAN KENOBI",
  description: "[NVIDIA Only] Wan2.2 video Studio (text/image-to-video, restyle, keyframe, talking-character S2V) + LoRA Gym + MOVA joint audio+video. From 12GB VRAM.",
  icon: "icon.png",
  pre: _WIN ? [{
    icon: "icon.png",
    title: "Visual Studio C++ Build Tools (Windows)",
    description: "Provides the MSVC C++ compiler (cl.exe) used to build the optimized CUDA kernel for Wan video generation. Pinokio's bundled toolchain usually already includes this — install only if generation reports a missing compiler. When installing, check the \"Desktop development with C++\" workload.",
    href: "https://visualstudio.microsoft.com/visual-cpp-build-tools/"
  }] : undefined,
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
        // No popout: load local.url in Pinokio's embedded iframe (popout
        // would open it in the OS browser via Electron shell.openExternal).
        // Matches the pinokiofactory/wan + forge canonical. Pinokio
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
