// BASIWAN — Save Disk Space (fs.link peers). Deduplicates model weights
// against the cocktailpeanut ecosystem so users who already have ComfyUI /
// Forge / Fooocus don't double-store Wan2.2 base weights or VAE files.
//
// Drives:
//   - checkpoints/Wan2.2-T2V-A14B/         (~28 GB experts + VAE + text encoder)
//   - checkpoints/Wan2.2-T2V-A14B-GGUF/    (Q4_K_M / Q8_0 variants if using quantized path)
//   - checkpoints/lightning_lora/          (Wan2.2-Lightning 4-step LoRA)
//   - checkpoints/taehv/                   (TAEHV tiny VAE)
//
// Peers list mirrors fluxgym / cogstudio canonical: pinokiofactory + cocktailpeanutlabs.
module.exports = {
  run: [
    {
      method: "fs.link",
      params: {
        drive: {
          checkpoints: "checkpoints",
        },
        peers: [
          "pinokiofactory/comfy",
          "pinokiofactory/stable-diffusion-webui-forge",
          "cocktailpeanutlabs/comfyui",
          "cocktailpeanutlabs/fooocus",
          "cocktailpeanutlabs/automatic1111",
        ],
      },
    },
  ],
};
