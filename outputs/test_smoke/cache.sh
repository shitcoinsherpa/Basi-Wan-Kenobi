#!/bin/bash
# BASI WAN K3N0B1 cache precompute for: test_smoke
set -e
cd /mnt/d/Ai/basi-wan-k3n0b1/ext/musubi-tuner
source /usr/bin/activate
/usr/bin/python \
    /mnt/d/Ai/basi-wan-k3n0b1/ext/musubi-tuner/src/musubi_tuner/wan_cache_text_encoder_outputs.py \
    --dataset_config \
    /mnt/d/Ai/basi-wan-k3n0b1/outputs/test_smoke/dataset.toml \
    --t5 \
    /x/t5 \
    --batch_size \
    16
/usr/bin/python \
    /mnt/d/Ai/basi-wan-k3n0b1/ext/musubi-tuner/src/musubi_tuner/wan_cache_latents.py \
    --dataset_config \
    /mnt/d/Ai/basi-wan-k3n0b1/outputs/test_smoke/dataset.toml \
    --vae \
    /x/vae
