"""Extract CLIP and DINOv2 embeddings for the Stable Diffusion generated images.

Run from the repository root:
    python scripts/generated_images_embedding_extraction.py

Reads:  generated_images/v1-5/ and generated_images/v2-1/
Writes: embeddings/{clip,dino}_embeddings_v1-5.pt
        embeddings/{clip,dino}_embeddings_v2-1.pt
"""

import os

import clip
import torch
from transformers import AutoModel, AutoProcessor

from utils import get_clip_embeddings, get_device, get_dino_embeddings

SD_VERSIONS = ["v1-5", "v2-1"]  # one folder of generated images per Stable Diffusion version
IMAGE_ROOT = "generated_images"
OUTPUT_DIR = "embeddings"
CLIP_MODEL = "ViT-B/32"
DINO_MODEL = "facebook/dinov2-base"

device = get_device()
print(f"Using device: {device}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- DINO ---
processor = AutoProcessor.from_pretrained(DINO_MODEL)
dino_model = AutoModel.from_pretrained(DINO_MODEL).to(device)

for version in SD_VERSIONS:
    print(f"Processing DINO embeddings for {IMAGE_ROOT}/{version}...")
    embeddings = get_dino_embeddings(
        processor=processor,
        model=dino_model,
        device=device,
        image_dir=os.path.join(IMAGE_ROOT, version),
    )
    torch.save(embeddings, os.path.join(OUTPUT_DIR, f"dino_embeddings_{version}.pt"))

# --- CLIP ---
clip_model, preprocess = clip.load(CLIP_MODEL, device=device)

for version in SD_VERSIONS:
    print(f"Processing CLIP embeddings for {IMAGE_ROOT}/{version}...")
    embeddings = get_clip_embeddings(
        preprocess=preprocess,
        model=clip_model,
        device=device,
        image_dir=os.path.join(IMAGE_ROOT, version),
    )
    torch.save(embeddings, os.path.join(OUTPUT_DIR, f"clip_embeddings_{version}.pt"))

print("All embeddings processed and saved successfully.")