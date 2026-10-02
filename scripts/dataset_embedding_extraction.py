"""Extract CLIP and DINOv2 embeddings for the COCO subset images and save them.

Run from the repository root:
    python scripts/dataset_embedding_extraction.py

Reads:  data/images/
Writes: embeddings/clip_embeddings.pt, embeddings/dino_embeddings.pt
"""

import os

import clip
import torch
from transformers import AutoModel, AutoProcessor

from utils import get_clip_embeddings, get_device, get_dino_embeddings

IMAGE_DIR = "data/images"
OUTPUT_DIR = "embeddings"
CLIP_MODEL = "ViT-B/32"
DINO_MODEL = "facebook/dinov2-base"

device = get_device()
print(f"Using device: {device}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- CLIP ---
print("Starting CLIP embedding extraction...")
clip_model, preprocess = clip.load(CLIP_MODEL, device=device)
clip_embeddings = get_clip_embeddings(
    preprocess=preprocess, model=clip_model, device=device, image_dir=IMAGE_DIR
)
torch.save(clip_embeddings, os.path.join(OUTPUT_DIR, "clip_embeddings.pt"))
print("CLIP embeddings saved.")

# --- DINO ---
print("Starting DINO embedding extraction...")
processor = AutoProcessor.from_pretrained(DINO_MODEL)
dino_model = AutoModel.from_pretrained(DINO_MODEL).to(device)
dino_embeddings = get_dino_embeddings(
    processor=processor, model=dino_model, device=device, image_dir=IMAGE_DIR
)
torch.save(dino_embeddings, os.path.join(OUTPUT_DIR, "dino_embeddings.pt"))
print("DINO embeddings saved.")