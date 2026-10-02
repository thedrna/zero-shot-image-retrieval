"""Helper functions for embedding extraction and similarity search."""

import os

import torch
from PIL import Image
from tqdm import tqdm

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def get_device():
    """Return the best available device: CUDA GPU, Apple MPS or CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def list_image_files(image_dir):
    """Return the sorted image filenames (png/jpg/jpeg) in a directory.

    Raises an error if none are found, so a wrong folder or file type
    fails loudly instead of silently producing empty embeddings.
    """
    image_files = sorted(
        f for f in os.listdir(image_dir) if f.lower().endswith(IMAGE_EXTENSIONS)
    )
    if not image_files:
        raise FileNotFoundError(f"No {IMAGE_EXTENSIONS} images found in '{image_dir}'")
    print(f"Found {len(image_files)} images in '{image_dir}'")
    return image_files


def get_dino_embeddings(processor, model, device, image_dir):
    """Compute one DINOv2 embedding per image.

    The embedding is the mean of all output tokens (CLS + patch tokens).
    Returns a dict {filename: tensor}.
    """
    embeddings = {}
    for fname in tqdm(list_image_files(image_dir), desc="DINO embeddings"):
        img_path = os.path.join(image_dir, fname)
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            pooled = last_hidden_state.mean(dim=1).squeeze().cpu()
        embeddings[fname] = pooled
    return embeddings


def get_clip_embeddings(preprocess, model, device, image_dir):
    """Compute one CLIP image embedding per image.

    Returns a dict {filename: tensor}.
    """
    embeddings = {}
    for fname in tqdm(list_image_files(image_dir), desc="CLIP embeddings"):
        img_path = os.path.join(image_dir, fname)
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image).squeeze().cpu()
        embeddings[fname] = image_features
    return embeddings


def cosine_similarity(a, b):
    """Cosine similarity between every row of `a` and every row of `b`."""
    a_norm = a / a.norm(dim=1, keepdim=True)
    b_norm = b / b.norm(dim=1, keepdim=True)
    return torch.mm(a_norm, b_norm.t())


def find_top_matches(gen_embeds, dataset_embeds, top_k=2):
    """Find the `top_k` dataset images most similar to each generated-image embedding.

    Args:
        gen_embeds: tensor of shape (N, D), one row per generated image.
        dataset_embeds: dict {filename: tensor of shape (D,)}.
        top_k: number of matches to return per row.

    Returns:
        A list with one (filenames, scores) pair per row of `gen_embeds`,
        best match first.
    """
    dataset_fnames = sorted(dataset_embeds.keys())
    dataset_matrix = torch.stack([dataset_embeds[f] for f in dataset_fnames])

    scores = cosine_similarity(gen_embeds, dataset_matrix)
    topk = torch.topk(scores, top_k, dim=1)
    top_indices = topk.indices
    top_scores = topk.values

    top_fnames = [[dataset_fnames[i] for i in row] for row in top_indices.tolist()]
    top_scores = top_scores.tolist()

    return list(zip(top_fnames, top_scores))