import torch
import json
from utils import find_top_matches
from tqdm import tqdm

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load all dataset embeddings
clip_dataset_embeds = torch.load("embeddings/clip_embeddings.pt")
dino_dataset_embeds = torch.load("embeddings/dino_embeddings.pt")

# Load generated embeddings
clip_embeds_v1 = torch.load("embeddings/clip_embeddings_v1-5.pt")
clip_embeds_v2 = torch.load("embeddings/clip_embeddings_v2-1.pt")
dino_embeds_v1 = torch.load("embeddings/dino_embeddings_v1-5.pt")
dino_embeds_v2 = torch.load("embeddings/dino_embeddings_v2-1.pt")

with open("prompts.txt", "r") as f:
    prompts = [line.strip() for line in f.readlines() if line.strip()]

results = {}

for i, prompt in enumerate(tqdm(prompts)):
    idx = i + 1
    key = f'{idx:02d}_{prompt}'
    results[key] = {}

    results[key]["clip_v1-5"] = find_top_matches(clip_embeds_v1[f'{idx:02d}_{prompt}.png'].unsqueeze(0), clip_dataset_embeds)
    results[key]["clip_v2-1"] = find_top_matches(clip_embeds_v2[f'{idx:02d}_{prompt}.png'].unsqueeze(0), clip_dataset_embeds)
    results[key]["dino_v1-5"] = find_top_matches(dino_embeds_v1[f'{idx:02d}_{prompt}.png'].unsqueeze(0), dino_dataset_embeds)
    results[key]["dino_v2-1"] = find_top_matches(dino_embeds_v2[f'{idx:02d}_{prompt}.png'].unsqueeze(0), dino_dataset_embeds)

with open("retrieved_similar_images.json", "w") as f:
    json.dump(results, f, indent=2)

print("Retrieval complete. Results saved to retrieved_similar_images.json")