"""Create the reference image set: a subset of the COCO 2017 validation images
that contain 2 or 3 unique object categories.

Run from the repository root:
    python scripts/dataset_creation.py

If the COCO annotations are not found, they are downloaded first (about 240 MB;
only the validation annotations are extracted).

Writes: data/images/*.jpg            the selected images
        data/subset_annotations.json their annotations

The first NUM_IMAGES matching images in annotation-file order are selected, so
the subset is the same every time the script is run.

COCO images come from Flickr and keep their original licences, see
https://cocodataset.org/#termsofuse
"""

import json
import os
import zipfile

import requests
from pycocotools.coco import COCO
from tqdm import tqdm

COCO_DIR = "coco_data"  # temporary download folder (git-ignored, can be deleted afterwards)
ANNOTATIONS_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANNOTATIONS_MEMBER = "annotations/instances_val2017.json"  # path inside the zip
ANN_PATH = os.path.join(COCO_DIR, ANNOTATIONS_MEMBER)
IMG_BASE_URL = "http://images.cocodataset.org/val2017/"

SAVE_DIR = "data/images"
SUBSET_ANNOTATIONS_PATH = "data/subset_annotations.json"

NUM_IMAGES = 30
MIN_CATEGORIES = 2
MAX_CATEGORIES = 3


def download_annotations():
    """Download the COCO annotations zip and extract the validation annotations."""
    os.makedirs(COCO_DIR, exist_ok=True)
    zip_path = os.path.join(COCO_DIR, "annotations_trainval2017.zip")

    print(f"Downloading COCO annotations to {zip_path} ...")
    with requests.get(ANNOTATIONS_ZIP_URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))

    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(ANNOTATIONS_MEMBER, COCO_DIR)


if not os.path.exists(ANN_PATH):
    download_annotations()

os.makedirs(SAVE_DIR, exist_ok=True)

coco = COCO(ANN_PATH)

# Select the first NUM_IMAGES images that contain 2-3 unique object categories
selected_imgs = []
for img_id in tqdm(coco.getImgIds(), desc="Selecting images"):
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
    categories = {ann["category_id"] for ann in anns}
    if MIN_CATEGORIES <= len(categories) <= MAX_CATEGORIES:
        selected_imgs.append(img_id)
    if len(selected_imgs) == NUM_IMAGES:
        break

# Download the selected images and collect their annotations
subset_data = []
for img_id in tqdm(selected_imgs, desc="Downloading images"):
    img_info = coco.loadImgs(img_id)[0]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
    category_names = sorted({coco.loadCats(ann["category_id"])[0]["name"] for ann in anns})

    response = requests.get(IMG_BASE_URL + img_info["file_name"], timeout=60)
    response.raise_for_status()
    with open(os.path.join(SAVE_DIR, img_info["file_name"]), "wb") as f:
        f.write(response.content)

    subset_data.append({
        "file_name": img_info["file_name"],
        "categories": category_names,
        "annotations": anns,
    })

with open(SUBSET_ANNOTATIONS_PATH, "w") as f:
    json.dump(subset_data, f, indent=2)

print(f"Saved {len(subset_data)} images to {SAVE_DIR} and annotations to {SUBSET_ANNOTATIONS_PATH}")