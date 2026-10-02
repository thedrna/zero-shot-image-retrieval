# Zero-Shot Image Retrieval Using Vision-Language Models

This project implements a zero-shot image retrieval system using CLIP and DINO. The system compares generated images with Stable Diffusion against a dataset of reference images to find visual similarities without requiring explicit training on the target dataset.

## Project Structure

### Directories

- `data/images/`: Contains the COCO dataset subset images used as reference data
- `embeddings/`: Stores pre-computed embeddings for both dataset and generated images
- `generated_images/`: Contains AI-generated images
  - `v1-5/`: Images generated using Stable Diffusion v1.5
  - `v2-1/`: Images generated using Stable Diffusion v2.1
- `scripts/`: Python scripts for dataset creation, embedding extraction, and similarity analysis
- `notebooks`: Python notebook for generating images

### Scripts

- `dataset_creation.py`: Creates a subset of COCO dataset with images containing 2-3 unique object categories
- `dataset_embedding_extraction.py`: Extracts CLIP and DINO embeddings from the dataset images
- `generated_images_embedding_extraction.py`: Extracts embeddings from AI-generated images
- `utils.py`: Contains utility functions for embedding extraction and similarity calculation
- `text_to_image.ipynb`: Python notebook for generating images from prompts using SD 1-5 and SD 2-1

### Other Files

- `prompts.txt`: Text prompts used to generate images with Stable Diffusion
- `requirements.txt`: Required Python packages for the project
- `Results.xlsx`: Manual evaluation of retrived results

## Setup and Usage

### Environment Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

1. Install requirements:

```bash
pip install -r requirements.txt
```

### Dataset Creation

To recreate the COCO dataset subset, first run:

```bash
mkdir -p coco_data/annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip -d coco_data/
```

To download the 2017 COCO val set annotations and URLs, then run:

```bash
python scripts/dataset_creation.py
```

This will download a subset of COCO images containing 2-3 unique object categories.

### Embedding Extraction

To extract embeddings from the dataset images:

```bash
python scripts/dataset_embedding_extraction.py
```

To extract embeddings from generated images:

```bash
python scripts/generated_images_embedding_extraction.py
```

These will extract both DINO and CLIP embeddings of dataset and generated images.

### Image Generation

Simply run `text-to-image.ipynb` notebook using Jupyter Notebook or Google Colab (for better GPU access).
Images can be generated using Stable Diffusion models with the prompts provided in `prompts.txt`, or even by writing new prompts. The generated images should be placed in their respective directories:

- `generated_images/v1-5/` for Stable Diffusion v1.5 images
- `generated_images/v2-1/` for Stable Diffusion v2.1 images

After generating images, remember to generate their embedding as well.
