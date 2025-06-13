import open_clip
import torch
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel

# Determine the device to use
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Initialize models
danbooru_model = CLIPModel.from_pretrained("cafeai/CLIPDanbooru").to(device)
danbooru_processor = CLIPProcessor.from_pretrained("cafeai/CLIPDanbooru")

# LAION ART CLIP
laion_model, _, laion_preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k', device=device
)

# DANBOORU CLIP
danbooru_model, _, danbooru_preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='danbooru', device=device
)

def embed_image(image_path):
    image = Image.open(image_path).convert("RGB")

    # LAION Embedding
    laion_input = laion_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        laion_emb = laion_model.encode_image(laion_input)

    # DANBOORU Embedding
    danbooru_input = danbooru_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        danbooru_emb = danbooru_model.get_image_features(**danbooru_input)

    return laion_emb.cpu(), danbooru_emb.cpu()

