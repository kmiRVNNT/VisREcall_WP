from PIL import Image
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction 
# OpenAI's CLIP embedding function
from chromadb.utils.data_loader import ImageLoader
import numpy as np
from tqdm import tqdm
import os

image_loader = ImageLoader()

client = chromadb.Client()
embedding_function = OpenCLIPEmbeddingFunction()
data_loader = ImageLoader()

collection = client.get_or_create_collection(
  name="multimodal_collection",
  embedding_function=embedding_function,
  data_loader=data_loader
)

def add_images_to_collection(folder_path):
  image_files = [os.path.join(folder_path, image_name) for image_name in os.listdir(folder_path) 
                 if os.path.isfile(os.path.join(folder_path, image_name)) and image_name.endswith(('.png', '.jpg', '.jpeg'))]
  
  for image_path in tqdm(image_files, desc="Finding references within a collection"):
    try:
      image = np.array(Image.open(image_path))
      collection.add(
        ids=[os.path.basename(image_path)],
        images=[image]
      )

    except Exception as e:
      print(f"Error adding {image_path}: {e}")
    
image_folder_path = r"C:\Users\jason\Desktop\streamlit\images"
add_images_to_collection(image_folder_path)
      
      
      
      
      




