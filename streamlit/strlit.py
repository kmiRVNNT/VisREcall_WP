import streamlit as st
import chromadb
import os
from pathlib import Path
import torch
from PIL import Image
import numpy as np
import open_clip

# Determine the device to use
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Initialize models
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k', device=device
)

def embed_image(image_path):
    image = Image.open(image_path).convert("RGB")
    
    # Process image
    inputs = preprocess(image).unsqueeze(0).to(device)
    
    # Get embeddings
    with torch.no_grad():
        image_features = model.encode_image(inputs)
        # Normalize the features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    return image_features.cpu()

def embed_text(text):
    # Tokenize and encode text
    text_tokens = open_clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        # Normalize the features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu()

def compute_similarity(query_emb, image_emb):
    # Convert to numpy if they're tensors
    if isinstance(query_emb, torch.Tensor):
        query_emb = query_emb.numpy()
    if isinstance(image_emb, torch.Tensor):
        image_emb = image_emb.numpy()
    
    # Compute cosine similarity
    similarity = np.dot(query_emb.flatten(), image_emb.flatten())
    return float(similarity)

# Set up paths relative to the script location
script_dir = Path(__file__).parent
db_path = script_dir / "chroma_db"
images_path = script_dir / "images"

# Create directories if they don't exist
db_path.mkdir(exist_ok=True)
images_path.mkdir(exist_ok=True)

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=str(db_path))

# Delete existing collection to start fresh
try:
    client.delete_collection("image_collection")
except:
    pass

# Create collection with explicit dimension and cosine similarity
collection = client.get_or_create_collection(
    name="image_collection",
    metadata={"hnsw:space": "cosine"},
    embedding_function=None  # We'll handle embeddings ourselves
)

# Store image embeddings in memory for direct comparison
image_embeddings = {}

# Add existing images to the collection if they're not already there
existing_images = list(images_path.glob("*"))
if existing_images:
    # Add all images
    for img in existing_images:
        try:
            # Get embeddings
            emb = embed_image(str(img))
            
            # Store embedding in memory
            image_embeddings[img.name] = emb
            
            # Flatten embedding before adding to collection
            flattened_emb = emb.numpy().flatten().tolist()
            
            # Add to collection
            collection.add(
                embeddings=[flattened_emb],
                ids=[img.name],
                metadatas=[{"source": "local", "filename": img.name}]
            )
            st.sidebar.success(f"Added {img.name} to the collection")
        except Exception as e:
            st.sidebar.error(f"Error processing {img.name}: {str(e)}")

st.title("Reference Search")
query = st.text_input("Enter a search query:")

if st.button("Search"):
    if not query:
        st.warning("Please enter a search query")
    else:
        try:
            # Get query embedding
            if os.path.exists(query):
                # If query is an image path
                query_emb = embed_image(query)
                st.write("Searching with image query...")
            else:
                # If query is text
                query_emb = embed_text(query)
                st.write("Searching with text query...")
            
            # Debug information
            st.write(f"Query embedding shape: {query_emb.shape}")
            st.write(f"Total images in collection: {len(image_embeddings)}")
            
            # Compute similarities directly
            similarities = []
            for img_name, img_emb in image_embeddings.items():
                similarity = compute_similarity(query_emb, img_emb)
                similarities.append((img_name, similarity))
            
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Display top results
            for img_name, similarity in similarities[:4]:
                image_path = images_path / img_name
                if image_path.exists():
                    st.image(str(image_path), caption=f"{img_name} (Similarity: {similarity:.2f})")
                else:
                    st.warning(f"Image not found: {img_name}")
                        
        except Exception as e:
            st.error(f"Error during search: {str(e)}")
            st.write("Full error details:", e)

# Add a section to show available images
st.sidebar.title("Available Images")
if images_path.exists():
    image_files = list(images_path.glob("*"))
    if image_files:
        st.sidebar.write(f"Found {len(image_files)} images:")
        for img in image_files:
            st.sidebar.write(f"- {img.name}")
    else:
        st.sidebar.info("No images found in the images directory")
else:
    st.sidebar.warning("Images directory not found")





