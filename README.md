# VisREcall - Visual Reference Search

A powerful image search application that uses CLIP (Contrastive Language-Image Pre-training) to find images based on text descriptions or similar images. Built with Streamlit and Python.

## Features

- 🔍 Search images using text descriptions
- 🖼️ Search using similar images
- 📊 Real-time similarity scoring
- 🚀 Fast and efficient search using CLIP embeddings
- 💾 Persistent storage using ChromaDB

## Prerequisites

- Python 3.9+
- pip (Python package manager)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kmiRVNNT/VisREcall_WP.git
cd VisREcall_WP
```

2. Install the required packages:
```bash
pip install streamlit chromadb torch torchvision open_clip pillow numpy
```

## Usage

1. Place your images in the `streamlit/images` directory

2. Run the Streamlit application:
```bash
cd streamlit
streamlit run strlit.py
```

3. Open your browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

4. Use the search bar to:
   - Enter text descriptions to find similar images
   - Or provide an image path to find visually similar images

## How It Works

The application uses CLIP (Contrastive Language-Image Pre-training) to:
1. Convert images into high-dimensional embeddings
2. Convert text queries into the same embedding space
3. Find the most similar images using cosine similarity

## Project Structure

```
VisREcall_WP/
├── streamlit/
│   ├── strlit.py          # Main Streamlit application
│   ├── images/            # Directory for storing images
│   └── chroma_db/         # ChromaDB database files
└── README.md
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License.
