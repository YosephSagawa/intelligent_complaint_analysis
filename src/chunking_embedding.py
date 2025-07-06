from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

# Set up paths
DATA_DIR = Path('../data')
VECTOR_STORE_DIR = Path('../vector_store')
FILTERED_DATA_PATH = DATA_DIR / 'filtered_complaints.csv'
FAISS_INDEX_PATH = VECTOR_STORE_DIR / 'faiss_index.bin'
METADATA_PATH = VECTOR_STORE_DIR / 'metadata.pkl'

# Create vector store directory
VECTOR_STORE_DIR.mkdir(exist_ok=True)

# Step 1: Load the filtered dataset
df = pd.read_csv(FILTERED_DATA_PATH)

# Step 2: Text chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)

# Create chunks and track metadata
chunks = []
metadata = []
for idx, row in df.iterrows():
    complaint_id = idx  # Using index as complaint ID
    product = row['Product_mapped']
    narrative = row['cleaned_narrative']
    split_texts = text_splitter.split_text(narrative)
    for i, chunk in enumerate(split_texts):
        chunks.append(chunk)
        metadata.append({
            'complaint_id': complaint_id,
            'product': product,
            'chunk_id': f"{complaint_id}_{i}"
        })

print(f"Total chunks created: {len(chunks)}")

# Step 3: Generate embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(chunks, batch_size=32, show_progress_bar=True)

# Step 4: Create and save FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, str(FAISS_INDEX_PATH))

# Save metadata
with open(METADATA_PATH, 'wb') as f:
    pickle.dump(metadata, f)

# Save chunks for reference
with open(VECTOR_STORE_DIR / 'chunks.pkl', 'wb') as f:
    pickle.dump(chunks, f)

print(f"FAISS index saved to: {FAISS_INDEX_PATH}")
print(f"Metadata saved to: {METADATA_PATH}")