from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import numpy as np
import pdfplumber
import torch
import json
import os
import gc

load_dotenv()

torch.set_num_threads(1)

# Provides method to extract vectors from text
class EmbeddingService:

    def __init__(self):
        print("Booting AI Model into RAM...")
        token = os.getenv('HF_TOKEN')
        cache_dir = './model_cache'
        self.model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_dir, token=token)
        print("AI Model loaded.")

    def embed(self, text: str):
        with torch.inference_mode():
            vector = self.model.encode(text, normalize_embeddings=True)
        return vector

# The main in-memory database
class LiteVectorDB:

    #initializes the required variables on creation
    def __init__(self, dimension: int, max_capacity: int = 10000):
        
        self.dimension = dimension
        self.max_capacity = max_capacity
        
        seed_vectors = np.load('seed_vectors.npy')
        self.current_count = seed_vectors.shape[0]
        self.vectors = np.zeros((max_capacity, dimension), dtype=np.float32)
        self.vectors[:self.current_count] = seed_vectors

        with open('seed_metadata.json', 'r') as f:
            seed_metadata = json.load(f)
        self.metadata = {int(k): v for k, v in seed_metadata.items()}
        
        gc.collect()
        print(f"Database fully initialized with {self.current_count} records.")

    #adds the embedding to the db while updating the metadata also
    def add_record(self, text: str, embedding: np.ndarray):

        if embedding.shape[0] != self.dimension:
            raise ValueError(f"Vector dimension mismatch. Expected dimension {self.dimension}")
        
        if self.current_count >= self.vectors.shape[0]:
            raise MemoryError("Database capacity reached.")
        
        self.vectors[self.current_count] = embedding
        self.metadata[self.current_count] = text

        self.current_count+=1

    #searches the db for semantically related texts using cosine similarity
    #cosine similarity = (A.B)/(||A||x||B||)
    #due to normalizing the embeddings when adding them to the db, ||A|| and ||B|| equal to 1
    #hence the new cosine similarity formula becomes: cosine_similarities = A.B
    def search(self, query_vector: np.ndarray, top_k: int):
        if self.current_count > 4000:
            valid_vectors = self.vectors[3999:self.current_count] # A
        else:
            valid_vectors = self.vectors[:self.current_count] # A

        #dot_products = np.dot(valid_vectors, query_vector) -> computes (A.B) where B = query_vector

        # query_mag = np.linalg.norm(query_vector) -> ||B||
        # db_mag = np.linalg.norm(valid_vectors, axis= 1) -> ||A||

        # cosine_similarities = dot_products/ (db_mag * query_mag) -> computes (A.B)/(||A|| x|B||)

        cosine_similarities = np.dot(valid_vectors, query_vector)

        # sorts the cosine_similarities array using argsort which is then reversed to be in descending order and the top_k indices are sliced
        top_indices = np.argsort(cosine_similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "text": self.metadata[int(idx)],
                "score": float(cosine_similarities[idx])
            })

        return results


class ChunkingEngine:
    def __init__(self, chunk: int = 100):
        self.chunk = chunk
        self.chunks = []
        self.page_chunk = []
    
    def generate_chunks(self, file):

        with pdfplumber.open(file) as pdf:
            for pg in pdf.pages:
                content = pg.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False, use_text_flow=False, horizontal_ltr=True, vertical_ttb=True, extra_attrs=[])
                for word in content:
                    if len(self.page_chunk) >= self.chunk:
                        self.chunks.append(tuple(self.page_chunk))
                        self.page_chunk.clear()
                    self.page_chunk.append(word['text'])
            self.chunks.append(tuple(self.page_chunk))
        
        return self.chunks
    
    def stringify_chunks(self):
        self.stringified = []
        for _ in self.chunks:
            self.stringified.append(' '.join(_))
        
        return self.stringified