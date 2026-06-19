from sentence_transformers import SentenceTransformer
import numpy as np
import os
import gc
from dotenv import load_dotenv

load_dotenv()

# Global placeholders
_model = None
_vectors = None

def get_resources():
    """Lazy loader: Initializes model and vectors only on first request."""
    global _model, _vectors
    
    if _model is None:
        print("Lazy loading model and vectors into RAM...")
        token = os.getenv('HF_TOKEN')
        cache_dir = './model_cache'
        
        # Initialize model
        _model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_dir, token=token)
        
        # Load pre-computed vectors
        _vectors = np.load('seed_vectors.npy')
        
        # Cleanup
        gc.collect()
        print("Initialization complete.")
        
    return _model, _vectors

# Provides method to extract vectors from text
class EmbeddingService:
    def embed(self, text: str):
        # Fetch the model lazily
        model, _ = get_resources()
        vector = model.encode(text, normalize_embeddings=True)
        return vector

# The main in-memory database
class LiteVectorDB:

    #initializes the required variables on creation
    def __init__(self, dimension: int, max_capacity: int = 10000):
        
        self.dimension = dimension
        self.max_capacity = max_capacity
        
        self.current_count = 0

        self.vectors = np.zeros((max_capacity, dimension), dtype= np.float32)

        self.metadata = {}

    def load_seed_data(self):
        _, seed_vectors = get_resources()
        self.vectors[:seed_vectors.shape[0]] = seed_vectors
        self.current_count = seed_vectors.shape[0]
        print(f"Loaded {self.current_count} seed vectors into DB instance.")

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