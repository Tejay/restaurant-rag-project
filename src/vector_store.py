import faiss
import numpy as np
from typing import List

class VectorStore:
    def __init__(self, embeddings: np.ndarray):
        """
        Initialize Faiss vector index
        
        Args:
            embeddings: Pre-computed document embeddings
        """
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        self.index = index
        self.embeddings = embeddings
    
    def search(self, query_embedding: np.ndarray, top_k: int = 3):
        """
        Perform vector similarity search
        
        Args:
            query_embedding: Embedding of the query
            top_k: Number of top results to retrieve
        
        Returns:
            Indices of most similar documents
        """
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1), 
            top_k
        )
        
        return indices[0]
