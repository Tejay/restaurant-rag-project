from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class EmbeddingGenerator:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize embedding model
        
        Args:
            model_name: Sentence Transformer model name
        """
        self.model = SentenceTransformer(model_name)
    
    def generate_embeddings(self, documents: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of documents
        
        Args:
            documents: List of preprocessed text documents
        
        Returns:
            NumPy array of document embeddings
        """
        embeddings = self.model.encode(documents, show_progress_bar=True)
        return embeddings
    
    def similarity_search(self, query: str, embeddings: np.ndarray, 
                          documents: List[str], top_k: int = 3) -> List[str]:
        """
        Perform semantic similarity search
        
        Args:
            query: Search query
            embeddings: Pre-computed document embeddings
            documents: Original documents
            top_k: Number of top similar documents to return
        
        Returns:
            Top similar documents
        """
        query_embedding = self.model.encode([query])[0]
        
        # Compute cosine similarities
        similarities = np.dot(embeddings, query_embedding)
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        return [documents[idx] for idx in top_indices]
