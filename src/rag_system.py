import os
from typing import List
import openai
from dotenv import load_dotenv

from src.data_loader import DocumentLoader
from src.embedding_generator import EmbeddingGenerator
from src.vector_store import VectorStore

class RestaurantRAG:
    def __init__(self, data_directory: str):
        """
        Initialize RAG system
        
        Args:
            data_directory: Path to restaurant guide documents
        """
        load_dotenv()  # Load environment variables
        
        # Configure OpenAI client with new syntax
        self.client = openai.OpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Load and preprocess documents
        self.loader = DocumentLoader(data_directory)
        self.documents = self.loader.load_pdf_documents()
        self.processed_docs = self.loader.preprocess_text(self.documents)
        
        # Generate embeddings
        self.embedding_generator = EmbeddingGenerator()
        self.embeddings = self.embedding_generator.generate_embeddings(
            self.processed_docs
        )
        
        # Create vector store
        self.vector_store = VectorStore(self.embeddings)
    
    def query(self, query: str, top_k: int = 3) -> str:
        """
        Perform RAG query
        
        Args:
            query: User's natural language query
            top_k: Number of context documents to retrieve
        
        Returns:
            AI-generated response
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.model.encode([query])[0]
        
        # Retrieve similar documents
        similar_doc_indices = self.vector_store.search(query_embedding, top_k)
        context_docs = [self.documents[idx] for idx in similar_doc_indices]
        
        # Prepare prompt with context
        full_prompt = f"""
        Context Documents:
        {' '.join(context_docs)}
        
        User Query: {query}
        
        Please provide a detailed and specific answer based on the context.
        """
        
        # Generate response using OpenAI (new syntax)
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful restaurant guide assistant."},
                {"role": "user", "content": full_prompt}
            ]
        )
        
        return response.choices[0].message.content
