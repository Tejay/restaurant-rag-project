import os
from pypdf import PdfReader
from typing import List

class DocumentLoader:
    def __init__(self, data_directory: str):
        self.data_directory = data_directory
    
    def load_pdf_documents(self) -> List[str]:
        """
        Load and extract text from PDF documents in the specified directory.
        
        Returns:
            List of extracted text from PDFs
        """
        documents = []
        for filename in os.listdir(self.data_directory):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(self.data_directory, filename)
                reader = PdfReader(pdf_path)
                
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                
                documents.append(text)
        
        return documents
    
    def preprocess_text(self, documents: List[str]) -> List[str]:
        """
        Basic text preprocessing: lowercase, remove extra whitespaces
        
        Args:
            documents: List of raw text documents
        
        Returns:
            Preprocessed documents
        """
        preprocessed_docs = []
        for doc in documents:
            # Basic preprocessing
            processed_doc = doc.lower().strip()
            preprocessed_docs.append(processed_doc)
        
        return preprocessed_docs
