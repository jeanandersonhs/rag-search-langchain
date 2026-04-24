import os

from langchain_community.vectorstores import PGVector
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from infraestructure.clients.google_ai_client import GoogleAIClient

class IngestUseCase:
    def __init__(self):
        """Ingest documents into the vector database.
        
        This method is responsible for ingesting documents into the vector database. It retrieves the necessary embeddings and language model, connects to the PostgreSQL database, and processes the documents for ingestion.
        
        Args: None """

        DATA_PATH = os.getenv("DATA_PATH", "..      /data/documents")


    def load_documents(self):
        documents = []

        for file in os.listdir(self.DATA_PATH):
            path = os.path.join(self.DATA_PATH, file) # Process the file and extract text

            if file.endswith(".pdf"):
                # Use PyPDF2 or similar to extract text from PDF
                loader = PyPDFLoader(path)
            else:
                # Use a simple text loader for other formats
                loader = TextLoader(path)

            documents.extend(loader.load())
        return documents



    def split_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100)
        return text_splitter.split_documents(documents)
    


    def ingest(self):
        documents = self.load_documents()
        chunks = self.split_documents(documents)

        # Get embeddings and LLM from GoogleAIClient
        embeddings = GoogleAIClient.get_embeddings()

        db = PGVector(
            collection_name="documents",
            connection_string=os.getenv("DATABASE_URL", "postgresql+psycopg://postgres:postgres@db:5432/rag_db"),
            embedding_function=embeddings,
        )
        db.add_documents(chunks)



    
    