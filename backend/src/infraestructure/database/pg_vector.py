from langchain_postgres import PGEngine, PGVectorStore
from dotenv import load_dotenv
from infraestructure.clients.google_ai_client import GoogleAIClient

load_dotenv()  # Load environment variables from .env file
import os

class PgVector_CONFIG:
    """Configuration for PostgreSQL vector store."""
    
    pg_engine = PGEngine.from_connection_string(
        url=os.getenv("DATABASE_URL_POSTGRES", "postgresql://postgres:root@localhost:5432/rag_db?sslmode=disable")
    )

    my_schema = "rag"

    vector_store = PGVectorStore.create_sync(
        engine=pg_engine,
        table_name='documents_rag',
        embedding_service=GoogleAIClient.get_embeddings(),
        schema=my_schema
    )

    @classmethod
    def get_vector_store(cls) -> PGVectorStore:
        """Return the initialized PGVectorStore instance."""
        return cls.vector_store
