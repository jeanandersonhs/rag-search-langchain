import os
from typing import List
from langchain_postgres import PGVector, PGVectorStore

from application.dtos.query_dto import AnswerResponse, DocumentSource, QuestionRequest
from infraestructure.clients.google_ai_client import GoogleAIClient
from infraestructure.database.pg_vector import PgVector_CONFIG

from dotenv import load_dotenv
load = load_dotenv()  # Load environment variables from .env file

class QueryUseCase:
    """Use case for handling question-answering with RAG pipeline."""
    
    def __init__(self): 
        """
        Initialize RAG components with LangChain and PostgreSQL vector store.
        Uses shared GoogleAIClient accessors to avoid repeated client creation.
        - Embeddings: GoogleGenerativeAIEmbeddings (from Google AI)
        - LLM: ChatGoogleGenerativeAI (Gemini Pro)
        - Vector Store: PGVector with PostgreSQL
        """
        #nao consegue pegar esse embeding
        self.embeddings = GoogleAIClient.get_embeddings()
        self.llm = GoogleAIClient.get_llm()
        
        # PostgreSQL connection string
        connection_string = os.getenv(
            "DATABASE_URL_POSTGRES",
            "postgresql+psycopg://postgres:postgres@db:5432/rag_db"
        )
        
        self.vector_store = PgVector_CONFIG.get_vector_store()
        
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Retrieve top 3 documents
        )

        # self.qa_chain = .from_chain_type(
        #     llm=self.llm,
        #     chain_type="stuff",
        #     retriever=self.retriever,
        #     return_source_documents=True,
        #     verbose=False
        # )
    
    async def execute(self, request: QuestionRequest) -> AnswerResponse:
        """
        Execute the RAG pipeline to answer a question.
        
        Args:
            request: QuestionRequest containing the query text
            
        Returns:
            AnswerResponse with the generated answer and source documents
            
        Raises:
            ValueError: If the query is empty or invalid
            Exception: If there's an error in the LLM or vector store
        """
        if not request.query or not request.query.strip():
            raise ValueError("Query cannot be empty")
        
        try:
            # Run the RAG pipeline
            result = self.qa_chain({"query": request.query})
            
            # Extract sources from the result
            sources = self._extract_sources(result.get("source_documents", []))
            
            # Build and return the response
            return AnswerResponse(
                question=request.query,
                answer=result.get("result", "No answer generated"),
                sources=sources
            )
        
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Error executing RAG pipeline: {str(e)}") from e
    
    def _extract_sources(self, source_documents: List) -> List[DocumentSource]:
        """
        Extract and format source documents from the retriever output.
        
        Args:
            source_documents: List of document objects from the retriever
            
        Returns:
            List of DocumentSource objects
        """
        sources = []
        
        for doc in source_documents:
            source = DocumentSource(
                content=doc.page_content,
                source=doc.metadata.get("source", "Unknown source") if hasattr(doc, "metadata") else "Unknown source"
            )
            sources.append(source)
        
        return sources
    
