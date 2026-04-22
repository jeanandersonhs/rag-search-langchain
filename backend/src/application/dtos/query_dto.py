from pydantic import BaseModel, Field
from typing import Optional, List


class QuestionRequest(BaseModel):
    """Data Transfer Object for incoming question requests."""
    query: str = Field(..., min_length=1, max_length=1000, description="The question to be answered")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?"
            }
        }


class DocumentSource(BaseModel):
    """Data Transfer Object for document sources in the response."""
    content: str = Field(..., description="The relevant document content")
    source: Optional[str] = Field(None, description="The source of the document (e.g., file name)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "content": "Machine learning is a subset of artificial intelligence...",
                "source": "document_1.pdf"
            }
        }


class AnswerResponse(BaseModel):
    """Data Transfer Object for question-answer responses."""
    question: str = Field(..., description="The original question")
    answer: str = Field(..., description="The generated answer from the RAG pipeline")
    sources: List[DocumentSource] = Field(default_factory=list, description="List of source documents used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is machine learning?",
                "answer": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience...",
                "sources": [
                    {
                        "content": "Machine learning is a subset of artificial intelligence...",
                        "source": "document_1.pdf"
                    }
                ]
            }
        }
