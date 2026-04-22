
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from application.use_cases.query import QueryUseCase
from presentation.api.routes.query import create_query_router

query_use_case: QueryUseCase | None = None


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Search API",
        description="API Retrieval-Augmented Generation",
        version="1.0.0",
    )
    
    # Configure CORS middleware
    cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    

    if query_use_case is not None:
        query_router = create_query_router(query_use_case)
        app.include_router(query_router)
    
    return app


# Create the app instancewww
app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=os.getenv("ENVIRONMENT", "production") == "development"
    )
