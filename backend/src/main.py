
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from presentation.api.routes.routes import create_api_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application mounted at /api."""

    

    # API sub-application — all routes and docs live here
    api = FastAPI(
        title="RAG Search API",
        description="API for Retrieval-Augmented Generation",
        version="1.0.0",
        root_path="/api",
    )

    # Configure CORS middleware
    cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

    api.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"], 
    )

    # Register all API routes
    try:
        api_router = create_api_router()
        api.include_router(api_router)
        print("All routes registered successfully.")
    except Exception as e:
        print(f"Warning: Could not register routes: {str(e)}")
        print("API will be available but some endpoints may not work.")

    # Mount at /api
    

    return  api


# Create the app instance
app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="localhost",
        port=8000,
        reload=os.getenv("ENVIRONMENT", "production") == "development"
    )
