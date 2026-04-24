"""Centralized route registration for the API."""

from sys import prefix
from fastapi import APIRouter

from application.use_cases.query import QueryUseCase
from presentation.api.routes.query import create_query_router


def create_api_router() -> APIRouter:
    """
    Create the main API router with all sub-routers included.
    
    Initializes use cases and registers all route modules 
    under the /api prefix. This is the single entry point 
    for route configuration — main.py only needs to call this.
    
    Returns:
        APIRouter with all application routes registered.
    """
    api_router = APIRouter()

    # --- Query routes ---
    query_use_case = QueryUseCase()
    query_router = create_query_router(query_use_case)
    api_router.include_router(query_router)

    # --- Document routes ---
    #  add document/ingest routes here when implemented
    # ingest_use_case = IngestUseCase()
    # document_router = create_document_router(ingest_use_case)
    # api_router.include_router(document_router)

    return api_router
