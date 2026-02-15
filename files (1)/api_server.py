"""
Legal RAG System - FastAPI Server
Clean, minimal, production-ready REST API
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List,Optional
import uvicorn
from core import LegalRAGSystem

# ============================================================================
# Initialize RAG System
# ============================================================================

rag_system = None

try:
    rag_system = LegalRAGSystem(
        child_docs_path="data/child_docs.json",
        parent_docs_path="data/parent_docs.json",
        use_reranker=True,
        reranker_type="hybrid",
        verbose=False
    )
    print("[OK] RAG system initialized")
except Exception as e:
    print(f"[ERROR] Failed to init RAG: {e}")


# ============================================================================
# Data Models
# ============================================================================

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    return_parent_text: bool = True
    deduplicate_parents: bool = True


class SearchResult(BaseModel):
    clause_id: str
    child_text: str
    parent_clause_id: str
    parent_text: str
    final_score: float
    legal_document_source:str
    chapter:Optional[str]=None
    part:Optional[str]=None


class SearchResponse(BaseModel):
    query: str
    results_count: int
    results: List[SearchResult]


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Legal RAG System API",
    description="REST API for legal document search",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "name": "Legal RAG System API",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/health", "method": "GET"},
            {"path": "/search", "method": "GET or POST"},
            {"path": "/docs", "method": "GET"}
        ]
    }


@app.get("/health")
def health():
    """Health check"""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    return {"status": "healthy", "message": "System is running"}


@app.post("/search", response_model=SearchResponse)
def search_post(request: SearchRequest):
    """Search using POST"""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if request.top_k < 1 or request.top_k > 100:
        raise HTTPException(status_code=400, detail="top_k must be 1-100")
    
    try:
        results = rag_system.search(
            query=request.query,
            top_k=request.top_k,
            return_parent_text=request.return_parent_text,
            deduplicate_parents=request.deduplicate_parents
        )
        
        items = [
            SearchResult(
                clause_id=r.clause_id,
                child_text=r.child_text,
                parent_clause_id=r.parent_clause_id,
                parent_text=r.parent_text,
                final_score=r.final_score,
                legal_document_source=r.legal_document_source,
                chapter=r.chapter,
                part=r.part
            )
            for r in results
        ]
        
        return SearchResponse(
            query=request.query,
            results_count=len(items),
            results=items
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search", response_model=SearchResponse)
def search_get(
    query: str = Query(...),
    top_k: int = Query(5, ge=1, le=100),
    return_parent_text: bool = Query(True),
    deduplicate_parents: bool = Query(True)
):
    """Search using GET"""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        results = rag_system.search(
            query=query,
            top_k=top_k,
            return_parent_text=return_parent_text,
            deduplicate_parents=deduplicate_parents
        )
        
        items = [
            SearchResult(
                clause_id=r.clause_id,
                child_text=r.child_text,
                parent_clause_id=r.parent_clause_id,
                parent_text=r.parent_text,
                final_score=r.final_score
            )
            for r in results
        ]
        
        return SearchResponse(
            query=query,
            results_count=len(items),
            results=items
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Legal RAG API")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("LEGAL RAG SYSTEM - API SERVER")
    print("=" * 80)
    print(f"\nStarting server at http://{args.host}:{args.port}")
    print(f"API Docs: http://{args.host}:{args.port}/docs")
    print("Press Ctrl+C to stop\n")
    
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
