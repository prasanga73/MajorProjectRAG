from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import threading
import uvicorn

# Import your custom classes
from src.data_processor import LegalDocProcessor
from src.hybrid_retriever import HybridRetriever

app = FastAPI(title="Nepal Law Search API")

# --- Configuration ---
INDEX_DIR = "index_storage"
PARENT_DATA = "data/parent_docs.json"
CHILD_DATA = "data/child_docs.json"

# Global state
retriever = None
is_ready = False

# --- Pydantic Models ---
class SearchQuery(BaseModel):
    query: str
    top_k: Optional[int] = 3

class SimpleSearchResult(BaseModel):
    clause_id: str
    clause_text: str
    legal_document_source: str
    chapter: str
    part: str
    score: float

# --- Background Loading Logic ---
def load_retriever_task():
    """Heavy AI loading task runs in a separate thread."""
    global retriever, is_ready
    try:
        print("--- Starting Background Loading (AI Models & Index) ---")
        
        if os.path.exists(INDEX_DIR):
            print("[*] Loading existing index...")
            retriever = HybridRetriever(index_dir=INDEX_DIR)
        else:
            print("[*] Building new index (this may take a few minutes)...")
            processor = LegalDocProcessor(PARENT_DATA, CHILD_DATA)
            docs = processor.load_and_clean()
            if not docs:
                print("[!] Error: No documents found to index.")
                return
            retriever = HybridRetriever(documents=docs, index_dir=INDEX_DIR)
            retriever.save_index()
            
        is_ready = True
        print("--- SUCCESS: Retriever Ready for Search ---")
    except Exception as e:
        print(f"--- FAILURE: Error during loading: {e} ---")

@app.on_event("startup")
async def startup_event():
    # Start the thread and immediately return so the port opens
    thread = threading.Thread(target=load_retriever_task)
    thread.start()

# --- Endpoints ---

@app.get("/")
def root():
    return {
        "message": "Nepal Law Search API is running", 
        "status": "ready" if is_ready else "loading"
    }

@app.get("/health")
def health_check():
    if not is_ready:
        return {"status": "loading", "details": "AI models are still initializing in the background."}
    return {"status": "ready"}

@app.post("/search", response_model=List[SimpleSearchResult])
async def search_law(request: SearchQuery):
    if not is_ready or retriever is None:
        raise HTTPException(
            status_code=503, 
            detail="The search engine is still warming up. Please try again in a few minutes."
        )

    try:
        # Get grouped results
        grouped_results = retriever.hybrid_search(request.query, top_k=request.top_k)
        
        flattened_results = []
        for group in grouped_results:
            flattened_results.append({
                "clause_id": str(group['parent_clause_id']).upper(),
                "clause_text": group['parent_clause_text'],
                "legal_document_source": group['legal_document_source'],
                "chapter": str(group.get('chapter') or "N/A"),
                "part": str(group.get('part') or "N/A"),
                "score": round(float(group.get('score', 0)), 4)
            })
            
        return flattened_results
    
    except Exception as e:
        print(f"Search Error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during the search process.")

# --- Render Port Binding ---
if __name__ == "__main__":
    # Get port from environment variable (Render sets this) or default to 8000
    port = int(os.environ.get("PORT", 8000))
    # Run uvicorn on 0.0.0.0 to be accessible externally
    uvicorn.run(app, host="0.0.0.0", port=port)