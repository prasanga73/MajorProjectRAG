from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union
import os

# Import your existing classes
from src.data_processor import LegalDocProcessor
from src.hybrid_retriever import HybridRetriever

app = FastAPI(title="Nepal Law Search API")

# --- Configuration ---
INDEX_DIR = "index_storage"
PARENT_DATA = "data/parent_docs.json"
CHILD_DATA = "data/child_docs.json"

retriever = None

# --- Simplified Pydantic Models ---
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

# --- Lifecycle Events ---
@app.on_event("startup")
async def startup_event():
    global retriever
    if os.path.exists(INDEX_DIR):
        print("--- Loading existing index ---")
        retriever = HybridRetriever(index_dir=INDEX_DIR)
    else:
        print("--- Building new index ---")
        processor = LegalDocProcessor(PARENT_DATA, CHILD_DATA)
        docs = processor.load_and_clean()
        if not docs:
            print("No documents found.")
            return
        retriever = HybridRetriever(documents=docs, index_dir=INDEX_DIR)
        retriever.save_index()
    print("--- Retriever Ready ---")

# --- Flattened Endpoint ---

@app.post("/search", response_model=List[SimpleSearchResult])
async def search_law(request: SearchQuery):
    if retriever is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized.")

    try:
        # 1. Get the grouped results from the retriever
        grouped_results = retriever.hybrid_search(request.query, top_k=request.top_k)
        
        flattened_results = []
        
        for group in grouped_results:
            # We determine which text to show: 
            # If the search hit a specific sub-clause, we use that.
            # Otherwise, we use the parent text.
            
            display_id = group['parent_clause_id']
            display_text = group['parent_clause_text']
            
            # if group['sub_clauses']:
            #     # Take the first matching sub-clause as the primary hit
            #     display_id = group['sub_clauses'][0]['id']
            #     display_text = group['sub_clauses'][0]['text']

            # Create the clean dictionary
            flattened_results.append({
                "clause_id": str(display_id).upper(),
                "clause_text": display_text,
                "legal_document_source": group['legal_document_source'],
                "chapter": str(group.get('chapter') or "N/A"),
                "part": str(group.get('part') or "N/A"),
                "score": round(float(group.get('score', 0)), 4)
            })
            
        return flattened_results
    
    except Exception as e:
        print(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)