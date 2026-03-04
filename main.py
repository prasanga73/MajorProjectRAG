# main.py
import os
from src.data_processor import LegalDocProcessor
from src.hybrid_retriever import HybridRetriever

INDEX_DIR = "index_storage"

def main():
    if os.path.exists(INDEX_DIR):
        print("--- Loading existing index ---")
        retriever = HybridRetriever(index_dir=INDEX_DIR)
    else:
        print("--- Building new index ---")
        processor = LegalDocProcessor('data/parent_docs.json', 'data/child_docs.json')
        docs = processor.load_and_clean()
        if not docs:
            print("No documents found.")
            return
        retriever = HybridRetriever(documents=docs, index_dir=INDEX_DIR)
        retriever.save_index()

    while True:
        query = input("\nEnter Question (or 'exit'): ")
        if query.lower() == 'exit': break
        
        results = retriever.hybrid_search(query, top_k=3)
        
        if not results:
            print("No matches found.")
            continue

        for i, parent in enumerate(results):
            print(f"\n{'='*75}")
            print(f"RESULT {i+1}: {parent['legal_document_source']}")
            
            if parent.get('part') and parent['part'] != "N/A":
                print(f"PART: {parent['part']}")
            if parent.get('chapter') and parent['chapter'] != "N/A":
                print(f"CHAPTER: {parent['chapter']}")
                
            print(f"PARENT {parent['parent_clause_id']}: {parent['parent_clause_text']}")
            print(f"{'-'*75}")
            for sub in parent['sub_clauses']:
                if sub['text'].strip().lower() != parent['parent_clause_text'].strip().lower():
                    print(f"  > [{sub['id'].upper()}] {sub['text']}")

if __name__ == "__main__":
    main()