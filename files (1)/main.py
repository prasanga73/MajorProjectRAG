"""Legal RAG System - Main Entry Point"""
from core import LegalRAGSystem, format_results
import sys

def main():
    print("=" * 80)
    print("LEGAL RAG SYSTEM - Document Retrieval")
    print("=" * 80)
    
    try:
        rag = LegalRAGSystem(
            child_docs_path="data/child_docs.json",
            parent_docs_path="data/parent_docs.json",
            use_reranker=True,
            reranker_type="hybrid",
            verbose=True
        )
    except FileNotFoundError as e:
        print(f"[ERROR] Error: {e}")
        print("Make sure child_docs.json and parent_docs.json are in the data/ directory")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("Ready to search. Type quit or exit to close.")
    print("=" * 80)
    
    while True:
        try:
            query = input("\n[INPUT] Enter your query: ").strip()
            if query.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break
            if not query:
                print("Please enter a query.")
                continue
            
            results = rag.search(query=query, top_k=5, deduplicate_parents=True, return_parent_text=True)
            print(format_results(results, verbose=True))
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"[ERROR] Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

