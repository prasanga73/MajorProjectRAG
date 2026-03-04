import os
import pickle
import re
from src.hybrid_retriever import HybridRetriever

def diagnostic_test():
    index_path = "index_storage/data.pkl"
    
    if not os.path.exists(index_path):
        print("[!] ERROR: No index found at index_storage/data.pkl. Please run your main app first.")
        return

    print("[*] Loading indexed data for analysis...")
    with open(index_path, "rb") as f:
        data = pickle.load(f)
        docs = data["docs"]
    
    print(f"[*] Total documents in index: {len(docs)}")

    # --- TEST 1: DATA PRESENCE ---
    print("\n--- TEST 1: Data Presence ---")
    test_keywords = ["consumer", "trafficking", "acid", "witchcraft"]
    found_map = {k: False for k in test_keywords}
    
    for d in docs:
        content = d['search_content'].lower()
        src = d['metadata'].get('legal_document_source', "").lower()
        for k in test_keywords:
            if k in content or k in src:
                found_map[k] = True
    
    for k, found in found_map.items():
        status = "✅ FOUND" if found else "❌ NOT FOUND"
        print(f"Keyword '{k}': {status}")

    # --- TEST 2: TRIGGER MAP ACCURACY ---
    print("\n--- TEST 2: Trigger Map Logic ---")
    retriever = HybridRetriever() # This loads the models
    test_queries = ["misleading ad", "trafficking", "acid attack"]
    
    for q in test_queries:
        filters = retriever._get_active_filters(q)
        if filters:
            print(f"Query '{q}': ✅ Triggers {filters[0].get('source')}")
            print(f"   Keywords: {filters[0].get('chapter_keywords')}")
        else:
            print(f"Query '{q}': ❌ NO TRIGGER FOUND")

    # --- TEST 3: MANUAL FILTER SIMULATION ---
    print("\n--- TEST 3: Filter Matching Simulation ---")
    # We find one document that SHOULD match and see if the filter logic likes it
    target_found = False
    for i, d in enumerate(docs):
        if "consumer" in d['search_content'].lower():
            target_found = True
            print(f"Testing Doc Index {i} (Consumer section)...")
            
            # Simulate the trigger for 'misleading ad'
            active_filters = retriever._get_active_filters("misleading ad")
            
            # Run manual check
            results = retriever._apply_filters([i], active_filters)
            
            if results:
                print("✅ Filter Logic: SUCCESS (Document passed the filter)")
            else:
                print("❌ Filter Logic: FAILED (The filter rejected this document)")
                # Show why
                meta = d['metadata']
                print(f"   Doc Source: '{meta.get('legal_document_source')}'")
                print(f"   Doc Chapter: '{meta.get('chapter')}'")
                print(f"   Filter Target Source: '{active_filters[0].get('source')}'")
            break
    
    if not target_found:
        print("❌ Could not perform Test 3: Consumer data not found in index.")

if __name__ == "__main__":
    diagnostic_test()