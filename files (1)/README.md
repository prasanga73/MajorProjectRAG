# Legal RAG System - Refactored

A clean, modular legal document retrieval system with intelligent parent clause deduplication.

## 🎯 Key Improvements

### Code Organization

- **Minimal files**: Consolidated from 10+ files to 4 core modules
- **Modular structure**: Separated concerns (extraction, scoring, retrieval, utilities)
- **Easy maintenance**: Clear imports and dependencies

### Parent Clause Deduplication

When multiple child clauses of the same parent are found:

- **Before**: Returns all children with duplicated parent information
- **After**: Returns only the best-scoring child per parent
- **Result**: Cleaner, non-redundant results

### Example

Query: "clause 22"

```
Without Deduplication:
  ✓ 22(a)  - Score: 0.92
  ✓ 22(b)  - Score: 0.85  (Parent 22 duplicated)
  ✓ 22(c)  - Score: 0.78  (Parent 22 duplicated)

With Deduplication:
  ✓ 22(a)  - Score: 0.92  (Best child of parent 22)
```

## 📁 Project Structure

```
legal-rag/
├── core/                          # Core modules
│   ├── __init__.py               # Module exports
│   ├── extractors.py             # Query and clause ID extraction
│   ├── scorers.py                # Reranking and scoring logic
│   ├── retriever.py              # Main RAG system (with deduplication)
│   └── utils.py                  # BM25, embeddings, formatting
│
├── data/                         # Data directory
│   ├── child_docs.json          # Child clause documents
│   └── parent_docs.json         # Parent clause documents
│
├── tests/                        # Test files
│   ├── test_search.py
│   ├── test_extraction.py
│   └── ...
│
├── main.py                      # Interactive CLI
├── examples.py                  # Usage examples
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Interactive Search

```bash
python main.py
```

Example queries:

- "Fetch clause 22 of criminal code"
- "Summarize clause 2(h)"
- "Show me clause 1(2)"
- "Explain section 239"

### Usage Examples

```bash
python examples.py
```

Shows:

1. Basic search without deduplication
2. Search with deduplication (recommended)
3. Search without reranking (faster)
4. Deduplication demo
5. Detailed score breakdown

### Programmatic Usage

```python
from core import LegalRAGSystem, format_results

# Initialize system
rag = LegalRAGSystem(
    child_docs_path="data/child_docs.json",
    parent_docs_path="data/parent_docs.json"
)

# Search with deduplication (recommended)
results = rag.search(
    query="Fetch clause 22",
    top_k=5,
    deduplicate_parents=True  # Eliminates duplicate parent clauses
)

# Format and display results
print(format_results(results, verbose=True))
```

## 🔧 Configuration

### Reranking Options

```python
# With hybrid reranking (default)
rag = LegalRAGSystem(
    ...,
    use_reranker=True,
    reranker_type="hybrid"  # 'hybrid', 'cross-encoder', or 'lightweight'
)

# Three reranker types:
# - 'hybrid': Combines semantic and reranking scores (best quality)
# - 'cross-encoder': Pure cross-encoder reranking (requires transformers)
# - 'lightweight': No external dependencies, uses heuristics (fastest)
```

### Deduplication Options

```python
# Enable deduplication (recommended)
results = rag.search(
    query="...",
    top_k=5,
    deduplicate_parents=True  # Only best child per parent is returned
)
```

### Scoring Weights

```python
rag = LegalRAGSystem(
    ...,
    similarity_weight=0.5,      # Semantic similarity weight
    bm25_weight=0.5,            # BM25 lexical weight
    direct_hit_bonus=2.0,       # Bonus for direct clause matches
    rerank_weight=0.5,          # Reranker score weight
    initial_weight=0.5          # Initial retrieval score weight
)
```

## 📊 How It Works

### Search Pipeline

1. **Query Parsing**: Extract clause IDs, document type, action type
2. **Initial Retrieval**: Semantic similarity + BM25 scoring
3. **Direct Hit Bonus**: Boost scores for explicitly requested clauses
4. **Reranking**: Cross-encoder model refines top candidates
5. **Deduplication**: Keep only best child per parent clause
6. **Parent Retrieval**: Fetch parent clause texts
7. **Scoring Breakdown**: Return detailed score information

### Deduplication Algorithm

```
For each result in ranked list:
  1. Extract base clause number (parent)
  2. Create unique key: (parent_id, document_source)
  3. If first child of this parent: include in results
  4. If not first child: compare scores, keep the better one
  5. Return deduplicated results limited to top_k
```

## 📦 Core Modules

### `extractors.py`

- `QueryExtractor`: Parses queries to extract clause IDs, document types, action types
- Supports multiple patterns: "clause 22", "22(a)(1)", "section 1(2)", etc.

### `scorers.py`

- `CrossEncoderReranker`: Neural reranking
- `HybridReranker`: Combines initial and rerank scores
- `LightweightReranker`: Heuristic-based (no dependencies)
- `create_reranker()`: Factory function

### `retriever.py`

- `LegalRAGSystem`: Main RAG system with deduplication
- `search()`: Execute queries with all pipeline steps
- `_deduplicate_by_parent()`: Core deduplication logic

### `utils.py`

- `BM25`: BM25 ranking algorithm
- `ProductionEmbeddingModel`: Embedding model wrapper
- `SearchResult`, `RetrievalResult`: Data classes
- `format_results()`: Pretty printing

## 🎨 Output Example

```
══════════════════════════════════════════════════════════════════════════════
SEARCH RESULTS (3 found)
══════════════════════════════════════════════════════════════════════════════

────────────────────────────────────────────────────────────────────────────────
RESULT #1
────────────────────────────────────────────────────────────────────────────────
Clause ID:      22(a)
Source:         National Criminal Code, 2017 AD
Part:           Part-2 (Specific Offences)
Chapter:        Chapter-3 (Against Persons)
Direct Hit:     ✓ YES
Final Score:    0.9245
  └─ Similarity:  0.8932
  └─ BM25:        0.9158
  └─ Initial:     0.9045
  └─ Rerank:      0.9245
  └─ Bonus:       2.0000

📄 CHILD CLAUSE:
   [Detailed child clause text...]

📋 PARENT CLAUSE:
   [Parent clause 22 text...]
```

## Performance

- Embedding Model: all-MiniLM-L6-v2 (~22M parameters)
- Reranker Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~22M parameters)
- Index Size: ~38,000 child documents
- Typical Search Time: 100-500ms (with reranking)

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_search.py
```

## Troubleshooting

### "sentence-transformers not installed"

- Install models: `pip install -r requirements.txt`
- Or use lightweight reranker: `reranker_type="lightweight"`

### Slow search

- Disable reranking: `use_reranker=False`
- Use lightweight reranker: `reranker_type="lightweight"`
- Reduce data: use `filter_document_source` parameter

### No results found

- Check JSON files are in `data/` directory
- Verify clause IDs in query match document structure
- Try broader queries or disable deduplication

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please:

1. Create a feature branch
2. Add tests for new features
3. Update documentation
4. Submit a pull request
