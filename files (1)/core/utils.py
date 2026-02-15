"""
Utility Module
Contains embedding models, BM25 scoring, and formatting utilities
"""

import json
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict
from dataclasses import dataclass


class BM25:
    """BM25 ranking algorithm implementation"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """Initialize BM25 with parameters"""
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.idf = {}
        self.doc_lengths = []
        self.avg_doc_length = 0
    
    def fit(self, corpus: List[str]):
        """Fit BM25 on corpus"""
        self.corpus = corpus
        self.doc_lengths = [len(doc.split()) for doc in corpus]
        self.avg_doc_length = sum(self.doc_lengths) / len(corpus) if corpus else 0
        
        # Calculate IDF
        num_docs = len(corpus)
        doc_freq = defaultdict(int)
        
        for doc in corpus:
            seen = set()
            for word in doc.split():
                if word not in seen:
                    doc_freq[word] += 1
                    seen.add(word)
        
        # IDF formula: log((N - df + 0.5) / (df + 0.5))
        for word, df in doc_freq.items():
            self.idf[word] = np.log((num_docs - df + 0.5) / (df + 0.5))
    
    def get_scores(self, query: str) -> np.ndarray:
        """Get BM25 scores for query against corpus"""
        scores = np.zeros(len(self.corpus))
        query_terms = query.split()
        
        for i, doc in enumerate(self.corpus):
            doc_words = doc.split()
            score = 0.0
            
            for term in query_terms:
                if term not in self.idf:
                    continue
                
                term_freq = doc_words.count(term)
                idf = self.idf[term]
                
                # BM25 formula
                numerator = idf * term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (
                    1 - self.b + self.b * (self.doc_lengths[i] / self.avg_doc_length)
                )
                
                score += numerator / denominator
            
            scores[i] = score
        
        return scores


class ProductionEmbeddingModel:
    """Wrapper for sentence-transformers embedding model"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """Initialize embedding model
        
        Default model: all-mpnet-base-v2 (110M params)
        - Better quality than MiniLM (larger)
        - Good balance of quality and speed
        - Optimized for semantic similarity
        - Works well with English legal documents
        """
        self.model_name = model_name
        self.model = None
        
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.enabled = True
            print(f"[OK] Embedding model loaded")
        except ImportError:
            print("[WARNING] sentence-transformers not installed. Using placeholder embeddings.")
            self.enabled = False
        except Exception as e:
            print(f"[WARNING] Failed to load embeddings: {e}. Using placeholders.")
            self.enabled = False
    
    def encode(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """Encode texts to embeddings"""
        if not self.enabled:
            # Return random embeddings
            return np.random.randn(len(texts), 384).astype(np.float32)
        
        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings


@dataclass
class SearchResult:
    """Represents a search result with scores and metadata"""
    clause_id: str
    text: str
    parent_clause_id: str
    legal_document_source: Optional[str]=None
    part: Optional[str] = None
    chapter: Optional[str] = None
    parent_text: str = ""
    similarity_score: float = 0.0
    bm25_score: float = 0.0
    initial_score: float = 0.0
    rerank_score: float = 0.0
    direct_hit_bonus: float = 0.0
    final_score: float = 0.0
    is_direct_hit: bool = False


@dataclass
class RetrievalResult:
    """Final retrieval result with parent clause text"""
    clause_id: str
    child_text: str
    parent_text: str
    parent_clause_id: str = ""
    legal_document_source: str = ""
    part: str = ""
    chapter: str = ""
    final_score: float = 0.0
    is_direct_hit: bool = False
    score_breakdown: Dict = None


def load_json(path: str) -> List[Dict]:
    """Load JSON file"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_results(results: List[RetrievalResult], verbose: bool = True) -> str:
    """Format search results for display"""
    output = []
    output.append("=" * 90)
    output.append(f"SEARCH RESULTS ({len(results)} found)")
    output.append("=" * 90)
    
    for i, result in enumerate(results, 1):
        output.append(f"\n{'─' * 90}")
        output.append(f"RESULT #{i}")
        output.append(f"{'─' * 90}")
        
        output.append(f"Clause ID:      {result.clause_id}")
        output.append(f"Parent Clause:  {result.parent_clause_id}")
        output.append(f"Source:         {result.legal_document_source}")
        output.append(f"Part:           {result.part}")
        output.append(f"Chapter:        {result.chapter}")
        output.append(f"Direct Hit:     {'[YES]' if result.is_direct_hit else '[NO]'}")
        output.append(f"Final Score:    {result.final_score:.4f}")
        
        if verbose and result.score_breakdown:
            sb = result.score_breakdown
            output.append(f"  └─ Similarity:  {sb.get('similarity', 0):.4f}")
            output.append(f"  └─ BM25:        {sb.get('bm25', 0):.4f}")
            output.append(f"  └─ Initial:     {sb.get('initial', 0):.4f}")
            if sb.get('rerank', 0) > 0:
                output.append(f"  └─ Rerank:      {sb['rerank']:.4f}")
            if sb.get('direct_hit_bonus', 0) > 0:
                output.append(f"  └─ Bonus:       {sb['direct_hit_bonus']:.4f}")
        
        output.append(f"\n📄 CHILD CLAUSE:")
        output.append(f"   {result.child_text}")
        
        output.append(f"\n📋 PARENT CLAUSE:")
        parent_preview = result.parent_text[:200] + "..." if len(result.parent_text) > 200 else result.parent_text
        output.append(f"   {parent_preview}")
    
    output.append(f"\n{'=' * 90}")
    
    return "\n".join(output)
