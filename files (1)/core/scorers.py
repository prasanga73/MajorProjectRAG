"""
Scoring and Reranking Module
Implements cross-encoder based reranking and score combination strategies
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class RerankResult:
    """Result after reranking"""
    index: int
    score: float
    rerank_score: float
    final_score: float


class CrossEncoderReranker:
    """Cross-encoder based reranker using sentence-transformers"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        """Initialize reranker with specified model
        
        Default model: cross-encoder/ms-marco-MiniLM-L-12-v2
        - Better than L-6 version (12 layers vs 6)
        - Stronger semantic understanding
        - Optimized for ranking documents
        - Good for legal document reranking
        """
        self.model_name = model_name
        self.model = None
        
        try:
            from sentence_transformers import CrossEncoder
            print(f"Loading reranker model: {model_name}")
            self.model = CrossEncoder(model_name)
            self.enabled = True
            print(f"[OK] Reranker loaded successfully")
        except ImportError:
            print("[WARNING] sentence-transformers not installed. Reranker disabled.")
            self.enabled = False
        except Exception as e:
            print(f"[WARNING] Failed to load reranker: {e}. Reranker disabled.")
            self.enabled = False
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        return_scores: bool = True
    ) -> List[RerankResult]:
        """Rerank documents based on query using cross-encoder"""
        if not self.enabled or not documents:
            return [
                RerankResult(index=i, score=0.0, rerank_score=0.0, final_score=0.0)
                for i in range(len(documents))
            ]
        
        pairs = [[query, doc] for doc in documents]
        rerank_scores = self.model.predict(pairs)
        
        results = [
            RerankResult(
                index=i,
                score=0.0,
                rerank_score=float(rerank_scores[i]),
                final_score=0.0
            )
            for i in range(len(documents))
        ]
        
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        if top_k is not None:
            results = results[:top_k]
        
        return results


class HybridReranker:
    """Hybrid reranker combining initial scores with cross-encoder"""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_weight: float = 0.5,
        initial_weight: float = 0.5
    ):
        """Initialize hybrid reranker"""
        self.cross_encoder = CrossEncoderReranker(model_name)
        self.rerank_weight = rerank_weight
        self.initial_weight = initial_weight
        
        if not np.isclose(rerank_weight + initial_weight, 1.0):
            print(f"Warning: Reranker weights sum to {rerank_weight + initial_weight}, not 1.0")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        initial_scores: List[float],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """Rerank with hybrid scoring combining initial and reranker scores"""
        if not self.cross_encoder.enabled:
            results = [
                RerankResult(
                    index=i, score=initial_scores[i],
                    rerank_score=0.0, final_score=initial_scores[i]
                )
                for i in range(len(documents))
            ]
            results.sort(key=lambda x: x.final_score, reverse=True)
            if top_k is not None:
                results = results[:top_k]
            return results
        
        pairs = [[query, doc] for doc in documents]
        rerank_scores = self.cross_encoder.model.predict(pairs)
        
        # Normalize scores
        rerank_scores_array = np.array(rerank_scores)
        rerank_norm = self._normalize(rerank_scores_array)
        initial_norm = self._normalize(np.array(initial_scores))
        
        # Combine scores
        final_scores = (
            self.initial_weight * initial_norm +
            self.rerank_weight * rerank_norm
        )
        
        results = [
            RerankResult(
                index=i,
                score=float(initial_scores[i]),
                rerank_score=float(rerank_scores[i]),
                final_score=float(final_scores[i])
            )
            for i in range(len(documents))
        ]
        
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        if top_k is not None:
            results = results[:top_k]
        
        return results
    
    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range"""
        if scores.max() == scores.min():
            return np.zeros_like(scores)
        return (scores - scores.min()) / (scores.max() - scores.min())


class LightweightReranker:
    """Lightweight reranker without external dependencies using simple heuristics"""
    
    def __init__(self):
        """Initialize lightweight reranker"""
        self.enabled = True
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        initial_scores: List[float],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """Rerank using lightweight heuristics without external models"""
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        rerank_scores = []
        
        for doc in documents:
            doc_lower = doc.lower()
            score = 0.0
            
            # Exact phrase match bonus
            if query_lower in doc_lower:
                score += 1.0
            
            # Query term coverage
            doc_terms = set(doc_lower.split())
            term_overlap = len(query_terms & doc_terms) / len(query_terms) if query_terms else 0
            score += term_overlap * 0.5
            
            # Prefer shorter documents
            if len(documents) > 0:
                avg_len = sum(len(d) for d in documents) / len(documents)
                length_factor = avg_len / max(len(doc), 1)
                score += min(length_factor * 0.3, 0.3)
            
            rerank_scores.append(score)
        
        # Normalize
        rerank_norm = self._normalize(np.array(rerank_scores))
        initial_norm = self._normalize(np.array(initial_scores))
        final_scores = 0.7 * initial_norm + 0.3 * rerank_norm
        
        results = [
            RerankResult(
                index=i,
                score=float(initial_scores[i]),
                rerank_score=float(rerank_norm[i]),
                final_score=float(final_scores[i])
            )
            for i in range(len(documents))
        ]
        
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        if top_k is not None:
            results = results[:top_k]
        
        return results
    
    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range"""
        if scores.max() == scores.min():
            return np.zeros_like(scores)
        return (scores - scores.min()) / (scores.max() - scores.min())


def create_reranker(
    reranker_type: str = "hybrid",
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
    rerank_weight: float = 0.5,
    initial_weight: float = 0.5
):
    """Factory function to create reranker of specified type"""
    if reranker_type == "hybrid":
        return HybridReranker(model_name, rerank_weight, initial_weight)
    elif reranker_type == "cross-encoder":
        return CrossEncoderReranker(model_name)
    elif reranker_type == "lightweight":
        return LightweightReranker()
    else:
        raise ValueError(f"Unknown reranker type: {reranker_type}")
