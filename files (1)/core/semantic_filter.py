"""
Semantic Domain Filtering Module
Prevents retrieval of unrelated clauses by filtering based on semantic domain coherence
Example: "I killed a dog" should retrieve animal protection clauses, not homicide clauses
"""

import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass


@dataclass
class DomainKeywords:
    """Keywords that indicate clause domain"""
    primary: Set[str]      # Strong indicators (e.g., "animal", "homicide")
    context: Set[str]      # Context keywords that support domain
    exclude: Set[str]      # Keywords that should exclude this domain


# Domain definitions for Nepali Legal Code
DOMAIN_KEYWORDS = {
    "animal_protection": DomainKeywords(
        primary={"animal", "dog", "cat", "livestock", "poultry", "cattle", "beast", 
                 "cruelty", "welfare", "neglect", "veterinary", "breed", "endangered"},
        context={"pet", "fauna", "wildlife", "nature", "protect", "conserv", "harm", 
                 "injury", "feed", "care", "abuse", "torture","killing"},
        exclude={"homicide","genocide","state","treason", "murder", "assault", "person", "human", "culpable"}
    ),
    
    "homicide_criminal": DomainKeywords(
        primary={"homicide", "murder", "manslaughter", "culpable", "death", "kill","killed", 
                 "slaughter", "person", "human", "intentional", "negligence"},
        context={"grievous", "hurt", "wound", "fatal", "dangerous", "weapon", 
                 "criminal", "offense", "punishment", "sentence"},
        exclude={"animal", "dog","cat","livestock", "beast", "pet", "slaughter-house", "custom"}
    ),
    
    "assault_battery": DomainKeywords(
        primary={"assault", "battery", "hurt", "injury", "wound", "grievous", 
                 "simple", "voluntarily", "cause", "physical"},
        context={"force", "threat", "violence", "attack", "harm", "damage", 
                 "pain", "suffering", "recover"},
        exclude={"animal", "veterinary", "surgical", "medical", "treatment"}
    ),
    
    "property_law": DomainKeywords(
        primary={"property", "ownership", "title", "possession", "land", "house", 
                 "building", "immovable", "movable", "right", "interest"},
        context={"inherit", "transfer", "own", "acquire", "dispose", "lease", 
                 "mortgage", "pledge", "encumbrance"},
        exclude={"crime", "punishment", "offense", "criminal", "sentence"}
    ),
    
    "contract_law": DomainKeywords(
        primary={"contract", "agreement", "promise", "party", "offer", "accept", 
                 "consideration", "obligation", "condition", "breach"},
        context={"execute", "perform", "fulfill", "payment", "deliver", "terminate", 
                 "void", "valid", "enforce"},
        exclude={"crime", "criminal", "offense", "punishment", "sentence"}
    ),
    
    "family_marriage": DomainKeywords(
        primary={"marriage", "divorce", "husband", "wife", "spouse", "child", "parent", 
                 "family", "relation", "descendant", "widow"},
        context={"consent", "ceremony", "conjugal", "matrimonial", "succession", 
                 "alimony", "maintenance"},
        exclude={"commercial", "business", "trade", "property", "contract"}
    ),
    
    "inheritance_succession": DomainKeywords(
        primary={"inherit", "succession", "heir", "legatee", "estate", "bequest", 
                 "will", "descent", "succession", "testator"},
        context={"property", "distribute", "share", "devolution", "succession", 
                 "intestate", "testate"},
        exclude={"crime", "theft", "fraud", "forgery"}
    )
}


class QueryIntentDetector:
    """Detects the intended domain/type of query"""
    
    def __init__(self):
        """Initialize with domain keywords"""
        self.domain_keywords = DOMAIN_KEYWORDS
    
    def detect_domain(self, query: str) -> Dict[str, float]:
        """
        Detect which legal domain(s) the query relates to.
        Returns dict of {domain: confidence_score}
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            # Score based on primary keywords (high weight)
            primary_matches = len(query_words & keywords.primary)
            primary_score = primary_matches * 0.8
            
            # Score based on context keywords (medium weight)
            context_matches = len(query_words & keywords.context)
            context_score = context_matches * 0.3
            
            # Penalty for exclude keywords (negative weight)
            exclude_matches = len(query_words & keywords.exclude)
            exclude_penalty = exclude_matches * 0.5
            
            total_score = max(0, primary_score + context_score - exclude_penalty)
            
            if total_score > 0:
                domain_scores[domain] = total_score
        
        # Normalize scores to [0, 1]
        if domain_scores:
            max_score = max(domain_scores.values())
            domain_scores = {d: s / max_score for d, s in domain_scores.items()}
        
        return domain_scores
    
    def get_primary_domain(self, query: str) -> Optional[str]:
        """Get the most likely domain for a query"""
        domain_scores = self.detect_domain(query)
        
        if not domain_scores:
            return None
        
        return max(domain_scores, key=domain_scores.get)


class SemanticDomainFilter:
    """Filters and penalizes semantically unrelated results"""
    
    def __init__(self):
        """Initialize intent detector"""
        self.intent_detector = QueryIntentDetector()
        self.domain_keywords = DOMAIN_KEYWORDS
    
    def extract_clause_domain(self, clause_text: str, part: str, chapter: str) -> Dict[str, float]:
        """
        Extract which domain(s) a clause belongs to.
        Analyzes clause text, part, and chapter information.
        """
        combined_text = f"{clause_text} {part} {chapter}".lower()
        words = set(combined_text.split())
        
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            # Primary keywords have higher weight
            primary_matches = len(words & keywords.primary)
            context_matches = len(words & keywords.context)
            
            score = (primary_matches * 0.7) + (context_matches * 0.3)
            
            if score > 0:
                domain_scores[domain] = score
        
        # Normalize if any matches found
        if domain_scores:
            max_score = max(domain_scores.values())
            domain_scores = {d: s / max_score for d, s in domain_scores.items()}
        
        return domain_scores
    
    def calculate_domain_penalty(
        self,
        query: str,
        clause_text: str,
        part: str,
        chapter: str
    ) -> float:
        """
        Calculate penalty (0-1) for semantic domain mismatch.
        
        Returns:
            - 0.0: Perfect match (no penalty)
            - 0.5: Partial mismatch (medium penalty)
            - 1.0: Complete mismatch (maximum penalty)
        """
        
        query_lower = query.lower()
        clause_lower = f"{clause_text} {part} {chapter}".lower()
        
        # Get expected domains from query
        query_domains = self.intent_detector.detect_domain(query)
        
        # Get actual domains from clause
        clause_domains = self.extract_clause_domain(clause_text, part, chapter)
            
        if "animal" in query_lower or "dog" in query_lower:
            if "person" in clause_lower or "human life" in clause_lower or "homicide" in clause_lower:
                return 1.0  # Maximum penalty
                
        # If the query is about humans, but the clause is about animals
        if "person" in query_lower or "human" in query_lower:
            if "animal" in clause_lower or "bird" in clause_lower:
                return 0.8
        
        if not query_domains:
            return 0.0  # No penalty if query intent unclear
        
        if not clause_domains:
            return 0.5  # Medium penalty if clause domain unclear
        
        # Find overlap between query domains and clause domains
        overlap = 0.0
        for domain in query_domains:
            if domain in clause_domains:
                # Overlap score weighted by confidence
                overlap += query_domains[domain] * clause_domains[domain]
        
        # Penalty = 1 - overlap
        penalty = max(0.0, 1.0 - overlap)
        
        return penalty
    
    def apply_filter_penalty(
        self,
        results: List,
        query: str,
        penalty_weight: float = 0.3
    ) -> List:
        """
        Apply domain-based penalty to final scores.
        
        Args:
            results: List of SearchResult objects
            query: Original query string
            penalty_weight: How much to weight domain penalty (0-1)
                           0.0 = no filtering, 1.0 = strict filtering
        
        Returns:
            Modified results with adjusted final_scores
        """
        for result in results:
            penalty = self.calculate_domain_penalty(
                query,
                result.text,
                result.part,
                result.chapter
            )
            
            # Apply penalty to final score
            penalty_factor = 1.0 - (penalty * penalty_weight)
            result.final_score *= penalty_factor
            
            # Store penalty in score breakdown if available
            if hasattr(result, 'score_breakdown') and result.score_breakdown:
                result.score_breakdown['domain_penalty'] = float(penalty)
        
        return results
    
    def filter_by_domain_coherence(
        self,
        results: List,
        query: str,
        threshold: float = 0.3
    ) -> List:
        """
        Filter out completely unrelated clauses.
        
        Args:
            results: List of SearchResult objects
            query: Original query string
            threshold: Minimum coherence score to keep result (0-1)
                      0.0 = keep all, 1.0 = only perfect matches
        
        Returns:
            Filtered results keeping only coherent ones
        """
        query_domains = self.intent_detector.detect_domain(query)
        
        if not query_domains:
            return results  # No filtering if intent unclear
        
        filtered = []
        
        for result in results:
            clause_domains = self.extract_clause_domain(
                result.text,
                result.part,
                result.chapter
            )
            
            # Check domain overlap
            overlap = 0.0
            for domain in query_domains:
                if domain in clause_domains:
                    overlap += query_domains[domain] * clause_domains[domain]
            
            # Keep if overlap meets threshold
            if overlap >= threshold:
                filtered.append(result)
        
        # Return at least some results if filtering is too strict
        if not filtered and results:
            return results[:1]
        
        return filtered


def analyze_query_domain(query: str) -> Dict:
    """
    Utility function to analyze query and return domain information.
    Useful for debugging and understanding query interpretation.
    
    Returns:
        Dict with 'detected_domains' and 'primary_domain'
    """
    detector = QueryIntentDetector()
    
    return {
        'detected_domains': detector.detect_domain(query),
        'primary_domain': detector.get_primary_domain(query)
    }
