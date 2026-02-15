"""
Query Normalizer for Legal RAG System
Transforms user queries into optimized forms for perfect retrieval
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class NormalizedQuery:
    """Structured normalized query"""
    original: str
    normalized: str
    intent: str
    domain: str
    key_terms: List[str]
    reformulated: str
    suggested_chapters: List[str]
    confidence: float


class QueryNormalizer:
    """Normalizes queries for optimal legal document retrieval"""
    
    # Intent classification patterns
    INTENT_PATTERNS = {
        'definition': r'\b(what\s+is|define|meaning|refers?|means?)\b',
        'procedure': r'\b(how\s+to|how\s+can|what\s+are\s+the\s+steps|process)\b',
        'requirement': r'\b(requirement|requirement|condition|must|should|necessary)\b',
        'eligibility': r'\b(eligible|eligibility|qualify|qualifi|who\s+can)\b',
        'right': r'\b(right|rights|entitle|entitled|claim)\b',
        'obligation': r'\b(obligation|must|shall|required|duty|bound)\b',
        'prohibition': r'\b(cannot|prohibited|not\s+allowed|unlawful|illegal)\b',
        'exception': r'\b(except|exception|unless|provided\s+that)\b',
        'punishment': r'\b(punishment|penalt|fine|imprisonment|sentence)\b',
    }
    
    # Domain/subject classification
    DOMAIN_PATTERNS = {
        'animal_liability': r'\b(dog|cat|animal|pet|livestock|cattle|beast|cow|goat)\b',
        'citizenship': r'\b(citizenship|citizen|citizens|national|nationality|naturali)\b',
        'family_law': r'\b(marriage|marital|matrimonial|divorce|domestic|family)\b',
        'property': r'\b(property|possession|owner|ownership|land|real\s+estate|movable|immovable)\b',
        'contract': r'\b(contract|agreement|obligation|term|covenant|bargain)\b',
        'succession': r'\b(succession|inheritance|heir|legacy|will|ancestor|testator|intestate)\b',
        'criminal': r'\b(crime|criminal|offense|assault|theft|murder|penalty|imprisonment)\b',
        'liability': r'\b(liability|liable|responsible|compensation|damages|claim)\b',
        'constitutional': r'\b(constitution|fundamental|right|freedom|equality|sovereignty)\b',
        'criminal_homicide': r'\b(murder|homicide|manslaughter|killing\s+a\s+person)\b'
    }
    
    # Chapter mappings for Nepali legal documents
    CHAPTER_MAPPINGS = {
        'citizenship': {
            'Constitution': ['Clause 11', 'Clause 35'],
            'National Civil Code': ['Chapter-18 (Citizenship and Naturalization)'],
        },
        'marriage': {
            'National Civil Code': [
                'Chapter-1 (Provisions Relating to Marriage)',
                'Chapter-2 (Consequences of Marriage)',
                'Chapter-11 (Offences Relating to Marriage)',
            ],
        },
        'property': {
            'National Civil Code': [
                'Chapter-1 (General Provisions)',
                'Chapter-11 (Transfer and Acquisition of Ownership)',
                'Chapter-12 (Mortgage)',
            ],
        },
        'succession': {
            'National Civil Code': [
                'Chapter-15 (Succession)',
                'Chapter-16 (Intestate Succession)',
                'Chapter-17 (Testamentary Succession)',
            ],
        },
        'criminal': {
            'National Criminal Code': [
                'Chapter-2 (Offences)',
                'Chapter-3 (Punishment)',
            ],
        },
        'contract': {
            'National Civil Code': [
                'Chapter-6 (Contract)',
                'Chapter-7 (Performance of Contract)',
            ],
        },
    }
    
    # Legal synonyms and canonical forms
    LEGAL_SYNONYMS = {
        'what is': 'definition of',
        'how to': 'procedure for',
        'can i': 'eligibility for',
        'who can': 'eligibility criteria for',
        'rules': 'provisions',
        'laws': 'legal codes',
        'get married': 'marriage provisions',
        'get divorced': 'divorce provisions',
        'buy property': 'property acquisition',
        'own property': 'property ownership',
        'transfer property': 'property transfer',
        'inherit': 'succession rights',
        'punish': 'penalties',
        'punished': 'penalties for',
        'crime': 'offense',
        'liable': 'responsibility',
        'responsible': 'liability',
        'claim': 'entitlements',
        'rights': 'entitlements',
        'fundamental': 'constitutional',
        'nepali': 'nepal',
        'nepalese': 'nepal',
        'killed a dog': 'animal injury liability',
        'killed a cat': 'animal injury liability',
        'kill dog': 'harming animal'
        
    }
    
    # Term frequency in documents (helps with reformulation)
    DOCUMENT_TERMINOLOGY = {
        'shall': ['must', 'obligation', 'requirement'],
        'may': ['entitled', 'right', 'permission'],
        'provided that': ['condition', 'except', 'exception'],
        'not withstanding': ['despite', 'regardless'],
        'in accordance with': ['following', 'pursuant to', 'per'],
    }
    
    def normalize(self, query: str) -> NormalizedQuery:
        """
        Normalize a query for optimal retrieval
        
        Args:
            query: User query string
            
        Returns:
            NormalizedQuery with normalized form, intent, domain, key terms, reformulation
        """
        original = query.strip()
        
        # Step 1: Basic cleaning
        normalized = self._clean_query(original)
        
        # Step 2: Identify intent
        intent = self._classify_intent(normalized)
        
        # Step 3: Identify domain
        domain = self._classify_domain(normalized)
        
        # Step 4: Extract key terms
        key_terms = self._extract_key_terms(normalized)
        
        # Step 5: Reformulate query
        reformulated = self._reformulate_query(normalized, intent, domain)
        
        # Step 6: Suggest chapters
        suggested_chapters = self._suggest_chapters(domain, key_terms)
        
        # Step 7: Calculate confidence
        confidence = self._calculate_confidence(intent, domain, key_terms)
        
        return NormalizedQuery(
            original=original,
            normalized=normalized,
            intent=intent,
            domain=domain,
            key_terms=key_terms,
            reformulated=reformulated,
            suggested_chapters=suggested_chapters,
            confidence=confidence
        )
    
    def _clean_query(self, query: str) -> str:
        """Clean and standardize query"""
        # Lowercase
        query = query.lower()
        
        # Replace common punctuation with spaces
        query = re.sub(r'[?.,;:\-!]', ' ', query)
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query)
        
        # Apply synonym replacements
        for old, new in self.LEGAL_SYNONYMS.items():
            query = re.sub(rf'\b{old}\b', new, query)
        
        return query.strip()
    
    def _classify_intent(self, query: str) -> str:
        """Classify user intention"""
        intents_found = []
        
        for intent, pattern in self.INTENT_PATTERNS.items():
            if re.search(pattern, query):
                intents_found.append(intent)
        
        # Return primary intent or default to 'information'
        if intents_found:
            # Prioritize: definition, eligibility, procedure
            for priority in ['definition', 'eligibility', 'procedure']:
                if priority in intents_found:
                    return priority
            return intents_found[0]
        
        return 'information'
    
    def _classify_domain(self, query: str) -> str:
        """Classify legal domain"""
        domain_scores = {}
        
        for domain, pattern in self.DOMAIN_PATTERNS.items():
            matches = len(re.findall(pattern, query))
            if matches > 0:
                domain_scores[domain] = matches
        
        # Return highest scoring domain
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        
        return 'constitutional'  # Default to constitutional law
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract important legal terms"""
        # Split into words
        words = query.split()
        
        # Filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'be', 'have', 'has', 'do', 'does',
            'can', 'could', 'will', 'would', 'should', 'may', 'might', 'must',
            'this', 'that', 'these', 'those', 'i', 'you', 'we', 'he', 'she', 'it',
            'from', 'about', 'as', 'who', 'what', 'where', 'when', 'why', 'which',
            'how', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }
        
        key_terms = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in key_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)
        
        return unique_terms[:10]  # Limit to top 10
    
    def _reformulate_query(self, normalized: str, intent: str, domain: str) -> str:
        """Reformulate query to match document structure better"""
        reformulated = normalized
        
        # Build reformulated query based on intent and domain
        if intent == 'definition':
            reformulated = f"definition of {domain}: {normalized}"
        elif intent == 'eligibility':
            reformulated = f"{domain} eligibility requirements: {normalized}"
        elif intent == 'procedure':
            reformulated = f"how to {domain}: {normalized}"
        elif intent == 'right':
            reformulated = f"rights in {domain}: {normalized}"
        elif intent == 'obligation':
            reformulated = f"obligations in {domain}: {normalized}"
        elif intent == 'punishment':
            reformulated = f"{domain} penalties: {normalized}"
        
        # Add document type hint if clear domain
        if domain in self.CHAPTER_MAPPINGS:
            doc_types = list(self.CHAPTER_MAPPINGS[domain].keys())
            if doc_types:
                reformulated += f" [from {doc_types[0]}]"
        
        return reformulated
    
    def _suggest_chapters(self, domain: str, key_terms: List[str]) -> List[str]:
        """Suggest relevant chapters/sections"""
        chapters = []
        
        # Get chapters for primary domain
        if domain in self.CHAPTER_MAPPINGS:
            for doc_type, doc_chapters in self.CHAPTER_MAPPINGS[domain].items():
                chapters.extend(doc_chapters)
        
        # Also check if any key terms map to other domains
        for term in key_terms[:3]:  # Check top 3 key terms
            for domain_name, mapping in self.CHAPTER_MAPPINGS.items():
                if term in domain_name:
                    for doc_type, doc_chapters in mapping.items():
                        for ch in doc_chapters:
                            if ch not in chapters:
                                chapters.append(ch)
        
        return chapters[:5]  # Return top 5 suggestions
    
    def _calculate_confidence(self, intent: str, domain: str, key_terms: List[str]) -> float:
        """Calculate confidence in the normalization"""
        confidence = 0.5
        
        # Boost confidence based on intent clarity
        if intent not in ['information']:
            confidence += 0.15
        
        # Boost confidence based on domain clarity
        if domain not in ['constitutional']:  # Default domain
            confidence += 0.15
        
        # Boost confidence based on key terms
        if len(key_terms) >= 3:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def batch_normalize(self, queries: List[str]) -> List[NormalizedQuery]:
        """Normalize multiple queries"""
        return [self.normalize(q) for q in queries]
    
    def to_search_string(self, norm_query: NormalizedQuery) -> str:
        """Convert normalized query to optimal search string"""
        parts = [norm_query.reformulated]
        
        if norm_query.key_terms:
            parts.append(' '.join(norm_query.key_terms))
        
        search_string = ' '.join(parts)
        return search_string.strip()
