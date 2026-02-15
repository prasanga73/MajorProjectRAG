"""
Query and Clause ID Extraction Module
Handles extraction of clause IDs, document types, and action types from queries
"""

import re
from typing import List, Optional, Dict


class QueryExtractor:
    """Extracts structured information from legal queries"""
    
    # Action verbs that indicate clause reference
    ACTION_VERBS = [
        'fetch', 'get', 'show', 'display', 'retrieve', 'find',
        'explain', 'summarize', 'describe', 'tell', 'what',
        'give', 'provide', 'read', 'view', 'see', 'look'
    ]
    
    # Document type patterns
    DOC_PATTERNS = {
        'civil': r'\b(civil|civil\s+code|national\s+civil)\b',
        'criminal': r'\b(criminal|penal|criminal\s+code|penal\s+code)\b',
        'constitution': r'\b(constitution|constitutional|nepal\s+constitution)\b'
    }
    
    @classmethod
    def extract_clause_ids(cls, query: str) -> List[str]:
        """
        Extract clause IDs from query with improved pattern matching
        
        Supports formats like:
        - "clause 22"
        - "22(a)(1)"
        - "section 1(2)"
        - "fetch clause 22"
        - "summarize clause 22 of criminal code"
        - "2(h)" or "fetch 2(h)"
        """
        clause_ids = []
        query_lower = query.lower()
        
        # Pattern 1: Complex nested clauses like 22(a)(1)
        pattern = r'(\d+)\s*\(([a-z])\)\s*\((\d+)\)'
        for match in re.finditer(pattern, query_lower):
            clause_id = f"{match.group(1)}({match.group(2)})({match.group(3)})"
            if clause_id not in clause_ids:
                clause_ids.append(clause_id)
        
        # Pattern 2: Sub-clauses like 22(a) or 1(2) or 2(h)
        pattern = r'(\d+)\s*\(([a-z0-9]+)\)'
        for match in re.finditer(pattern, query_lower):
            clause_id = f"{match.group(1)}({match.group(2)})"
            if clause_id not in clause_ids:
                is_subpattern = any(cid.startswith(clause_id + '(') for cid in clause_ids)
                if not is_subpattern:
                    clause_ids.append(clause_id)
        
        # Pattern 3: Explicit clause/section references
        keywords = ['clause', 'section', 'article', 'provision', 'rule', 
                   'paragraph', 'subsection', 'subclause']
        
        for keyword in keywords:
            pattern = rf'\b{keyword}\s+(\d+)\s*\(([a-z0-9]+)\)'
            for match in re.finditer(pattern, query_lower):
                clause_id = f"{match.group(1)}({match.group(2)})"
                if clause_id not in clause_ids:
                    clause_ids.append(clause_id)
            
            pattern = rf'\b{keyword}\s+(\d+)\b'
            for match in re.finditer(pattern, query_lower):
                clause_num = match.group(1)
                if clause_num not in clause_ids:
                    has_subclause = any(cid.startswith(clause_num + '(') for cid in clause_ids)
                    if not has_subclause:
                        clause_ids.append(clause_num)
        
        # Pattern 4: Action verb + clause patterns
        for verb in cls.ACTION_VERBS:
            pattern = rf'\b{verb}\s+(?:clause\s+|section\s+)?(\d+)\s*\(([a-z0-9]+)\)'
            for match in re.finditer(pattern, query_lower):
                clause_id = f"{match.group(1)}({match.group(2)})"
                if clause_id not in clause_ids:
                    clause_ids.append(clause_id)
            
            pattern = rf'\b{verb}\s+(?:clause\s+|section\s+)?(\d{{1,3}})\b'
            for match in re.finditer(pattern, query_lower):
                clause_num = match.group(1)
                num = int(clause_num)
                if num < 1000 and clause_num not in clause_ids:
                    has_subclause = any(cid.startswith(clause_num + '(') for cid in clause_ids)
                    if not has_subclause:
                        clause_ids.append(clause_num)
        
        # Pattern 5: Standalone numbers in legal context
        legal_keywords = ['clause', 'section', 'article', 'code', 'act', 
                         'law', 'provision', 'rule', 'civil', 'criminal', 
                         'constitution', 'of', 'in']
        
        if any(kw in query_lower for kw in legal_keywords):
            pattern = r'\b(\d{1,3})\b'
            for match in re.finditer(pattern, query_lower):
                clause_num = match.group(1)
                num = int(clause_num)
                if num < 1000 and clause_num not in clause_ids:
                    has_subclause = any(cid.startswith(clause_num + '(') for cid in clause_ids)
                    if not has_subclause:
                        match_pos = match.start()
                        context_before = query_lower[max(0, match_pos-30):match_pos]
                        context_after = query_lower[match_pos:min(len(query_lower), match_pos+30)]
                        
                        if any(kw in context_before or kw in context_after 
                              for kw in legal_keywords):
                            clause_ids.append(clause_num)
        
        return clause_ids
    
    @classmethod
    def extract_document_type(cls, query: str) -> Optional[str]:
        """
        Extract legal document type from query
        Returns: 'civil', 'criminal', 'constitution', or None
        """
        query_lower = query.lower()
        
        for doc_type, pattern in cls.DOC_PATTERNS.items():
            if re.search(pattern, query_lower):
                return doc_type
        
        return None
    
    @classmethod
    def extract_action_type(cls, query: str) -> Optional[str]:
        """Extract the action type from query (fetch, summarize, explain, etc.)"""
        query_lower = query.lower()
        
        for verb in cls.ACTION_VERBS:
            if re.search(rf'\b{verb}\b', query_lower):
                return verb
        
        return None
    
    @classmethod
    def parse_query(cls, query: str) -> Dict:
        """Complete query parsing - returns all extracted information"""
        return {
            'clause_ids': cls.extract_clause_ids(query),
            'document_type': cls.extract_document_type(query),
            'action_type': cls.extract_action_type(query),
            'original_query': query
        }
