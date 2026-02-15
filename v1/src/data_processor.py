import json
import os
import re

class LegalDocProcessor:
    def __init__(self, parent_path, child_path):
        self.parent_path = parent_path
        self.child_path = child_path
        # RECOMENDED: Simplify these to base keywords for maximum "looseness"
        self.allowed_sources = [
            "Constitution",
            "Criminal Code",
            "Civil Code",
            "Electronic Transactions",
            "Domestic Violence",
            "Human Trafficking",
            "Motor Vehicles",
            "Labor Act",
            "Income Tax",
            "Banking",
            "Consumer Protection",
            "Environment",
            "Citizenship",
            "Witchcraft",
            "Acid",
            "Muluki Ain",
            "Land Act",
            "Public Health",
            "Copyright Act",
            "Education Act",
            "Public Health",
            "Banks",
            "Companies Act",
            "Muluki Civil"
        ]

    def _get_base_clause(self, clause_id):
        if not clause_id: return None
        match = re.match(r"([0-9A-Za-z]+)", str(clause_id))
        return match.group(1) if match else str(clause_id)

    # NEW HELPER: Reusable loose check
    def _is_source_allowed(self, src_name):
        if not src_name: return False
        src_lower = str(src_name).lower()
        return any(allowed.lower() in src_lower for allowed in self.allowed_sources)

    def load_and_clean(self):
        parent_lookup = {}
        processed_docs = []

        # 1. PROCESS PARENTS (Now with loose matching)
        if os.path.exists(self.parent_path):
            with open(self.parent_path, 'r', encoding='utf-8') as f:
                parents = json.load(f)
                for p in parents:
                    src = p.get('legal_document_source', "").strip()
                    
                    # LOOSE CHECK APPLIED HERE
                    if self._is_source_allowed(src):
                        cid = str(p.get('clause_id')).strip().lower()
                        # Use (src, cid) to match exactly how children identify parents
                        parent_lookup[(src, cid)] = p.get('text')

        # 2. PROCESS CHILDREN
        if os.path.exists(self.child_path):
            with open(self.child_path, 'r', encoding='utf-8') as f:
                children = json.load(f)
                for child in children:
                    src = child.get('legal_document_source', "").strip()
                    
                    # LOOSE CHECK APPLIED HERE
                    if not self._is_source_allowed(src):
                        continue
                    
                    raw_id = str(child.get('clause_id')).strip().lower()
                    raw_p_id = str(child.get('parent_clause_id') or child.get('clause_id')).strip().lower()
                    base_p_id = self._get_base_clause(raw_p_id).lower()
                    
                    # Try to find parent using the exact source name found in this chunk
                    p_text = parent_lookup.get((src, raw_p_id)) or \
                             parent_lookup.get((src, base_p_id), "Parent context not found.")

                    processed_docs.append({
                        "search_content": child.get('text', ""), 
                        "metadata": {
                            "clause_id": raw_id,
                            "text": child.get('text'),
                            "legal_document_source": src,
                            "parent_clause_id": base_p_id,
                            "parent_clause_text": p_text,
                            "chapter": child.get('chapter', ""),
                            "part": child.get('part', "")
                        }
                    })
        
        return processed_docs