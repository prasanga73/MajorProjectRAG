# src/hybrid_retriever.py
import numpy as np
import faiss
import re
import pickle
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self, documents=None, index_dir="index_storage"):
        self.index_dir = index_dir
        
        # 1. Hardcoded Source Mapping
        self.alias_map = {
            "constitution": "Constitution of Nepal, 2015",
            "criminal": "National Criminal Code, 2017 AD",
            "penal": "National Criminal Code, 2017 AD",
            "civil": "National Civil Code, 2017 AD",
            "trafficking": "Human Trafficking and Transportation (Control) Act, 2007",
            "cyber": "Electronic Transactions Act, 2063 (2008)",
            "traffic": "Motor Vehicles and Transport Management Act, 2049",
            "labor": "Labor Act, 2074",
            "tax": "Income Tax Act, 2058",
            "banking": "Banks and Financial Institutions Act, 2073",
            "consumer": "Consumer Protection Act, 2075",
            "environment": "Environment Protection Act, 2076",
            "citizenship": "Nepal Citizenship Act, 2006",
            "domestic violence": "Domestic Violence (Crime and Punishment) Act, 2009",
            "dv act": "Domestic Violence (Crime and Punishment) Act, 2009"
        }
        
        # 2. KEYWORD TRIGGER MAP: Strictly routes specific topics to Chapters/Parts
# 2. COMPREHENSIVE TRIGGER MAP (Criminal, Civil, and Constitution)
        self.TRIGGER_MAP = {
            # ============================================================
            # 1. NATIONAL CRIMINAL CODE, 2017 AD
            # ============================================================
            
            # General Criminal Principles
            r"\b(criminal justice|conspiracy|attempt|abetment|accomplice|punishment|sentencing|mitigating|aggravating)\b": {
                "source": "National Criminal Code, 2017 AD",
                "chapter_keywords": ["General Principles of Criminal Justice", "Criminal Conspiracy, Attempt, Abetment and Accomplice", "Punishment and Interim Relief"]
            },
            
            # Offences Against the State and Public Order
            r"\b(treason|rebellion|sedition|insurrection|state|public tranquility|riot|unlawful assembly|contempt|public servant|justice|perjury)\b": {
                "source": "National Criminal Code, 2017 AD",
                "chapter_keywords": ["Offences Against the State", "Public Tranquility", "Contempt of Public Servants", "Public Justice"]
            },

            # Public Health and Safety
            r"\b(public health|adulteration|medicine|safety|morals|obscene|pollution)\b": {
                "source": "National Criminal Code, 2017 AD",
                "chapter_keywords": ["Public Interest, Health, Safety and Morals"]
            },

            # Weapons and Heritage
            r"\b(weapon|gun|arms|ammunition|explosive|bomb|heritage|ancient monument|archaeological)\b": {
                "source": "National Criminal Code, 2017 AD",
                "chapter_keywords": ["Arms and Ammunitions", "Explosives", "National and Public Heritage"]
            },

            # Discrimination and Religion (Criminal)
            r"\b(untouchability|discrimination|degrading treatment|caste|religion|religious site|blasphemy)\b": {
                "source": "National Criminal Code, 2017 AD",
                "chapter_keywords": ["Religion-related Offences", "Discrimination and Degrading Treatment"]
            },

            # Violence Against Human Body
            r"\b(murder|homicide|killing|kill|killed|kills|murder|murdered|suicide|assault|beating|hurt|grievous hurt|acid|injury)\b": {
                "source": "National Criminal Code, 2017 AD",
                "chapter_keywords": ["Human Body and Assault", "Hurt/Grievous Hurt"]
            },

            # Liberty and Kidnapping
            r"\b(detention|arrest|disappearance|kidnapping|hostage|abduction|unlawful detention)\b": {
                "source": "National Criminal Code, 2017 AD",
                "chapter_keywords": ["Unlawful Detention", "Enforced Disappearance", "Kidnapping/Hostage-taking"]
            },

            # Pregnancy and Sexual Offences
            r"\b(abortion|pregnancy|foetus|rape|sexual intercourse|harassment|incest|molestation)\b": {
                "source": "National Criminal Code, 2017 AD",
                "chapter_keywords": ["Protection of Pregnancy", "Sexual Offences"]
            },

            # Financial Crimes (Theft, Fraud, Forgery)
            r"\b(theft|robbery|stealing|burglary|robbed|stole|steals|robs|robbing)\b": {
                "source": "National Criminal Code, 2017 AD",
                "chapter_keywords": ["Theft and Robbery"]
            },
            r"\b(cheating|fraud|forgery|fake document|extortion|breach of trust)\b": {
                "source": "National Criminal Code, 2017 AD",
                "chapter_keywords": ["Offences Relating to Documents"]
            },
            
            r"\b(stamp)\b": {
                "source": "National Criminal Code, 2017 AD",
                "chapter_keywords": ["Stamps"]
            },

            # Privacy and Animal Cruelty
            r"\b(animal|killing animal|cow|bull|bird|dog)\b": {
                "source": "National Criminal Code, 2017 AD",
                "chapter_keywords": ["Animals and Birds","Animals","Birds"]
            },
            r"\b(privacy|wiretapping|confidentiality|eavesdropping)\b":{
                "source":"National Criminal Code, 2017 AD",
                "chapter_keywords":["Privacy Offences","Privacy"]
            },

            # Medical Negligence
            r"\b(medical|doctor|negligence|surgery|treatment|wrong treatment)\b": {
                "source": "National Criminal Code, 2017 AD",
                "chapter_keywords": ["Medical Treatment Offences"]
            },

            # ============================================================
            # 2. NATIONAL CIVIL CODE, 2017 AD
            # ============================================================
            
            # Persons and Rights
            r"\b(natural person|legal person|minor|incapacity|bankruptcy|civil rights)\b": {
                "source": "National Civil Code, 2017 AD",
                "chapter_keywords": ["Natural Persons", "Legal Persons", "Bankruptcy of Natural Persons", "Civil Rights"]
            },

            # Marriage and Family (Civil aspects)
            r"\b(marriage|divorce|wedding|wife|husband|separation|alimony|nullity)\b": {
                "sources": ["National Civil Code, 2017 AD", "National Criminal Code, 2017 AD"],
                "chapter_keywords": ["Marriage", "Divorce", "Consequences of Marriage", "Marriage-related"]
            },

            # Children and Adoption
            r"\b(adoption|adopted|adopt|child|guardian|curatorship|custody|maternal authority|paternal authority)\b": {
                "source": "National Civil Code, 2017 AD",
                "chapter_keywords": ["Relationship of Parents and Children", "Maternal and Paternal Authority", "Guardianship", "Curatorship", "Adoption"]
            },

            # Inheritance and Property Division
            r"\b(partition|succession|property|inheritance|will|legacy|ancestral property)\b": {
                "source": "National Civil Code, 2017 AD",
                "chapter_keywords": ["Partition", "Succession"]
            },

            # Land and Property Law
            r"\b(property|land|tenancy|ownership|possession|usufruct|servitude|rent|house rent|mortgage|pre-emption|registration of deeds)\b": {
                "source": "National Civil Code, 2017 AD",
                "chapter_keywords": ["General Provisions Relating to Property", "Ownership and Possession", "House Rent", "Mortgage of Immovable Property", "Registration of Deeds"]
            },

            # Contracts and Obligations
            r"\b(contract|agreement|guarantee|bailment|pledge|deposit|agency|sales|hire-purchase|wages|unjust enrichment|obligation)\b": {
                "source": "National Civil Code, 2017 AD",
                "chapter_keywords": ["Formation of Contracts", "Validity of Contracts", "Performance of Contracts", "Contracts of Sales of Goods", "Contracts of Agency"]
            },

            # Torts and Liability
            r"\b(tort|torts)\b": {
                "source": "National Civil Code, 2017 AD",
                "chapter_keywords": ["Provisions Relating to Torts"]
            },
            r"\b(damage|compensation|defective product|vicarious liability|defective|damaged|damaged product)\b": {
                "source": "National Civil Code, 2017 AD",
                "chapter_keywords": ["Defective Products"]
            },

            # ============================================================
            # 3. CONSTITUTION OF NEPAL, 2015
            # ============================================================
            
            # Fundamental Rights and Citizenship
            r"\b(fundamental right|citizen|citizenship|freedom|equality|justice|press|speech|expression|rights of woman|rights of dalit)\b": {
                "source": "Constitution of Nepal, 2015",
                "part_keywords": ["Fundamental Rights and Duties", "Citizenship"]
            },

            # Organs of State (Executive, Legislative, Judiciary)
            r"\b(president|vice-president|prime minister|council of ministers|parliament|house of representatives|national assembly|supreme court|judge|judiciary|attorney general|state legislature)\b": {
                "source": "Constitution of Nepal, 2015",
                "part_keywords": ["Federal Executive", "Federal Legislature", "Judiciary", "President and Vice-President", "Attorney General"]
            },

            # Federalism and Local Governance
            r"\b(federal|province|local level|municipality|village body|distribution of state power|federal power|provincial power|concurrent power)\b": {
                "source": "Constitution of Nepal, 2015",
                "part_keywords": ["Structure of State and Distribution of State Power", "Interrelations between Federation, State and Local level", "Local Executive", "Local Legislature"]
            },

            # Constitutional Commissions
            r"\b(ciaa|abuse of authority|corruption commission|auditor general|public service commission|election commission|human rights commission|tharu commission|madhesi commission)\b": {
                "source": "Constitution of Nepal, 2015",
                "part_keywords": ["Commission for the Investigation of Abuse of Authority", "Auditor General", "Public Service Commission", "Election Commission", "National Human Rights Commission"]
            },

            # Emergency and Security
            r"\b(emergency|crisis|suspension of rights|national security|army|police|political party)\b": {
                "source": "Constitution of Nepal, 2015",
                "part_keywords": ["Emergency Power", "Provision Relating National Security", "Provision relating to Political Parties"]
            },

            # Symbols and Miscellaneous
            r"\b(flag|anthem|coat of arms|official language|preamble|amendment|schedules|Nepal)\b": {
                "source": "Constitution of Nepal, 2015",
                "part_keywords": ["Preamble", "Amendment to the Constitution", "Short Title, Commencement and Repeal"]
            },
            
            #Extras
            r"\b(trafficking|human trafficking|selling person|prostitution|transportation of people)\b": {
                "source": "Human Trafficking and Transportation (Control) Act",
                "chapter_keywords": ["Trafficking"] # Empty list means "Match any chapter in this source"
            },

            # 5. DOMESTIC VIOLENCE ACT
            r"\b(domestic violence|domestic abuse|family violence|protection order|beating wife|husband abuse|beat my wife|beat my husband|husband beats|wife beats)\b": {
                "source": "Domestic Violence (Crime and Punishment) Act",
                "chapter_keywords": ["Domestic Violence"]
            },
            r"\b(women's rights|lineage|reproductive health|safe motherhood|proportional inclusion|equal property|affirmative action|positive discrimination)\b": {
                "source": "Constitution of Nepal, 2015",
                "part_keywords": ["Fundamental Rights"]
            },
            r"\b(citizenship|descent|birth|naturalized|nrn|non-resident|honorary citizenship|district administration office)\b": {
                "source": "Constitution of Nepal, 2015",
                "part_keywords": ["Citizenship"]
            },

            # === FAMILY and DOMESTIC VIOLENCE ===
            r"\b(domestic violence|domestic abuse|beating wife|mental torture|protection order|economic abuse|family member abuse|physical harm)\b": {
                "source": "Domestic Violence (Crime and Punishment) Act, 2009",
                "chapter_keywords": ["Domestic Violence"]
            },
            r"\b(muslim|nikah|talaq|triple talaq|mahr|dower|polygamy|bigamy|multiple wives)\b": {
                "sources": ["Muluki","Criminal","Civil","Muluki Civil (Code) Act, 2074 (2017), Section 173"],
                "chapter_keywords": ["Marriage", "Divorce","Polygamy","Prohibition of Polygamy"]
            },
            r"\b(prenuptial|postnuptial|alimony|separation of assets|marital property agreement|section 102)\b": {
                "source": "Civil Code",
                "chapter_keywords": ["Marriage", "Divorce", "Property Partition"]
            },

            # === SPECIFIC CRIMINAL OFFENCES (Muluki Ain / Penal Code) ===
            r"\b(rape|marital rape|sexual assault|sexual exploitation|forced nudity|prostitution|blackmail)\b": {
                "sources": ["Muluki Ain (General Code)","Criminal Code"],
                "chapter_keywords": ["Rape", "Sexual Offenses"]
            },
            r"\b(acid attack|corrosive|hazardous chemicals|section 193)\b": {
                "sources": ["The Acid and Other Hazardous Chemicals (Regulation) Act", "Criminal Code"],
                "chapter_keywords": ["Hurt and Battery"]
            },
            r"\b(witchcraft|bokxi|boksi|witch|accusation of witch|witch accusation|superstition)\b": {
                "source": "Anti-Witchcraft (Crime and Punishment) Act",
                "chapter_keywords": ["Witchcraft"]
            },
            r"\b(child marriage|age of 20|early marriage)\b": {
                "sources": ["Muluki Ain (General Code)"],
                "chapter_keywords": ["Child Marriage","Offences Relating to Marriage"]
            },
            r"\b(infanticide|honor killing|homicide|murder|abduction|kidnapping)\b": {
                "source": ["Muluki Ain (General Code)","Criminal Code"],
                "chapter_keywords": ["Homicide", "Abduction","Offences Relating to Human Body"]
            },

            # === TRAFFIC and VEHICLES (Ma Pa Se, License, Registration) ===
            r"\b(traffic|drunk driving|ma pa se|ma-pa-se|alcohol|speeding|helmet|seatbelt|reckless driving|accident)\b": {
                "source": "Motor Vehicles and Transport Management Act",
                "chapter_keywords": ["Traffic Rules","Traffic"]
            },
            r"\b(license|driving license|10 year validity|renew license|bluebook|registration|embossed plate|number plate|ownership transfer)\b": {
                "source": "Motor Vehicles and Transport Management Act",
                "chapter_keywords": ["Vehicle Registration", "Bluebook"]
            },
            r"\b(insurance|third party|3rd party|liability|beema samiti|comprehensive insurance)\b": {
                "source": "Motor Vehicles and Transport Management Act",
                "chapter_keywords": ["Insurance"]
            },

            # === LABOR and EMPLOYMENT ===
            r"\b(labor|minimum wage|19550|overtime|overtime pay|maternity leave|paternity leave|sick leave|working hours|8 hours)\b": {
                "source": "Labor Act",
                "chapter_keywords": ["Labor"]
            },
            r"\b(ssf|social security fund|provident fund|gratuity|pension|contribution based)\b": {
                "source": "Contribution-Based Social Security Act",
                "chapter_keywords": ["Social Security"]
            },
            r"\b(termination|dismissal|resignation|labor court|dispute|trade union)\b": {
                "source": "Labor Act, 2074",
                "chapter_keywords": ["Termination", "Dispute Resolution","Labor"]
            },

            # === CYBER CRIME and ELECTRONIC TRANSACTIONS ===
            r"\b(cyber|hacking|unauthorized access|hacking|computer fraud|online harassment|deepfake|social media scam|phishing|e-wallet fraud|doxxing)\b": {
                "source": "Electronic Transactions Act, 2063 (2008)",
                "chapter_keywords": ["Electronic Transactions"]
            },

            # === BUSINESS, TAX and BANKING ===
            r"\b(register company|ocr|camis|private limited|sole proprietorship|fdi|foreign investment|repatriation)\b": {
                "source": "Companies Act",
                "chapter_keywords": ["Business Setup"]
            },
            r"\b(tax|income tax|vat|excise|pan|tax slabs|13%|evasion|ird|inland revenue)\b": {
                "source": "Income Tax Act",
                "chapter_keywords": ["Tax","Income Tax"]
            },
            r"\b(banking|nrb|bafia|cheque bounce|aml|cft|money laundering|kyc|blacklisting|loan fraud)\b": {
                "source": "Banks and Financial Institutions Act",
                "chapter_keywords": ["Banks","Financial Institutions"]
            },

            # === LAND and PROPERTY ===
            r"\b(land|lalpurja|ownership|mohi|tenancy|land ceiling|ropani|land revenue|acquisition|fragmentation)\b": {
                "source": "Land Act",
                "chapter_keywords": ["Land"]
            },

            # === PUBLIC SERVICES (Health, Education, Reservation) ===
            r"\b(medical negligence|hospital|free basic health|organ transplant|epidemic|doctor|negligence|consumer court)\b": {
                "source": "Public Health",
                "chapter_keywords": ["Public Health", "Medical Law"]
            },
            r"\b(education|compulsory education|grade 1-8|teacher license|scholarship|school licensing|fee regulation)\b": {
                "source": "Education Act, 2028 B.S.",
                "chapter_keywords": ["Education Law"]
            },
            r"\b(reservation|quota|dalit|janajati|madhesi|33% women|inclusion|public service commission|psc)\b": {
                "source": "Constitution of Nepal, 2015",
                "part_keywords": ["Fundamental Rights", "Reservation"]
            },

            # === CONSUMER and ENVIRONMENT ===
            r"\b(consumer|misleading ad|overcharging|labeling|defective product|fake discount|compensation)\b": {
                "source": "Consumer Protection Act, 2075",
                "chapter_keywords": ["Consumer Protection","Consumer"]
            },
            r"\b(environment|pollution|eia|iee|climate change|solid waste|noise pollution|forest|biodiversity)\b": {
                "source": "Environment Protection Act, 2076",
                "chapter_keywords": ["Environment Protection","Environment"]
            },
            r"\b(copyright|trademark|patent|industrial design|gi|ilam tea|software protection|infringement)\b": {
                "source": "Copyright Act, 2059",
                "chapter_keywords": ["Intellectual Property","Copyright"]
            }
            
            
        }

        # 3. Models
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
        
        if documents is not None:
            self.docs = documents
            self._build_index()
        else:
            self._load_index()

    def _build_index(self):
        print("[*] Encoding vectors and building BM25...")
        self.raw_texts = [d['search_content'] for d in self.docs]
        embeddings = self.model.encode(self.raw_texts, show_progress_bar=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings.astype('float32'))
        tokenized_corpus = [doc.lower().split() for doc in self.raw_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def save_index(self):
        if not os.path.exists(self.index_dir): os.makedirs(self.index_dir)
        faiss.write_index(self.index, os.path.join(self.index_dir, "faiss.index"))
        with open(os.path.join(self.index_dir, "data.pkl"), "wb") as f:
            pickle.dump({"docs": self.docs, "bm25": self.bm25}, f)
        print(f"[*] Index saved to '{self.index_dir}'")

    def _load_index(self):
        print("[*] Loading index from disk...")
        self.index = faiss.read_index(os.path.join(self.index_dir, "faiss.index"))
        with open(os.path.join(self.index_dir, "data.pkl"), "rb") as f:
            data = pickle.load(f)
            self.docs = data["docs"]
            self.bm25 = data["bm25"]
        print("[*] Index loaded successfully.")

    def _parse_query(self, query):
        """Extracts Source and Clause ID (e.g., 'clause 22')."""
        q = query.lower()
        target_src, target_id = None, None
        for alias, formal in self.alias_map.items():
            if alias in q:
                target_src = formal
                break
        match = re.search(r"\b(clause|section|article|sec|art|no\.?)\s*([0-9a-zA-Z\(\)]+)", q)
        if match:
            target_id = match.group(2).strip().lower()
        return target_src, target_id

    def _get_active_filters(self, query):
        """Identifies if the query should be restricted to specific chapters/parts."""
        q = query.lower()
        filters = []
        for pattern, config in self.TRIGGER_MAP.items():
            if re.search(pattern, q):
                filters.append(config)
        return filters

    def _apply_filters(self, doc_indices, filters):
        if not filters: return doc_indices
        filtered_indices = []
        
        for idx in doc_indices:
            doc = self.docs[idx]
            meta = doc['metadata']
            
            # Normalize Document Info
            doc_src = str(meta.get('legal_document_source') or "").lower()
            doc_ch = str(meta.get('chapter') or "").lower()
            doc_text = str(doc.get('search_content') or "").lower()

            match_found = False
            for f in filters:
                # 1. Gather all potential source targets from the trigger map
                targets = []
                if f.get('source'): targets.append(f['source'].lower())
                if f.get('sources'): targets.extend([s.lower() for s in f['sources']])
                
                # 2. LOOSE SOURCE MATCH: Check if any target is a substring of the doc source
                # Or if the doc source is a substring of the target
                src_match = any((t in doc_src or doc_src in t) for t in targets)
                
                if src_match:
                    # 3. LOOSE KEYWORD MATCH
                    all_kws = [str(kw).lower() for kw in (f.get('chapter_keywords', []) + f.get('part_keywords', []))]
                    
                    if not all_kws:
                        match_found = True
                        break
                    
                    # Check if keywords appear in Chapter, Source Name, or the actual Text
                    kw_match = any(kw in doc_ch for kw in all_kws) or \
                               any(kw in doc_src for kw in all_kws) or \
                               any(kw in doc_text for kw in all_kws)
                    
                    if kw_match:
                        match_found = True
                        break
            
            if match_found:
                filtered_indices.append(idx)
                
        return filtered_indices
        
    def _rerank(self, query, candidate_docs):
        if not candidate_docs: return []
        pairs = [[query, doc['search_content']] for doc in candidate_docs]
        scores = self.reranker.predict(pairs)
        for i, score in enumerate(scores): candidate_docs[i]['rerank_score'] = score
        return sorted(candidate_docs, key=lambda x: x['rerank_score'], reverse=True)

    def _deduplicate_and_group(self, flat_results):
        grouped = {}
        for item in flat_results:
            meta = item['metadata']
            parent_key = (meta['legal_document_source'], meta['parent_clause_id'])
            if parent_key not in grouped:
                grouped[parent_key] = {
                    "legal_document_source": meta['legal_document_source'],
                    "parent_clause_id": meta['parent_clause_id'],
                    "parent_clause_text": meta['parent_clause_text'],
                    "chapter": meta.get('chapter', 'N/A'),
                    "part": meta.get('part', 'N/A'),
                    "sub_clauses": [],
                    "score": item.get('rerank_score', 0)
                }
            if not any(s['id'] == meta['clause_id'] for s in grouped[parent_key]["sub_clauses"]):
                grouped[parent_key]["sub_clauses"].append({"id": meta['clause_id'], "text": meta['text']})
        return sorted(list(grouped.values()), key=lambda x: x['score'], reverse=True)

    def hybrid_search(self, query, top_k=5):
        target_src, target_id = self._parse_query(query)
        active_filters = self._get_active_filters(query)
        
        # --- PATH 1: DIRECT HIT ---
        raw_results = []
        if target_id:
            for d in self.docs:
                meta = d['metadata']
                id_match = (meta['clause_id'] == target_id or meta['parent_clause_id'] == target_id)
                if target_src:
                    if meta['legal_document_source'] == target_src and id_match: raw_results.append(d)
                elif id_match: raw_results.append(d)
            if raw_results: return self._deduplicate_and_group(raw_results)[:top_k]

        # --- PATH 2: HYBRID SEARCH ---
        search_depth = 1000 if active_filters else 100
        
        if active_filters:
            print(f"[*] Strict Filtering Active. Searching deeper (top {search_depth})...")
            
        
        triggered_names = [f.get('source') or f.get('sources') for f in active_filters]
        if active_filters: print(f"[*] Strict Filtering Active for keywords in: {[f.get('chapter_keywords') or f.get('part_keywords') for f in active_filters]}")
        if active_filters: print(f"Legal Source:{triggered_names}")
        
        # Retrieve larger pool for filtering (search_depth = 100 if no filter 500 if filter)
        query_vec = self.model.encode([query]).astype('float32')
        _, v_indices = self.index.search(query_vec, search_depth)
        v_indices = v_indices[0].tolist()
        
        scores = self.bm25.get_scores(query.lower().split())
        b_indices = np.argsort(scores)[::-1][:search_depth].tolist()

        # Apply Metadata Scoping
        v_indices = self._apply_filters(v_indices, active_filters)
        b_indices = self._apply_filters(b_indices, active_filters)

        # RRF
        rrf_scores = {}
        for rank, idx in enumerate(v_indices): rrf_scores[idx] = rrf_scores.get(idx, 0) + 1/(rank+60)
        for rank, idx in enumerate(b_indices): rrf_scores[idx] = rrf_scores.get(idx, 0) + 1/(rank+60)
        
        sorted_idx = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        top_candidates = [self.docs[i] for i in sorted_idx[:25]]

        # Rerank and Group
        reranked = self._rerank(query, top_candidates)
        return self._deduplicate_and_group(reranked)[:top_k]