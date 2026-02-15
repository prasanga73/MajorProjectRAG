"""
Legal RAG System with Deduplication
Main retrieval module with improved parent clause deduplication
"""

import json
import numpy as np
from typing import List, Dict, Optional

from .extractors import QueryExtractor
from .query_normalizer import QueryNormalizer, NormalizedQuery
from .scorers import create_reranker
from .semantic_filter import SemanticDomainFilter
from .utils import (
    BM25, ProductionEmbeddingModel, SearchResult, RetrievalResult,
    load_json, format_results
)


class LegalRAGSystem:
    """
    Legal Retrieval-Augmented Generation System
    Combines semantic search, BM25, reranking with deduplication of parent clauses
    """
    
    def __init__(
        self,
        child_docs_path: str,
        parent_docs_path: str,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        use_reranker: bool = True,
        reranker_type: str = "hybrid",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        similarity_weight: float = 0.6,
        bm25_weight: float = 0.4,
        rerank_weight: float = 0.5,
        initial_weight: float = 0.5,
        direct_hit_bonus: float = 2.0,
        rerank_top_k: int = 20,
        enable_semantic_filter: bool = True,
        semantic_filter_threshold: float = 0.4,
        semantic_filter_weight: float = 0.7,
        verbose: bool = True
    ):
        """Initialize RAG system
        
        Args:
            enable_semantic_filter: Enable domain coherence filtering
            semantic_filter_threshold: Minimum coherence score (0-1) to keep result
            semantic_filter_weight: How much domain penalty affects final score (0-1)
        """
        self.verbose = verbose
        self.similarity_weight = similarity_weight
        self.bm25_weight = bm25_weight
        self.direct_hit_bonus = direct_hit_bonus
        self.use_reranker = use_reranker
        self.rerank_top_k = rerank_top_k
        self.enable_semantic_filter = enable_semantic_filter
        self.semantic_filter_threshold = semantic_filter_threshold
        self.semantic_filter_weight = semantic_filter_weight
        
        if self.verbose:
            print("[INFO] Initializing Legal RAG System...")
        
        # Load documents
        self.child_docs = load_json(child_docs_path)
        self.parent_docs = load_json(parent_docs_path)
        
        if self.verbose:
            print(f"  [OK] Loaded {len(self.child_docs)} child documents")
            print(f"  [OK] Loaded {len(self.parent_docs)} parent documents")
        
        # Initialize components
        self.embedding_model = ProductionEmbeddingModel(embedding_model_name)
        self.bm25 = BM25()
        self.clause_extractor = QueryExtractor()
        self.query_normalizer = QueryNormalizer()  # Initialize query normalizer
        
        # Keyword expansion dictionary for natural language queries
        self.keyword_expansions = {
            'citizenship': ['citizen', 'citizens', 'citizenship', 'naturalized'],
            'nepal': ['nepal', 'nepali', 'nepalese', 'nation', 'state', 'country', 'republic'],
            'nation': ['nation', 'national', 'country', 'state', 'nepal'],
            'constitution': ['constitution', 'constitutional', 'fundamental', 'law', 'article'],
            'civil': ['civil', 'civil code', 'civil law'],
            'criminal': ['criminal', 'criminal code', 'penal', 'penal code', 'crime'],
            'property': ['property', 'properties', 'possession', 'movable', 'immovable', 'ownership'],
            'marriage': ['marriage', 'marital', 'matrimonial', 'spouse', 'husband', 'wife'],
            'inheritance': ['inheritance', 'heir', 'succession', 'legacy', 'will', 'ancestor'],
            'contract': ['contract', 'agreement', 'obligation', 'commitment', 'terms'],
            'liability': ['liability', 'liable', 'responsible', 'responsibility', 'compensation'],
            'right': ['right', 'rights', 'entitlement', 'claim', 'freedom'],
            'discrimination': ['discrimination', 'discriminate', 'discriminatory', 'equality'],
        }
        
        # COMPREHENSIVE DOCUMENT KEYWORD ROUTING LIST
        # Automatically routes queries to relevant documents based on keywords
        self.document_keyword_mapping = {
            'Constitution of Nepal, 2015': {
                'keywords': [
                    # Political structure
                    'citizenship', 'citizen', 'constitution', 'constitutional', 'state',
                    'nepal', 'nepali', 'nation', 'national', 'parliament', 'legislature',
                    'executive', 'judicial', 'president', 'prime minister',
                    
                    # Fundamental rights
                    'right', 'rights', 'fundamental', 'freedom', 'liberty', 'equality',
                    'justice', 'discrimination', 'equal', 'life', 'liberty',
                    
                    # Governance
                    'government', 'governance', 'authority', 'public', 'election',
                    'voting', 'representative', 'amendment', 'article', 'schedule',
                    
                    # National concepts
                    'sovereignty', 'integrity', 'federal', 'division', 'province',
                    'local', 'decentralization', 'security', 'defence',
                    
                    # Administrative
                    'republic', 'secular', 'democratic', 'parliament', 'cabinet',
                    'minister', 'ministry', 'commission', 'council',
                ],
                'weight': 1.0  # Higher weight = prioritize this document
            },
            
            'National Civil Code, 2017 AD': {
                'keywords': [
                    # Family law
                    'marriage', 'marital', 'matrimonial', 'spouse', 'husband', 'wife',
                    'divorce', 'separation', 'annulment', 'dissolution', 'bigamy',
                    'adoption', 'child', 'children', 'parent', 'parental', 'guardianship',
                    'guardian', 'custodian', 'custody', 'succession', 'heir', 'inheritance',
                    'will', 'testator', 'testament', 'legacy', 'estate', 'succession',
                    
                    # Property law
                    'property', 'properties', 'ownership', 'owner', 'possess', 'possession',
                    'movable', 'immovable', 'land', 'house', 'building', 'real estate',
                    'mortgage', 'mortgage', 'pledge', 'lien', 'security', 'transfer',
                    'sale', 'purchase', 'conveyance', 'lease', 'rent', 'tenant', 'landlord',
                    'pre-emption', 'pre-empt', 'exclusive', 'exclusive right',
                    
                    # Contracts & Obligations
                    'contract', 'agreement', 'obligation', 'legally binding', 'parties',
                    'consideration', 'offer', 'acceptance', 'terms', 'condition',
                    'validity', 'formation', 'breach', 'breach of contract', 'remedy',
                    'damages', 'specific performance', 'penalty', 'compensation',
                    
                    # Commercial transactions
                    'goods', 'sale of goods', 'merchant', 'buyer', 'seller', 'vendor',
                    'price', 'payment', 'delivery', 'shipment', 'warranty',
                    'bailment', 'bailee', 'bailor', 'pledge', 'hypothecation',
                    'agency', 'agent', 'principal', 'commission',
                    
                    # Torts & Liability
                    'tort', 'torts', 'liability', 'liable', 'tortious', 'negligence',
                    'injury', 'damage', 'harm', 'wrongful', 'trespass', 'defamation',
                    'breach of duty', 'duty of care', 'causation', 'vicarious liability',
                    'product liability', 'defective product', 'strict liability',
                    
                    # General provisions
                    'civil code', 'civil law', 'provision', 'section', 'clause',
                    'person', 'legal entity', 'corporation', 'firm', 'partnership',
                    'minor', 'majority', 'capacity', 'incompetent', 'guardian',
                ],
                'weight': 1.0
            },
            
            'National Criminal Code, 2017 AD': {
                'keywords': [
                    # Serious crimes
                    'crime', 'crimes', 'criminal', 'offense', 'offence', 'offence',
                    'murder', 'homicide', 'killing', 'kill', 'killed', 'death',
                    'assault', 'battery', 'violence', 'violent', 'aggression',
                    'rape', 'sexual assault', 'sexual offense', 'sexual crime',
                    'robbery', 'theft', 'steal', 'stealing', 'stolen', 'larceny',
                    'burglary', 'breaking', 'trespassing', 'breaking and entering',
                    'fraud', 'cheating', 'deceive', 'deception', 'forgery', 'counterfeit',
                    'embezzlement', 'misappropriation', 'breach of trust', 'criminal breach',
                    
                    # Against person/body
                    'hurt', 'grievous', 'grievous hurt', 'bodily injury', 'injury',
                    'detain', 'detention', 'unlawful detention', 'kidnapping', 'abduction',
                    'disappearance', 'enforced disappearance', 'hostage',
                    'harassment', 'intimidation', 'threats', 'threatening',
                    'attempt', 'conspiracy', 'abetment', 'accessory',
                    
                    # Against property/state
                    'mischief', 'criminal mischief', 'vandalism', 'arson', 'fire',
                    'currency', 'counterfeit currency', 'stamps', 'documents',
                    'forgery', 'falsification', 'adulteration', 'weights and measures',
                    'smuggling', 'contraband', 'contraband goods',
                    
                    # Against public/state
                    'treason', 'sedition', 'rebellion', 'insurrection', 'mutiny',
                    'terrorism', 'terrorist', 'bomb', 'bombing', 'explosive',
                    'weapons', 'arms', 'ammunition', 'firearms', 'illegal weapons',
                    'public tranquility', 'public order', 'peace', 'disturbing peace',
                    'contempt', 'contempt of authority', 'public authority',
                    
                    # Environmental/Heritage
                    'environment', 'environmental', 'pollution', 'hazardous',
                    'heritage', 'protected area', 'wildlife', 'endangered species',
                    'animal', 'cruelty', 'animal cruelty', 'bird protection',
                    'forest', 'timber', 'logging', 'deforestation',
                    
                    # Sentencing & Procedure
                    'punishment', 'sentence', 'fine', 'imprisonment', 'custodial',
                    'penalty', 'penalties', 'aggravating', 'mitigating', 'circumstances',
                    'gravity', 'first offense', 'repeat offense', 'recidivist',
                    'imprisonment', 'duration', 'term', 'years', 'months', 'days',
                    'pardon', 'amnesty', 'commutation', 'remission', 'relief',
                    
                    # General criminal
                    'criminal code', 'penal', 'penal code', 'offender', 'perpetrator',
                    'accused', 'charge', 'charges', 'charged', 'conviction',
                    'guilty', 'guilty verdict', 'evidence', 'confession',
                    'accomplice', 'co-accused', 'principal', 'accessory after fact',
                ],
                'weight': 1.0
            }
        }
        
        # Query-to-clause boosting for fundamental constitutional questions
        # PHASE 1: ENTITY CONTEXT DETECTION
        # Detects what entity is being discussed: person, animal, property, business
        self.entity_context_keywords = {
            'animal': {
                'keywords': ['dog', 'cat', 'animal', 'pet', 'bird', 'horse', 'cow', 'goat', 'elephant', 'monkey', 'livestock', 'cattle'],
                'context': 'animal',
                'blocks_criminal_homicide': True,  # Killing animal != killing person
                'route_to_domains': ['property', 'liability', 'tort']
            },
            'person': {
                'keywords': ['person', 'man', 'woman', 'child', 'human', 'people', 'someone', 'victim', 'accused', 'defendant', 'plaintiff'],
                'context': 'person',
                'blocks_criminal_homicide': False,
                'route_to_domains': ['criminal', 'homicide']
            },
            'business': {
                'keywords': ['company', 'business', 'shop', 'store', 'firm', 'enterprise', 'corporation', 'organization', 'venture'],
                'context': 'business',
                'blocks_criminal_homicide': True,
                'route_to_domains': ['contract', 'property', 'commercial']
            },
            'property': {
                'keywords': ['property', 'house', 'land', 'building', 'asset', 'possession', 'item', 'object', 'goods'],
                'context': 'property',
                'blocks_criminal_homicide': True,
                'route_to_domains': ['property', 'ownership', 'theft']
            }
        }
        
        # PHASE 2: MULTI-WORD PHRASE MATCHING
        # Context-aware phrase matching for disambiguation
        self.phrase_matching_rules = {
            # HARMFUL ACTION + ENTITY COMBINATIONS
            'kill animal': {
                'patterns': ['kill (a |an |the )?dog','killed (a |an |the )?dog','killed (a |an |the )?cat','kill (a |an |the )?cat', 'kill animal', 'injure (a |an |the )?dog', 'harm (a |an |the )?pet'],
                'domain': 'animal_liability',
                'boost_keywords': ['animal','cruelty'],
                'block_keywords': ['homicide', 'murder', 'manslaughter'],
                'boost_chapters': ['Chapter-27 (Offences Relating to Animals)'],
                'exclude_chapters': ['Chapter-12 (Offences Relating to Human Body)'],
                'chapter_boost_multiplier': 2.5,
                'penalty_multiplier': 0.01
            },
            'kill person': {
                'patterns': ['kill (a |an |the )?(person|man|woman|human|child)', 'murder', 'homicide', 'fatal'],
                'domain': 'homicide',
                'boost_keywords': ['crime', 'offense', 'homicide', 'murder'],
                'block_keywords': ['animal', 'dog', 'cat', 'pet', 'livestock'],
                'boost_chapters': ['Chapter-12 (Offences Relating to Human Body)'],
                'exclude_chapters': ['Chapter-27 (Offences Relating to Animals)'],
                'chapter_boost_multiplier': 2.5
            },
            
            # THEFT VARIANTS
            'steal property': {
                'patterns': ['steal', 'robbery', 'burglary', 'loot', 'take (without )?permission'],
                'domain': 'theft',
                'boost_keywords': ['theft', 'property'],
                'block_keywords': ['homicide', 'violence', 'assault'],
                'boost_chapters': ['Chapter-20 (Offences Relating to Theft and Robbery)'],
                'exclude_chapters': ['Chapter-19 (Offences Relating to Human Life)'],
                'chapter_boost_multiplier': 1.5
            },
            
            # MARRIAGE/FAMILY SCENARIOS
            'marry person': {
                'patterns': ['marry', 'marriage', 'wedding', 'husband', 'wife', 'spouse'],
                'domain': 'marriage',
                'boost_keywords': ['marriage'],
                'block_keywords': ['crime', 'offense', 'theft'],
                'boost_chapters': ['Chapter-1 (Provisions Relating to Marriage)', 'Chapter-2 (Provisions Relating to Consequences of Marriage)'],
                'exclude_chapters': ['Chapter-20 (Offences Relating to Theft and Robbery)'],
                'chapter_boost_multiplier': 1.4
            },
            'divorce person': {
                'patterns': ['divorce', 'separate', 'marital dissolved', 'end marriage'],
                'domain': 'divorce',
                'boost_keywords': ['divorce'],
                'block_keywords': ['crime', 'theft'],
                'boost_chapters': ['Chapter-3 (Provisions Relating to Divorce)'],
                'exclude_chapters': ['Chapter-20 (Offences Relating to Theft and Robbery)'],
                'chapter_boost_multiplier': 1.5
            },
            
            # INHERITANCE/SUCCESSION
            'inherit property': {
                'patterns': ['inherit', 'inheritance', 'succession', 'will', 'estate', 'bequeath', 'legacy'],
                'domain': 'succession',
                'boost_keywords': ['inheritance', 'succession'],
                'block_keywords': ['crime', 'theft', 'homicide'],
                'boost_chapters': ['Chapter-11 (Provisions Relating to Succession)', 'Chapter-12 (Provisions Relating to Will)'],
                'exclude_chapters': ['Chapter-20 (Offences Relating to Theft and Robbery)'],
                'chapter_boost_multiplier': 1.4
            },
            
            # COMMERCIAL/CONTRACT
            'sell goods': {
                'patterns': ['sell', 'sale', 'purchase', 'buy', 'merchant', 'goods', 'commercial'],
                'domain': 'contract',
                'boost_keywords': ['contract', 'agreement', 'sale'],
                'block_keywords': ['crime', 'homicide', 'theft'],
                'boost_chapters': ['Chapter-6 (Provisions Relating to Contract of Sale of Goods)'],
                'exclude_chapters': ['Chapter-20 (Offences Relating to Theft and Robbery)'],
                'chapter_boost_multiplier': 1.4
            },
        }
        
        # PHASE 3: ENHANCED SINGLE KEYWORD BOOSTING
        # Maps query keywords to clause IDs that should be boosted in results
        # Now includes CONTEXT-SENSITIVE boosting (can be overridden by phrase matching)
        self.query_clause_boosts = {
            # CONSTITUTIONAL QUERIES (Nepal, Nation, State)
            'nepal': {
                'boost_clause_ids': ['2', '3', '4'],
                'boost_multiplier': 1.8,
                'boost_source_filter': 'Constitution of Nepal, 2015',
                'require_context': None  # Always apply
            },
            'citizenship': {
                'boost_clause_ids': ['11', '35'],  # Constitution citizenship clauses
                'boost_multiplier': 1.6,
                'boost_source_filter': 'Constitution of Nepal, 2015',
                'require_context': 'person'  # Only apply when discussing persons
            },
            'nation': {
                'boost_clause_ids': ['3', '4', '2'],
                'boost_multiplier': 1.8,
                'boost_source_filter': 'Constitution of Nepal, 2015',
                'require_context': None
            },
            'fundamental': {
                'boost_clause_ids': ['1', '2', '3'],  # Fundamental rights/principles
                'boost_multiplier': 1.5,
                'boost_source_filter': None,
                'require_context': None
            },
            
            # MARRIAGE & FAMILY QUERIES
            'marriage': {
                'boost_clause_ids': ['1', '2', '3', '4', '5'],  # Civil Code marriage provisions
                'boost_multiplier': 1.7,
                'boost_source_filter': 'National Civil Code, 2017 AD',
                'require_context': 'person'
            },
            'divorce': {
                'boost_clause_ids': ['3', '4', '5', '6'],  # Civil Code divorce provisions
                'boost_multiplier': 1.7,
                'boost_source_filter': 'National Civil Code, 2017 AD',
                'require_context': 'person'
            },
            'adoption': {
                'boost_clause_ids': ['8', '9'],  # Civil Code adoption provisions
                'boost_multiplier': 1.5,
                'boost_source_filter': 'National Civil Code, 2017 AD',
                'require_context': 'person'
            },
            'guardianship': {
                'boost_clause_ids': ['6', '7'],  # Guardianship clauses
                'boost_multiplier': 1.5,
                'boost_source_filter': 'National Civil Code, 2017 AD',
                'require_context': 'person'
            },
            
            # PROPERTY & OWNERSHIP QUERIES
            'property': {
                'boost_clause_ids': ['1', '11', '12', '13'],
                'boost_multiplier': 1.6,
                'boost_source_filter': 'National Civil Code, 2017 AD',
                'require_context': None
            },
            'ownership': {
                'boost_clause_ids': ['2', '11'],
                'boost_multiplier': 1.5,
                'boost_source_filter': 'National Civil Code, 2017 AD',
                'require_context': None
            },
            'land': {
                'boost_clause_ids': ['4', '11', '13'],
                'boost_multiplier': 1.6,
                'boost_source_filter': 'National Civil Code, 2017 AD',
                'require_context': 'property'
            },
            'inheritance': {
                'boost_clause_ids': ['11', '15', '16', '17'],
                'boost_multiplier': 1.7,
                'boost_source_filter': 'National Civil Code, 2017 AD',
                'require_context': 'property'
            },
            'succession': {
                'boost_clause_ids': ['11', '15', '16', '17'],
                'boost_multiplier': 1.7,
                'boost_source_filter': 'National Civil Code, 2017 AD',
                'require_context': 'property'
            },
            
            # CONTRACT & OBLIGATION QUERIES
            'contract': {
                'boost_clause_ids': ['2', '3', '4', '5', '6', '7'],
                'boost_multiplier': 1.6,
                'boost_source_filter': 'National Civil Code, 2017 AD',
                'require_context': None
            },
            'agreement': {
                'boost_clause_ids': ['2', '6'],
                'boost_multiplier': 1.5,
                'boost_source_filter': 'National Civil Code, 2017 AD',
                'require_context': None
            },
            'liability': {
                'boost_clause_ids': ['8', '17'],
                'boost_multiplier': 1.5,
                'boost_source_filter': 'National Civil Code, 2017 AD',
                'require_context': 'property'  # Property damage liability
            },
            
            # CRIMINAL & OFFENSE QUERIES - NOW MORE SPECIFIC
            'crime': {
                'boost_clause_ids': ['2', '3'],
                'boost_multiplier': 1.6,
                'boost_source_filter': 'National Criminal Code, 2017 AD',
                'require_context': 'person'  # Only criminal acts against persons
            },
            'homicide': {
                'boost_clause_ids': ['15', '16'],  # Specific homicide clauses
                'boost_multiplier': 1.7,
                'boost_source_filter': 'National Criminal Code, 2017 AD',
                'require_context': 'person',
                'block_if_context': 'animal'  # Don't boost if talking about animals
            },
            'murder': {
                'boost_clause_ids': ['15'],  # Murder clause
                'boost_multiplier': 1.8,
                'boost_source_filter': 'National Criminal Code, 2017 AD',
                'require_context': 'person',
                'block_if_context': 'animal'
            },
            'offense': {
                'boost_clause_ids': ['1', '2'],
                'boost_multiplier': 1.6,
                'boost_source_filter': 'National Criminal Code, 2017 AD',
                'require_context': 'person'
            },
            'punishment': {
                'boost_clause_ids': ['4', '5'],
                'boost_multiplier': 1.5,
                'boost_source_filter': 'National Criminal Code, 2017 AD',
                'require_context': 'person'
            },
            'theft': {
                'boost_clause_ids': ['20'],  # Chapter 20: Theft and Robbery
                'boost_multiplier': 1.5,
                'boost_source_filter': 'National Criminal Code, 2017 AD',
                'require_context': 'property'  # Theft is about property
            },
            'rape': {
                'boost_clause_ids': ['18'],  # Chapter 18: Sexual Offences
                'boost_multiplier': 1.5,
                'boost_source_filter': 'National Criminal Code, 2017 AD',
                'require_context': 'person'
            },
            
            # NEW: ANIMAL & INJURY RELATED (NON-CRIMINAL)
            'animal': {
                'boost_clause_ids': ['17', '18'],  # Tort/liability chapters
                'boost_multiplier': 1.4,
                'boost_source_filter': 'National Civil Code, 2017 AD',
                'require_context': 'animal'
            },
            'veterinary': {
                'boost_clause_ids': ['17'],  # Liability for damage
                'boost_multiplier': 1.3,
                'boost_source_filter': 'National Civil Code, 2017 AD',
                'require_context': 'animal'
            },
            'kill': {
                'boost_clause_ids': [],  # NO auto-boost for "kill" - context-dependent
                'boost_multiplier': 1.0,
                'boost_source_filter': None,
                'require_context': None,
                'needs_phrase_disambiguation': True  # Requires phrase matching
            },
        }
        
        # PHASE 4: CHAPTER-LEVEL FILTERING & BOOSTING (ENHANCED with context awareness)
        self.query_chapter_config = {
            # CITIZENSHIP QUERIES
            'citizenship': {
                'exclude_chapters': [
                    'Chapter-1 (Provisions Relating to Marriage)',
                    'Chapter-2 (Provisions Relating to Consequences of Marriage)',
                    'Chapter-3 (Provisions Relating to Divorce)',
                    'Chapter-8 (Provisions Relating to Adoption)',
                    'Chapter-11 (Offences Relating to Marriage)',
                ],
                'boost_chapters': [
                    'Chapter-1 (General Provisions)',
                ],
                'chapter_boost_multiplier': 1.3,
                'context_dependent': 'person'
            },
            
            # MARRIAGE QUERIES
            'marriage': {
                'exclude_chapters': [
                    'Chapter-2 (Offences against Public Tranquility)',
                    'Chapter-20 (Offences Relating to Theft and Robbery)',
                ],
                'boost_chapters': [
                    'Chapter-1 (Provisions Relating to Marriage)',
                    'Chapter-2 (Provisions Relating to Consequences of Marriage)',
                    'Chapter-11 (Offences Relating to Marriage)',
                ],
                'chapter_boost_multiplier': 1.4,
                'context_dependent': 'person'
            },
            
            # DIVORCE QUERIES
            'divorce': {
                'exclude_chapters': [
                    'Chapter-1 (Provisions Relating to Marriage)',
                    'Chapter-20 (Offences Relating to Theft and Robbery)',
                ],
                'boost_chapters': [
                    'Chapter-3 (Provisions Relating to Divorce)',
                    'Chapter-11 (Offences Relating to Marriage)',
                ],
                'chapter_boost_multiplier': 1.5,
                'context_dependent': 'person'
            },
            
            # PROPERTY QUERIES
            'property': {
                'exclude_chapters': [
                    'Chapter-1 (Provisions Relating to Marriage)',
                    'Chapter-2 (Provisions Relating to Consequences of Marriage)',
                    'Chapter-3 (Provisions Relating to Divorce)',
                    'Chapter-12 (Offences Relating to Human Body)',  # Exclude homicide
                ],
                'boost_chapters': [
                    'Chapter-1 (General Provisions Relating to Property)',
                    'Chapter-2 (Provisions Relating to Ownership and Possession)',
                    'Chapter-3 (Provisions Relating to Uses of Property)',
                    'Chapter-4 (Provisions Relating to Cultivation, Use and Registration of Land)',
                    'Chapter-11 (Provisions Relating to Transfer and Acquisition of Property)',
                    'Chapter-12 (Provisions Relating to Mortgage of Immovable Property)',
                    'Chapter-13 (Provisions Relating to Pre-emption of Immovable Property)',
                ],
                'chapter_boost_multiplier': 1.4,
                'context_dependent': 'property'
            },
            
            # OWNERSHIP QUERIES
            'ownership': {
                'exclude_chapters': [
                    'Chapter-1 (Provisions Relating to Marriage)',
                    'Chapter-20 (Offences Relating to Theft and Robbery)',
                    'Chapter-12 (Offences Relating to Human Body)',
                ],
                'boost_chapters': [
                    'Chapter-2 (Provisions Relating to Ownership and Possession)',
                    'Chapter-11 (Provisions Relating to Transfer and Acquisition of Property)',
                ],
                'chapter_boost_multiplier': 1.4,
                'context_dependent': 'property'
            },
            
            # LAND QUERIES
            'land': {
                'exclude_chapters': [
                    'Chapter-1 (Provisions Relating to Marriage)',
                    'Chapter-20 (Offences Relating to Theft and Robbery)',
                    'Chapter-12 (Offences Relating to Human Body)',
                ],
                'boost_chapters': [
                    'Chapter-4 (Provisions Relating to Cultivation, Use and Registration of Land)',
                    'Chapter-11 (Provisions Relating to Transfer and Acquisition of Property)',
                    'Chapter-12 (Provisions Relating to Mortgage of Immovable Property)',
                    'Chapter-13 (Provisions Relating to Pre-emption of Immovable Property)',
                ],
                'chapter_boost_multiplier': 1.4,
                'context_dependent': 'property'
            },
            
            # INHERITANCE & SUCCESSION QUERIES
            'inheritance': {
                'exclude_chapters': [
                    'Chapter-1 (Provisions Relating to Marriage)',
                    'Chapter-20 (Offences Relating to Theft and Robbery)',
                    'Chapter-19 (Offences Relating to Human Life)',
                ],
                'boost_chapters': [
                    'Chapter-11 (Provisions Relating to Succession)',
                ],
                'chapter_boost_multiplier': 1.5,
                'context_dependent': 'property'
            },
            
            'succession': {
                'exclude_chapters': [
                    'Chapter-1 (Provisions Relating to Marriage)',
                    'Chapter-20 (Offences Relating to Theft and Robbery)',
                    'Chapter-19 (Offences Relating to Human Life)',
                ],
                'boost_chapters': [
                    'Chapter-11 (Provisions Relating to Succession)',
                ],
                'chapter_boost_multiplier': 1.5,
                'context_dependent': 'property'
            },
            
            # CONTRACT QUERIES
            'contract': {
                'exclude_chapters': [
                    'Chapter-1 (Provisions Relating to Marriage)',
                    'Chapter-3 (Provisions Relating to Divorce)',
                    'Chapter-8 (Provisions Relating to Adoption)',
                    'Chapter-20 (Offences Relating to Theft and Robbery)',
                ],
                'boost_chapters': [
                    'Chapter-2 (Provisions Relating to Formation of Contracts)',
                    'Chapter-3 (Validity of Contracts)',
                    'Chapter-4 (Provisions Relating to Performance of Contracts)',
                    'Chapter-5 (Provisions Relating to Breach of Contract and Remedies)',
                    'Chapter-6 (Provisions Relating to Contract of Sale of Goods)',
                    'Chapter-7 (Provisions Relating to Contracts of Guarantee)',
                    'Chapter-8 (Provisions Relating to Contracts of Bailment)',
                ],
                'chapter_boost_multiplier': 1.4,
                'context_dependent': None
            },
            
            # ADOPTION QUERIES
            'adoption': {
                'exclude_chapters': [
                    'Chapter-1 (Provisions Relating to Marriage)',
                    'Chapter-2 (Offences against Public Tranquility)',
                    'Chapter-20 (Offences Relating to Theft and Robbery)',
                    'Chapter-19 (Offences Relating to Human Life)',
                ],
                'boost_chapters': [
                    'Chapter-8 (Provisions Relating to Adoption)',
                    'Chapter-9 (Provisions Relating to Inter-Country Adoption)',
                ],
                'chapter_boost_multiplier': 1.4,
                'context_dependent': 'person'
            },
            
            # GUARDIANSHIP QUERIES
            'guardianship': {
                'exclude_chapters': [
                    'Chapter-1 (Provisions Relating to Marriage)',
                    'Chapter-20 (Offences Relating to Theft and Robbery)',
                    'Chapter-19 (Offences Relating to Human Life)',
                ],
                'boost_chapters': [
                    'Chapter-6 (Provisions Relating to Guardianship)',
                    'Chapter-7 (Provisions Relating to Curatorship)',
                ],
                'chapter_boost_multiplier': 1.4,
                'context_dependent': 'person'
            },
            
            # CRIMINAL QUERIES - NOW REQUIRE PERSON CONTEXT
            'crime': {
                'exclude_chapters': [
                    'Chapter-1 (Provisions Relating to Marriage)',
                    'Chapter-2 (Provisions Relating to Formation of Contracts)',
                    'Chapter-3 (Provisions Relating to Divorce)',
                ],
                'boost_chapters': [
                    'Chapter-2 (Offences against Public Tranquility)',
                    'Chapter-3 (Offences Relating to Contempt of Authority)',
                ],
                'chapter_boost_multiplier': 1.4,
                'context_dependent': 'person'
            },
            
            'offense': {
                'exclude_chapters': [
                    'Chapter-1 (Provisions Relating to Marriage)',
                    'Chapter-2 (Provisions Relating to Formation of Contracts)',
                ],
                'boost_chapters': [
                    'Chapter-1 (Offences against the State)',
                    'Chapter-2 (Offences against Public Tranquility)',
                    'Chapter-3 (Offences Relating to Contempt of Authority)',
                    'Chapter-4 (Gravity of Offence Aggravating and Mitigating Factors)',
                ],
                'chapter_boost_multiplier': 1.4,
                'context_dependent': 'person'
            },
            
            'theft': {
                'exclude_chapters': [
                    'Chapter-1 (Provisions Relating to Marriage)',
                    'Chapter-2 (Provisions Relating to Formation of Contracts)',
                    'Chapter-19 (Offences Relating to Human Life)',  # Exclude homicide
                ],
                'boost_chapters': [
                    'Chapter-20 (Offences Relating to Theft and Robbery)',
                ],
                'chapter_boost_multiplier': 1.5,
                'context_dependent': 'property'
            },
            
            'rape': {
                'exclude_chapters': [
                    'Chapter-1 (Provisions Relating to Marriage)',
                    'Chapter-20 (Offences Relating to Theft and Robbery)',
                ],
                'boost_chapters': [
                    'Chapter-18 (Sexual Offences)',
                ],
                'chapter_boost_multiplier': 1.5,
                'context_dependent': 'person'
            },
            
            'punishment': {
                'exclude_chapters': [
                    'Chapter-1 (Provisions Relating to Marriage)',
                ],
                'boost_chapters': [
                    'Chapter-4 (Gravity of Offence Aggravating and Mitigating Factors)',
                    'Chapter-5 (Provisions Relating to Punishment and Interim Relief)',
                ],
                'chapter_boost_multiplier': 1.4,
                'context_dependent': 'person'
            },
            
            # GENERAL LIABILITY & TORTS (for non-criminal injuries/damage)
            'liability': {
                'exclude_chapters': [
                    'Chapter-1 (Provisions Relating to Marriage)',
                    'Chapter-20 (Offences Relating to Theft and Robbery)',
                ],
                'boost_chapters': [
                    'Chapter-17 (Provisions Relating to Torts)',
                    'Chapter-18 (Provisions Relating to Liability for Defective Products)',
                ],
                'chapter_boost_multiplier': 1.5,
                'context_dependent': None
            },
            
            'tort': {
                'exclude_chapters': [
                    'Chapter-1 (Provisions Relating to Marriage)',
                    'Chapter-20 (Offences Relating to Theft and Robbery)',
                ],
                'boost_chapters': [
                    'Chapter-17 (Provisions Relating to Torts)',
                ],
                'chapter_boost_multiplier': 1.4,
                'context_dependent': None
            },
            
            # ANIMAL-SPECIFIC HANDLING
            'animal': {
                'exclude_chapters': [
                    'Chapter-12 (Offences Relating to Human Body)',  # Exclude homicide
                    'Chapter-1 (Offences against the State)',
                    'Chapter-5 (Provisions Relating to Punishment)'
                ],
                'boost_chapters': [
                    'Chapter-27 (Offences Relating to Animals and Birds)'  # Animal torts
                    
                ],
                'chapter_boost_multiplier': 1.9,
                'context_dependent': 'animal'
            },
        }
        
        
        self.semantic_filter = SemanticDomainFilter() if enable_semantic_filter else None
        
        # Initialize reranker
        if use_reranker:
            self.reranker = create_reranker(
                reranker_type=reranker_type,
                model_name=reranker_model,
                rerank_weight=rerank_weight,
                initial_weight=initial_weight
            )
        else:
            self.reranker = None
        
        # Build indexes
        self._build_indexes()
    
    def _build_indexes(self):
        """Build embeddings and BM25 index"""
        if self.verbose:
            print("\nBuilding indexes...")
        
        self.child_texts = [doc['text'] for doc in self.child_docs]
        
        if self.verbose:
            print("  Building BM25 index...")
        self.bm25.fit(self.child_texts)
        
        if self.verbose:
            print("  Building semantic embeddings...")
        self.child_embeddings = self.embedding_model.encode(
            self.child_texts,
            show_progress=self.verbose
        )
        
        if self.verbose:
            print("  Building parent document lookup...")
        self.parent_lookup = {}
        for parent_doc in self.parent_docs:
            key = (parent_doc['clause_id'], parent_doc['legal_document_source'])
            self.parent_lookup[key] = parent_doc
        
        if self.verbose:
            print(f"\n[OK] Indexing complete!")
            print(f"  - {len(self.child_docs)} child documents indexed")
            print(f"  - {len(self.parent_lookup)} parent documents indexed")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        return_parent_text: bool = True,
        deduplicate_parents: bool = True,
        filter_document_source: Optional[str] = None,
        enable_semantic_filter: Optional[bool] = None,
        normalize_query: bool = True
    ) -> List[RetrievalResult]:
        """
        Perform hybrid search with reranking, semantic filtering, and deduplication
        
        Pipeline:
        1. Normalize query (extract intent, domain, key terms)
        2. Extract clause IDs and document type from query
        3. Compute similarity and BM25 scores
        4. Apply direct hit bonuses
        5. Get top-N candidates for reranking
        6. Rerank using cross-encoder
        7. Apply semantic domain filtering (prevents domain mismatches)
        8. Deduplicate by parent clause (returns only best child per parent)
        9. Return top-K final results
        
        Args:
            query: Search query
            top_k: Number of final results to return
            return_parent_text: Whether to retrieve parent clause text
            deduplicate_parents: If True, only one best child per parent is returned
            filter_document_source: Optional filter for document source
            enable_semantic_filter: Override default semantic filter setting
            normalize_query: If True, normalize query for optimal retrieval (default: True)
            
        Returns:
            List of RetrievalResult objects with deduplicated parents
        """
        # Use provided filter setting or default
        use_filter = enable_semantic_filter if enable_semantic_filter is not None else self.enable_semantic_filter
        
        # Step 0: Normalize query for optimal retrieval
        normalized_query_info = None
        search_query = query
        
        if normalize_query:
            norm_result = self.query_normalizer.normalize(query)
            normalized_query_info = norm_result
            search_query = self.query_normalizer.to_search_string(norm_result)
            
            if self.verbose:
                print(f"\n[Query Normalization]")
                print(f"  Original: {query}")
                print(f"  Intent: {norm_result.intent}")
                print(f"  Domain: {norm_result.domain}")
                if norm_result.suggested_chapters:
                    print(f"  Suggested chapters: {', '.join(norm_result.suggested_chapters[:3])}")
                print(f"  Confidence: {norm_result.confidence:.1%}")
        
        # Step 1: Parse query
        query_info = self.clause_extractor.parse_query(search_query)
        direct_hit_clause_ids = query_info['clause_ids']
        doc_type = query_info['document_type']
        
        if self.verbose and direct_hit_clause_ids:
            print(f"\n[Query Analysis]")
            print(f"  Clause IDs: {direct_hit_clause_ids}")
            print(f"  Document type: {doc_type}")
        
        # Step 2: Compute initial retrieval scores
        similarity_scores = self._compute_similarity_scores(query)
        similarity_norm = self._normalize_scores(similarity_scores)
        
        # Expand query for better keyword matching when no clause IDs found
        expanded_query = self._expand_query(query, direct_hit_clause_ids)
        
        bm25_scores = self.bm25.get_scores(expanded_query)
        bm25_norm = self._normalize_scores(bm25_scores)
        
        # Step 3: Create search results
        search_results = []
        for idx, child_doc in enumerate(self.child_docs):
            if filter_document_source and child_doc['legal_document_source'] != filter_document_source:
                continue
            
            # Skip documents excluded by chapter filtering
            if self._should_exclude_by_chapter(query, child_doc):
                continue
            
            result = self._create_search_result(
                child_doc, idx, similarity_norm, bm25_norm,
                direct_hit_clause_ids, doc_type, query
            )
            search_results.append(result)
        
        # Step 4: Sort and get candidates
        search_results.sort(key=lambda x: x.initial_score, reverse=True)
        rerank_candidates_count = min(self.rerank_top_k, len(search_results))
        candidates = search_results[:rerank_candidates_count]
        
        # Step 5: Rerank if enabled
        if self.use_reranker and self.reranker and len(candidates) > 1:
            candidates = self._rerank_results(query, candidates)
        else:
            for result in candidates:
                result.final_score = result.initial_score
        
        # Combine with remaining
        remaining = search_results[rerank_candidates_count:]
        for result in remaining:
            result.final_score = result.initial_score
        
        all_results = candidates + remaining
        all_results.sort(key=lambda x: x.final_score, reverse=True)
        
        # Step 6: Apply semantic domain filtering
        if use_filter and self.semantic_filter:
            all_results = self._apply_semantic_filter(query, all_results)
        
        # Step 7: Filter exact matches if clauses were requested
        if direct_hit_clause_ids:
            all_results = self._filter_exact_matches(
                all_results, direct_hit_clause_ids, doc_type
            )
        
        # Step 8: Deduplicate by parent clause
        if deduplicate_parents:
            top_results = self._deduplicate_by_parent(all_results, top_k)
        else:
            top_results = all_results[:top_k]
        
        # Step 9: Retrieve parent texts
        if return_parent_text:
            return self._create_retrieval_results(top_results, doc_type)
        else:
            return self._convert_to_retrieval_results(top_results)
    
    def _compute_similarity_scores(self, query: str) -> np.ndarray:
        """Compute cosine similarity scores"""
        query_embedding = self.embedding_model.encode([query])[0]
        similarities = np.dot(self.child_embeddings, query_embedding)
        return similarities
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range"""
        if len(scores) == 0:
            return scores
        
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score == min_score:
            return np.zeros_like(scores)
        
        return (scores - min_score) / (max_score - min_score)
    
    def _expand_query(self, query: str, direct_hit_clause_ids: List[str]) -> str:
        """
        Expand query with related keywords for better natural language search
        Only applies when no direct clause IDs are found
        """
        if direct_hit_clause_ids:
            return query  # Don't expand if specific clauses are requested
        
        query_lower = query.lower()
        expanded_terms = []
        
        # Look for keywords in the query and expand them
        for base_term, expansions in self.keyword_expansions.items():
            if base_term in query_lower:
                # Add all expansions except the base term itself to avoid duplication
                for expansion in expansions:
                    if expansion != base_term:
                        expanded_terms.append(expansion)
        
        # Return expanded query with additional keywords
        if expanded_terms:
            return f"{query} {' '.join(expanded_terms)}"
        return query
    
    def _create_search_result(
        self,
        child_doc: Dict,
        idx: int,
        similarity_scores: np.ndarray,
        bm25_scores: np.ndarray,
        direct_hit_clause_ids: List[str],
        doc_type: Optional[str],
        query: str = ""
    ) -> SearchResult:
        """Create search result with score calculation"""
        # Use 'or ""' and 'str()' to ensure these are never NoneType
        clean_clause_id = str(child_doc.get('clause_id') or "")
        clean_parent_id = str(child_doc.get('parent_clause_id') or "")
        clean_text = child_doc.get('text') or ""
        clean_source = child_doc.get('legal_document_source') or ""
        clean_chapter = child_doc.get('chapter') or ""
        clean_part = child_doc.get('part') or ""

        result = SearchResult(
            clause_id=clean_clause_id,
            text=clean_text,
            part=clean_part,
            chapter=clean_chapter,
            parent_clause_id=clean_parent_id,
            parent_text=child_doc.get('parent_text') or "",
            legal_document_source=clean_source,
            similarity_score=similarity_scores[idx],
            bm25_score=bm25_scores[idx]
        )
        
        # Check for direct hit
        if child_doc['clause_id'] in direct_hit_clause_ids:
            result.is_direct_hit = True
            result.direct_hit_bonus = self._calculate_direct_hit_bonus(
                child_doc, doc_type
            )
        
        # Calculate initial score
        base_score = (
            self.similarity_weight * result.similarity_score +
            self.bm25_weight * result.bm25_score
        )
        
        if result.is_direct_hit:
            result.initial_score = base_score * (1 + result.direct_hit_bonus)
        else:
            result.initial_score = base_score
        
        # Apply query-specific clause boosts for better constitutional question handling
        clause_boost = self._get_clause_boost_multiplier(query, child_doc)
        result.initial_score *= clause_boost
        
        # Apply chapter-specific boosts to prioritize relevant chapters
        chapter_boost = self._get_chapter_boost_multiplier(query, child_doc)
        result.initial_score *= chapter_boost
        
        # Apply document keyword matching boosts (routes to relevant documents)
        document_boost = self._get_document_boost_multiplier(query, child_doc)
        result.initial_score *= document_boost
        
        phrase_match = self._check_phrase_match(query)
        if phrase_match == 'kill animal':
            # If we detected killing an animal, check if this doc is about humans
            doc_text_lower = child_doc['text'].lower()
            if 'person' in doc_text_lower or 'human' in doc_text_lower or 'homicide' in doc_text_lower:
                result.initial_score *= 0.1
        
        entity_context = self._detect_entity_context(query)
        if entity_context == 'animal':
            text_lower = child_doc['text'].lower()
            # If the text mentions humans or state crimes while we are looking for animal laws
            if 'genocide' in text_lower or 'human group' in text_lower or 'against the state' in text_lower:
                result.initial_score *= 0.0 # 100% penalty to move it to the bottom
        
        
        return result
    
    def _calculate_direct_hit_bonus(self, child_doc: Dict, doc_type: Optional[str]) -> float:
        """Calculate bonus multiplier for direct hits"""
        if not doc_type:
            return self.direct_hit_bonus
        
        doc_source_lower = child_doc['legal_document_source'].lower()
        
        type_matches = {
            'civil': 'civil' in doc_source_lower,
            'criminal': 'criminal' in doc_source_lower,
            'constitution': 'constitution' in doc_source_lower
        }
        
        return self.direct_hit_bonus if type_matches.get(doc_type, False) else self.direct_hit_bonus * 0.5
    
    def _detect_target_documents(self, query: str) -> Dict[str, float]:
        """
        Detect which documents are relevant based on query keywords.
        Returns a dictionary mapping document names to relevance scores (0-1).
        
        Uses the comprehensive document_keyword_mapping to match query keywords
        with relevant legal documents.
        """
        query_lower = query.lower()
        document_scores = {}
        
        # Calculate relevance score for each document
        for doc_name, doc_config in self.document_keyword_mapping.items():
            keywords = doc_config['keywords']
            weight = doc_config['weight']
            
            # Count how many keywords from this document appear in the query
            matched_keywords = 0
            for keyword in keywords:
                if keyword in query_lower:
                    matched_keywords += 1
            
            # Calculate relevance score (0-1)
            if matched_keywords > 0:
                # Normalize by total keywords to avoid bias toward documents with more keywords
                relevance = min(1.0, matched_keywords / 5.0)  # Max at 5 keyword matches
                document_scores[doc_name] = relevance * weight
            else:
                document_scores[doc_name] = 0.0
        
        return document_scores
    
    def _get_document_boost_multiplier(self, query: str, child_doc: Dict) -> float:
        """
        Get boost multiplier for results from documents that match query keywords.
        
        Returns:
            Multiplier > 1.0 if document is relevant to query
            Multiplier ~= 1.0 if no strong match
            Multiplier could be < 1.0 if document seems irrelevant (but not penalized heavily)
        """
        doc_source = child_doc.get('legal_document_source') or ""
        
        # Get document relevance scores
        doc_scores = self._detect_target_documents(query)
        
        # Find the relevance score for this document's source
        for doc_name, score in doc_scores.items():
            if doc_name in doc_source:
                # Convert relevance score to boost multiplier
                # 0.0 relevance → 1.0x multiplier (no boost)
                # 1.0 relevance → 1.6x multiplier (strong boost)
                return 1.0 + (score * 0.6)  # Range: 1.0 to 1.6
        
        return 1.0  # No boost if document not explicitly matched
    
    def _should_exclude_by_document_type(self, query: str, child_doc: Dict) -> bool:
        """
        Check if a document should be excluded based on document type mismatch.
        Only excludes if the query has very strong document-specific keywords
        and the document doesn't match.
        
        For example, if query is about "criminal offenses" (100% criminal),
        don't exclude documents from Criminal Code.
        
        But if query mentions multiple document types, be lenient.
        """
        query_lower = query.lower()
        doc_source = child_doc.get('legal_document_source') or ""
        
        # Get document relevance scores
        doc_scores = self._detect_target_documents(query)
        
        # Check if any document has a very high relevance score (strong direction)
        max_relevance = max(doc_scores.values()) if doc_scores else 0.0
        
        # Only exclude if: (1) very strong match to another document AND (2) this doc has 0 match
        if max_relevance > 0.8:
            # Check if this document has any relevance
            for doc_name, score in doc_scores.items():
                if doc_name in doc_source:
                    return False if score > 0.0 else True
        
        # Default: don't exclude (be inclusive)
        return False
    
    def _detect_entity_context(self, query: str) -> Optional[str]:
        """
        Detect what entity is being discussed in the query.
        Returns: 'animal', 'person', 'property', 'business', or None
        """
        query_lower = query.lower()
        
        for context_type, context_data in self.entity_context_keywords.items():
            keywords = context_data['keywords']
            for keyword in keywords:
                if keyword in query_lower:
                    return context_data['context']
        
        return None
    
    def _check_phrase_match(self, query: str) -> Optional[str]:
        """
        Check if query matches any multi-word phrase patterns.
        Returns: matching phrase key or None
        """
        import re
        query_lower = query.lower()
        
        for phrase_key, phrase_config in self.phrase_matching_rules.items():
            patterns = phrase_config.get('patterns', [])
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    return phrase_key
        
        return None
    
    def _get_clause_boost_multiplier(self, query: str, child_doc: Dict) -> float:
        """
        Get boost multiplier for clauses with CONTEXT-AWARE boosting.
        Handles entity detection, phrase matching, and disambiguation.
        Returns 1.0 (no boost) if no specific boost applies.
        """
        if not query:
            return 1.0
        
        query_lower = query.lower()
        clause_id = str(child_doc.get('clause_id', '')).strip()
        doc_source = child_doc.get('legal_document_source') or ""
        
        # STEP 1: Detect entity context in query
        entity_context = self._detect_entity_context(query)
        
        # STEP 2: Check for phrase-based matching (highest priority)
        phrase_match = self._check_phrase_match(query)
        if phrase_match:
            phrase_config = self.phrase_matching_rules[phrase_match]
            # Don't boost blocked keywords
            block_keywords = phrase_config.get('block_keywords', [])
            for block_kw in block_keywords:
                if block_kw in query_lower:
                    # Still check if specific clauses match
                    for keyword, boost_config in self.query_clause_boosts.items():
                        if keyword in query_lower and clause_id in boost_config.get('boost_clause_ids', []):
                            # But apply reduced multiplier for blocked context
                            return boost_config['boost_multiplier'] * 0.5
                    return 1.0
        
        # STEP 3: Check standard clause boost rules with context filtering
        for keyword, boost_config in self.query_clause_boosts.items():
            if keyword in query_lower:
                # Check if this boost requires a specific context
                required_context = boost_config.get('require_context')
                if required_context and entity_context != required_context:
                    # Skip this boost - wrong entity context
                    continue
                
                # Check if this boost should be blocked by context
                block_if_context = boost_config.get('block_if_context')
                if block_if_context and entity_context == block_if_context:
                    # Don't boost - context blocks this keyword
                    continue
                
                # Check if this keyword needs phrase disambiguation
                if boost_config.get('needs_phrase_disambiguation'):
                    # Only boost if we have a matching phrase
                    if not phrase_match:
                        continue
                
                # Check if this clause should be boosted
                if clause_id in boost_config.get('boost_clause_ids', []):
                    # Check if document source matches
                    if boost_config.get('boost_source_filter'):
                        if boost_config['boost_source_filter'] in doc_source:
                            return boost_config['boost_multiplier']
                    else:
                        return boost_config['boost_multiplier']
        
        return 1.0
    
    def _should_exclude_by_chapter(self, query: str, child_doc: Dict) -> bool:
        """
        Check if a document should be excluded based on its chapter.
        Now includes CONTEXT-AWARE filtering.
        Returns True if the document should be filtered out.
        """
        query_lower = query.lower()
        chapter = child_doc.get('chapter')
        chapter = str(chapter) if chapter is not None else "" # Handle None case
        
        if not chapter:
            return False  # Don't exclude documents without chapters
        
        # STEP 1: Check phrase-based exclusions (highest priority)
        phrase_match = self._check_phrase_match(query)
        if phrase_match:
            phrase_config = self.phrase_matching_rules[phrase_match]
            excluded_chapters = phrase_config.get('exclude_chapters', [])
            for excluded_ch in excluded_chapters:
                if excluded_ch.lower() in chapter.lower():
                    return True
        
        # STEP 2: Detect entity context
        entity_context = self._detect_entity_context(query)
        
        # STEP 3: Check standard chapter-level exclusions with context
        for keyword, config in self.query_chapter_config.items():
            if keyword in query_lower:
                # Check if this exclusion requires context
                context_dependent = config.get('context_dependent')
                if context_dependent and entity_context != context_dependent:
                    # Skip this exclusion - wrong context
                    continue
                
                excluded_chapters = config.get('exclude_chapters', [])
                for excluded_ch in excluded_chapters:
                    if excluded_ch.lower() in chapter.lower():
                        return True
        
        return False
    
    def _get_chapter_boost_multiplier(self, query: str, child_doc: Dict) -> float:
        """
        Get boost multiplier for clauses based on chapter relevance.
        Now includes CONTEXT-AWARE boosting.
        Returns 1.0 (no boost) if no specific boost applies.
        """
        query_lower = query.lower()
        chapter = child_doc.get('chapter') or ""  # Handle None case
        
        if not chapter:
            return 1.0  # No boost for documents without chapters
        
        # STEP 1: Check phrase-based chapter boosts (highest priority)
        phrase_match = self._check_phrase_match(query)
        if phrase_match:
            phrase_config = self.phrase_matching_rules[phrase_match]
            boost_chapters = phrase_config.get('boost_chapters', [])
            multiplier = phrase_config.get('chapter_boost_multiplier', 1.0)
            
            for boost_ch in boost_chapters:
                if boost_ch.lower() in chapter.lower():
                    return multiplier
        
        # STEP 2: Detect entity context
        entity_context = self._detect_entity_context(query)
        
        # STEP 3: Check standard chapter boosts with context
        for keyword, config in self.query_chapter_config.items():
            if keyword in query_lower:
                # Check if this boost requires context
                context_dependent = config.get('context_dependent')
                if context_dependent and entity_context != context_dependent:
                    # Skip this boost - wrong context
                    continue
                
                boost_chapters = config.get('boost_chapters', [])
                boost_multiplier = config.get('chapter_boost_multiplier', 1.0)
                
                for boost_ch in boost_chapters:
                    if boost_ch.lower() in chapter.lower():
                        return boost_multiplier
        
        return 1.0
    
    
    def _filter_exact_matches(
        self,
        results: List[SearchResult],
        direct_hit_clause_ids: List[str],
        doc_type: Optional[str] = None
    ) -> List[SearchResult]:
        """Filter results to include only direct hits and their sub-clauses"""
        if not direct_hit_clause_ids:
            return results
        
        filtered = []
        
        for result in results:
            clause_id = result.clause_id
            is_match = False
            
            for requested_id in direct_hit_clause_ids:
                if clause_id == requested_id or clause_id.startswith(requested_id + "("):
                    is_match = True
                    break
                if requested_id.startswith(clause_id + "("):
                    is_match = True
                    break
            
            if is_match:
                if doc_type:
                    doc_source_lower = result.legal_document_source.lower()
                    type_matches = {
                        'civil': 'civil' in doc_source_lower,
                        'criminal': 'criminal' in doc_source_lower,
                        'constitution': 'constitution' in doc_source_lower
                    }
                    if type_matches.get(doc_type, False):
                        filtered.append(result)
                    elif result.clause_id in direct_hit_clause_ids:
                        filtered.append(result)
                else:
                    filtered.append(result)
        
        if filtered:
            return filtered
        
        # Fallback
        fallback_filtered = []
        for result in results:
            clause_id = result.clause_id
            is_match = False
            
            for requested_id in direct_hit_clause_ids:
                if clause_id == requested_id or clause_id.startswith(requested_id + "("):
                    is_match = True
                    break
            
            if is_match:
                fallback_filtered.append(result)
        
        return fallback_filtered if fallback_filtered else results
    
    def _rerank_results(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """Rerank results using cross-encoder"""
        texts = [r.text for r in results]
        initial_scores = [r.initial_score for r in results]
        
        rerank_results = self.reranker.rerank(
            query=query,
            documents=texts,
            initial_scores=initial_scores,
            top_k=None
        )
        
        for i, rerank_result in enumerate(rerank_results):
            original_idx = rerank_result.index
            results[original_idx].rerank_score = rerank_result.rerank_score
            results[original_idx].final_score = rerank_result.final_score
        
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        return results
    
    def _apply_semantic_filter(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Apply semantic domain filtering to prevent unrelated clauses.
        
        This filters results to keep only those semantically coherent with the query domain.
        Example: "I killed a dog" will filter out homicide clauses, keeping only animal-related ones.
        """
        if not self.semantic_filter:
            return results
        
        if self.verbose:
            print(f"\n[Semantic Filtering]")
            print(f"  Threshold: {self.semantic_filter_threshold:.2f}")
            print(f"  Weight: {self.semantic_filter_weight:.2f}")
        
        # Option 1: Apply penalty-based filtering (keeps more results)
        results_with_penalty = self.semantic_filter.apply_filter_penalty(
            results,
            query,
            penalty_weight=self.semantic_filter_weight
        )
        
        # Option 2: Hard filter (removes unrelated clauses)
        filtered_results = self.semantic_filter.filter_by_domain_coherence(
            results_with_penalty,
            query,
            threshold=self.semantic_filter_threshold
        )
        
        # Re-sort after filtering
        filtered_results.sort(key=lambda x: x.final_score, reverse=True)
        
        if self.verbose:
            removed_count = len(results) - len(filtered_results)
            if removed_count > 0:
                print(f"  Filtered out {removed_count} semantically unrelated results")
        
        return filtered_results
    
    def _deduplicate_by_parent(
        self,
        results: List[SearchResult],
        top_k: int
    ) -> List[SearchResult]:
        """
        Deduplicate results keeping only the best child per parent clause.
        
        When multiple children of the same parent are found (e.g., 22(a), 22(b), 22(c)),
        only the best-scoring one is returned, eliminating duplicate parent information.
        """
        seen_parents = {}  # parent_base_clause -> best SearchResult
        deduplicated = []
        
        for result in results:
            # Extract base clause number (parent)
            parent_base = self._extract_base_clause_number(result.clause_id)
            
            # Create a unique key for parent
            parent_key = (parent_base, result.legal_document_source)
            
            # Keep only the best child for each parent
            if parent_key not in seen_parents:
                seen_parents[parent_key] = result
                deduplicated.append(result)
            else:
                # Compare scores and keep the better one
                if result.final_score > seen_parents[parent_key].final_score:
                    # Remove the old one
                    deduplicated = [r for r in deduplicated if
                                   not (r.clause_id == seen_parents[parent_key].clause_id and
                                        r.legal_document_source == seen_parents[parent_key].legal_document_source)]
                    seen_parents[parent_key] = result
                    deduplicated.append(result)
        
        return deduplicated[:top_k]
    
    def _extract_base_clause_number(self, clause_id: str) -> str:
        """Extract base clause number from clause_id. E.g., "42(6)(e)" -> "42" """
        if clause_id is None:
            return ""
        
        clause_id = str(clause_id)
        
        if '(' in clause_id:
            return clause_id.split('(')[0]
        return clause_id
    
    def _get_parent_text(self, result: SearchResult, doc_type: Optional[str] = None) -> str:
        """Retrieve parent clause text from parent_docs"""
        base_clause_id = self._extract_base_clause_number(result.clause_id)
        
        # Try by base clause ID
        key = (base_clause_id, result.legal_document_source)
        if key in self.parent_lookup:
            return self.parent_lookup[key]['text']
        
        # Try parent_clause_id
        if result.parent_clause_id:
            key = (result.parent_clause_id, result.legal_document_source)
            if key in self.parent_lookup:
                return self.parent_lookup[key]['text']
        
        # Try clause_id itself
        key = (result.clause_id, result.legal_document_source)
        if key in self.parent_lookup:
            return self.parent_lookup[key]['text']
        
        # Fallback
        return result.text
    
    def _create_retrieval_results(
        self,
        search_results: List[SearchResult],
        doc_type: Optional[str]
    ) -> List[RetrievalResult]:
        """Create retrieval results with parent texts"""
        retrieval_results = []
        
        for result in search_results:
            parent_text = self._get_parent_text(result, doc_type)
            parent_clause_id = self._extract_base_clause_number(result.clause_id)
            
            retrieval_result = RetrievalResult(
                clause_id=result.clause_id,
                child_text=result.text,
                parent_text=parent_text,
                parent_clause_id=parent_clause_id,
                legal_document_source=result.legal_document_source,
                part=result.part or "",
                chapter=result.chapter or "",
                final_score=result.final_score,
                is_direct_hit=result.is_direct_hit,
                score_breakdown={
                    'similarity': float(result.similarity_score),
                    'bm25': float(result.bm25_score),
                    'initial': float(result.initial_score),
                    'rerank': float(result.rerank_score),
                    'direct_hit_bonus': float(result.direct_hit_bonus),
                    'final': float(result.final_score)
                }
            )
            retrieval_results.append(retrieval_result)
        
        return retrieval_results
    
    def _convert_to_retrieval_results(
        self,
        search_results: List[SearchResult]
    ) -> List[RetrievalResult]:
        """Convert search results to retrieval results without parent lookup"""
        return [
            RetrievalResult(
                clause_id=r.clause_id,
                child_text=r.text,
                parent_text=r.parent_text or "",
                legal_document_source=r.legal_document_source,
                part=r.part,
                chapter=r.chapter,
                final_score=r.final_score,
                is_direct_hit=r.is_direct_hit,
                score_breakdown={
                    'similarity': float(r.similarity_score),
                    'bm25': float(r.bm25_score),
                    'initial': float(r.initial_score),
                    'rerank': float(r.rerank_score),
                    'direct_hit_bonus': float(r.direct_hit_bonus),
                    'final': float(r.final_score)
                }
            )
            for r in search_results
        ]
