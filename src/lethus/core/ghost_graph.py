"""
Ghost Graph: Entity extraction and linking for pronoun resolution.
Solves the "pronoun problem" by tracking entity relationships across turns.
"""
import json
import re
from typing import List, Dict, Set, Optional
from dataclasses import dataclass, field

from ..config import settings


@dataclass
class Entity:
    """Represents an entity in the Ghost Graph"""
    name: str
    entity_type: str  # PERSON, ORG, CONFIG, FUNCTION, URL, etc.
    aliases: Set[str] = field(default_factory=set)
    linked_entities: Set[str] = field(default_factory=set)


class GhostGraph:
    """
    Lightweight in-memory entity graph for context anchoring.
    Enables entity-based retrieval and pronoun resolution.
    
    spaCy is REQUIRED for proper entity extraction. Without it, only
    technical patterns (env vars, functions, URLs) are detected.
    """
    
    # Technical patterns for code/config extraction
    TECH_PATTERNS = [
        (r'\b[A-Z][A-Z0-9_]{2,}\b', 'CONFIG'),  # ENV_VAR, API_KEY
        (r'\b[a-z_][a-z0-9_]*\([^)]*\)', 'FUNCTION'),  # function_name()
        (r'https?://[^\s]+', 'URL'),
        (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', 'IP'),
        (r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', 'EMAIL'),
        (r'\b(?:localhost|127\.0\.0\.1)(?::\d+)?\b', 'HOST'),  # localhost:5432
        (r'\b\d+\.\d+\.\d+(?:\.\d+)?\b', 'VERSION'),  # Version numbers like 3.11.0
        (r'\bport\s*[:=]?\s*(\d{2,5})\b', 'PORT'),  # port 5432, port=5432
    ]
    
    # Expanded spaCy entity labels to capture
    SPACY_LABELS = {
        "PERSON", "ORG", "GPE", "PRODUCT", "WORK_OF_ART", "EVENT",
        "FAC", "LOC", "NORP", "LAW", "LANGUAGE", "DATE", "TIME",
        "MONEY", "QUANTITY", "CARDINAL", "ORDINAL"
    }
    
    _spacy_checked = False
    _spacy_available = False
    
    def __init__(self, use_spacy: bool = None):
        """
        Args:
            use_spacy: If True (default), use spaCy for NER. Raises error if unavailable.
        """
        self.entities: Dict[str, Entity] = {}
        self.use_spacy = use_spacy if use_spacy is not None else settings.use_spacy
        self._nlp = None
        
        # Force spaCy check on init if enabled
        if self.use_spacy:
            self._ensure_spacy()
    
    @classmethod
    def _ensure_spacy(cls) -> bool:
        """Ensure spaCy is installed and model is available. Raises if not."""
        if cls._spacy_checked:
            return cls._spacy_available
        
        cls._spacy_checked = True
        
        try:
            import spacy
            try:
                spacy.load("en_core_web_sm")
                cls._spacy_available = True
                return True
            except OSError:
                raise RuntimeError(
                    "spaCy model 'en_core_web_sm' not found.\n"
                    "Install it with: python -m spacy download en_core_web_sm"
                )
        except ImportError:
            raise RuntimeError(
                "spaCy is required for Ghost Graph entity extraction.\n"
                "Install it with: pip install spacy && python -m spacy download en_core_web_sm"
            )
    
    def _get_nlp(self):
        """Lazy load spaCy model."""
        if self._nlp is None and self.use_spacy:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
        return self._nlp
    
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Extract entities from text using spaCy NER and regex patterns.
        
        Returns:
            List of {"name": str, "type": str}
        """
        entities = []
        
        # Always run tech patterns (code-specific)
        for pattern, etype in self.TECH_PATTERNS:
            matches = re.findall(pattern, text)
            for match in matches:
                # Handle groups (e.g., PORT pattern captures the number)
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[-1]
                if match and len(match) > 1:
                    entities.append({"name": match, "type": etype})
        
        # Run spaCy NER - this is the primary extraction method
        if self.use_spacy:
            nlp = self._get_nlp()
            if nlp:
                doc = nlp(text)
                for ent in doc.ents:
                    if ent.label_ in self.SPACY_LABELS:
                        # Skip very short entities or pure numbers for certain types
                        if len(ent.text.strip()) > 1:
                            entities.append({"name": ent.text.strip(), "type": ent.label_})
                
                # Also extract noun chunks as potential entities (projects, concepts)
                for chunk in doc.noun_chunks:
                    # Only meaningful noun phrases (not pronouns, not too long)
                    if (2 < len(chunk.text) < 50 and 
                        chunk.root.pos_ in ("NOUN", "PROPN") and
                        chunk.root.text.lower() not in ("it", "this", "that", "we", "they", "i")):
                        entities.append({"name": chunk.text.strip(), "type": "CONCEPT"})
        
        # Deduplicate - prefer longer names and spaCy types over regex
        seen_lower = {}
        for e in entities:
            key = e["name"].lower()
            if key not in seen_lower:
                seen_lower[key] = e
            else:
                # Keep the one with longer name or better type
                existing = seen_lower[key]
                if (len(e["name"]) > len(existing["name"]) or 
                    e["type"] in self.SPACY_LABELS and existing["type"] not in self.SPACY_LABELS):
                    seen_lower[key] = e
        
        return list(seen_lower.values())
    
    def register_entities(
        self,
        entities: List[Dict[str, str]],
        context_entities: Optional[List[str]] = None
    ):
        """
        Add entities to the graph and link co-occurring entities.
        
        Args:
            entities: List of extracted entities
            context_entities: Other entity names in the same turn (for linking)
        """
        entity_names = [e["name"] for e in entities]
        all_context = set(entity_names + (context_entities or []))
        
        for e in entities:
            name = e["name"]
            if name not in self.entities:
                self.entities[name] = Entity(name=name, entity_type=e["type"])
            
            # Link to other entities in the same context
            for other in all_context:
                if other != name:
                    self.entities[name].linked_entities.add(other)
    
    def get_linked_entities(self, entity_name: str, depth: int = 1) -> Set[str]:
        """
        Get entities linked to the given entity (graph traversal).
        
        Args:
            entity_name: Starting entity
            depth: How many hops to traverse
        """
        if entity_name not in self.entities:
            return set()
        
        result = set()
        current_level = {entity_name}
        
        for _ in range(depth):
            next_level = set()
            for name in current_level:
                if name in self.entities:
                    linked = self.entities[name].linked_entities
                    next_level.update(linked)
                    result.update(linked)
            current_level = next_level - result
        
        return result
    
    def find_entity_mentions(self, text: str) -> List[str]:
        """
        Find which known entities are mentioned in the text.
        Useful for pronoun resolution ("it", "that", "the config").
        """
        text_lower = text.lower()
        mentions = []
        
        for name in self.entities:
            if name.lower() in text_lower:
                mentions.append(name)
        
        return mentions
    
    # Semantic mappings: query keywords -> entity types to boost
    SEMANTIC_TYPE_MAPPINGS = {
        # Config/env var queries
        "environment variable": ["CONFIG"],
        "env var": ["CONFIG"],
        "config": ["CONFIG"],
        "setting": ["CONFIG"],
        "secret": ["CONFIG"],
        "api key": ["CONFIG"],
        "credential": ["CONFIG"],
        # Database queries
        "database": ["HOST", "PORT", "CONFIG", "IP"],
        "db": ["HOST", "PORT", "CONFIG", "IP"],
        "port": ["HOST", "PORT", "IP"],
        "connection": ["HOST", "PORT", "URL", "CONFIG"],
        # Person queries  
        "who": ["PERSON"],
        "person": ["PERSON"],
        "team": ["PERSON", "ORG"],
        "colleague": ["PERSON"],
        # Time queries
        "when": ["DATE", "TIME"],
        "timeline": ["DATE", "TIME"],
        "deadline": ["DATE", "TIME"],
        "schedule": ["DATE", "TIME"],
        # Location/infra queries
        "where": ["GPE", "LOC", "HOST", "URL"],
        "server": ["HOST", "IP", "URL"],
        "url": ["URL", "HOST"],
        "endpoint": ["URL", "FUNCTION"],
    }
    
    def _get_query_type_boosts(self, query: str) -> Set[str]:
        """Determine which entity types should be boosted based on query semantics."""
        query_lower = query.lower()
        types_to_boost = set()
        
        for keyword, entity_types in self.SEMANTIC_TYPE_MAPPINGS.items():
            if keyword in query_lower:
                types_to_boost.update(entity_types)
        
        return types_to_boost
    
    def boost_similarities(
        self,
        query: str,
        turns: List[dict],
        similarities: "np.ndarray",
        boost_factor: float = None
    ) -> "np.ndarray":
        """
        Boost similarity scores using multiple strategies:
        1. Entity linking - boost turns with entities linked to query entities
        2. Semantic type matching - ADDITIVE boost for entity types relevant to query
        3. Keyword matching - boost turns containing query keywords
        
        Args:
            query: The query text
            turns: List of turn dictionaries with 'entities' field
            similarities: Original similarity scores
            boost_factor: Multiplicative boost (default from config)
            
        Returns:
            Boosted similarity scores
        """
        import numpy as np
        import json as json_module
        
        boost = boost_factor or settings.ghost_graph_boost
        boosted = similarities.copy()
        
        # Strategy 1: Entity linking (multiplicative boost)
        mentioned_entities = self.find_entity_mentions(query)
        for entity in mentioned_entities:
            linked = self.get_linked_entities(entity, depth=1)
            for i, turn in enumerate(turns):
                turn_entities_raw = turn.get("entities", "[]")
                if isinstance(turn_entities_raw, str):
                    turn_entities = json_module.loads(turn_entities_raw)
                else:
                    turn_entities = turn_entities_raw
                
                turn_entity_names = [
                    e["name"] if isinstance(e, dict) else e 
                    for e in turn_entities
                ]
                
                for linked_ent in linked:
                    if linked_ent in turn_entity_names:
                        boosted[i] *= boost
                        break
        
        # Strategy 2: Semantic type matching - ADDITIVE boost to overcome embedding gaps
        # This is critical when embeddings don't capture semantic relationships
        # e.g., "environment variables" should boost turns with CONFIG entities
        types_to_boost = self._get_query_type_boosts(query)
        if types_to_boost:
            # Use additive boost proportional to number of matching entity types
            additive_boost = settings.ghost_graph_additive_boost
            for i, turn in enumerate(turns):
                turn_entities_raw = turn.get("entities", "[]")
                if isinstance(turn_entities_raw, str):
                    turn_entities = json_module.loads(turn_entities_raw)
                else:
                    turn_entities = turn_entities_raw
                
                # Count matching entity types
                type_matches = 0
                for ent in turn_entities:
                    ent_type = ent.get("type") if isinstance(ent, dict) else None
                    if ent_type is None:
                        ent_name = ent if isinstance(ent, str) else ent.get("name", "")
                        if ent_name in self.entities:
                            ent_type = self.entities[ent_name].entity_type
                    
                    if ent_type in types_to_boost:
                        type_matches += 1
                
                # Add boost for each matching type (capped at 3)
                if type_matches > 0:
                    boosted[i] += additive_boost * min(type_matches, 3)
        
        # Strategy 3: Direct keyword matching in turn content
        query_words = set(query.lower().split())
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "how", "do", "did", "we", "i", "you", "to", "on", "in", "for"}
        query_keywords = query_words - stop_words
        
        for i, turn in enumerate(turns):
            content_lower = turn.get("content", "").lower()
            # Count keyword matches
            matches = sum(1 for kw in query_keywords if kw in content_lower and len(kw) > 2)
            if matches >= 2:  # Multiple keyword match
                boosted[i] *= (1 + 0.1 * matches)  # Additive boost based on matches
        
        return boosted
    
    def to_json(self) -> str:
        """Serialize graph for storage."""
        data = {}
        for name, entity in self.entities.items():
            data[name] = {
                "type": entity.entity_type,
                "aliases": list(entity.aliases),
                "links": list(entity.linked_entities)
            }
        return json.dumps(data)
    
    def from_json(self, json_str: str):
        """Load graph from storage."""
        data = json.loads(json_str)
        for name, info in data.items():
            self.entities[name] = Entity(
                name=name,
                entity_type=info["type"],
                aliases=set(info.get("aliases", [])),
                linked_entities=set(info.get("links", []))
            )
    
    def clear(self):
        """Clear all entities from the graph."""
        self.entities.clear()
    
    def get_state_summary(self, limit: int = 20) -> str:
        """Get a human-readable summary of the graph state."""
        if not self.entities:
            return "Ghost Graph is empty. No entities tracked yet."
        
        lines = ["--- GHOST GRAPH STATE ---"]
        for name, entity in list(self.entities.items())[:limit]:
            links = list(entity.linked_entities)[:5]
            lines.append(f"  {name} ({entity.entity_type}) -> {links}")
        
        if len(self.entities) > limit:
            lines.append(f"  ... and {len(self.entities) - limit} more entities")
        
        lines.append("--- END GRAPH ---")
        return "\n".join(lines)
