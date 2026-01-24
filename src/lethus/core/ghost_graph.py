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
    """
    
    # Technical patterns for code/config extraction
    TECH_PATTERNS = [
        (r'\b[A-Z][A-Z0-9_]{2,}\b', 'CONFIG'),  # ENV_VAR, API_KEY
        (r'\b[a-z_][a-z0-9_]*\([^)]*\)', 'FUNCTION'),  # function_name()
        (r'https?://[^\s]+', 'URL'),
        (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', 'IP'),
        (r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', 'EMAIL'),
    ]
    
    def __init__(self, use_spacy: bool = None):
        """
        Args:
            use_spacy: If True, use spaCy for NER. If False, use regex patterns only.
        """
        self.entities: Dict[str, Entity] = {}
        self.use_spacy = use_spacy if use_spacy is not None else settings.use_spacy
        self._nlp = None
    
    def _get_nlp(self):
        """Lazy load spaCy model."""
        if self._nlp is None and self.use_spacy:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
            except (ImportError, OSError):
                print("Warning: spaCy not available, falling back to regex extraction")
                self.use_spacy = False
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
                entities.append({"name": match, "type": etype})
        
        # Run spaCy NER if available
        if self.use_spacy:
            nlp = self._get_nlp()
            if nlp:
                doc = nlp(text)
                for ent in doc.ents:
                    if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "WORK_OF_ART", "EVENT"]:
                        entities.append({"name": ent.text, "type": ent.label_})
        
        # Deduplicate
        seen = set()
        unique = []
        for e in entities:
            key = (e["name"].lower(), e["type"])
            if key not in seen:
                seen.add(key)
                unique.append(e)
        
        return unique
    
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
    
    def boost_similarities(
        self,
        query: str,
        turns: List[dict],
        similarities: "np.ndarray",
        boost_factor: float = None
    ) -> "np.ndarray":
        """
        Boost similarity scores for turns containing entities linked to query entities.
        
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
        
        mentioned_entities = self.find_entity_mentions(query)
        if not mentioned_entities:
            return similarities
        
        boosted = similarities.copy()
        
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
