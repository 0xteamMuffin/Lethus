"""
Ghost Graph: Lightweight entity extraction and linking for pronoun resolution.
Extracts entities from conversation turns and enables entity-based retrieval.
"""
import json
import re
from typing import List, Dict, Set, Optional
from dataclasses import dataclass, field

@dataclass
class Entity:
    name: str
    entity_type: str  # PERSON, ORG, CODE, TECH, etc.
    aliases: Set[str] = field(default_factory=set)
    linked_entities: Set[str] = field(default_factory=set)

class GhostGraph:
    """
    Lightweight in-memory entity graph for context anchoring.
    Solves the "pronoun problem" by tracking entity relationships.
    """
    
    def __init__(self, use_spacy: bool = True):
        """
        Args:
            use_spacy: If True, use spaCy for NER. If False, use regex patterns.
        """
        self.entities: Dict[str, Entity] = {}
        self.use_spacy = use_spacy
        self._nlp = None
        
        # Technical patterns for code/config extraction
        self.tech_patterns = [
            (r'\b[A-Z][A-Z0-9_]{2,}\b', 'CONFIG'),  # ENV_VAR, API_KEY
            (r'\b[a-z_][a-z0-9_]*\([^)]*\)', 'FUNCTION'),  # function_name()
            (r'https?://[^\s]+', 'URL'),
            (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', 'IP'),
            (r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', 'EMAIL'),
        ]
    
    def _get_nlp(self):
        """Lazy load spaCy model."""
        if self._nlp is None and self.use_spacy:
            try:
                import spacy
                # Use small model for speed
                self._nlp = spacy.load("en_core_web_sm")
            except (ImportError, OSError):
                print("Warning: spaCy not available, falling back to regex extraction")
                self.use_spacy = False
        return self._nlp
    
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Extract entities from text using spaCy NER or regex fallback.
        
        Returns:
            List of {"name": str, "type": str}
        """
        entities = []
        
        # Always run tech patterns (code-specific)
        for pattern, etype in self.tech_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                entities.append({"name": match, "type": etype})
        
        # Run spaCy NER if available
        if self.use_spacy:
            nlp = self._get_nlp()
            if nlp:
                doc = nlp(text)
                for ent in doc.ents:
                    # Filter to important types
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
    
    def register_entities(self, entities: List[Dict[str, str]], context_entities: Optional[List[str]] = None):
        """
        Add entities to the graph and link them if they appear in the same context.
        
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
        Get entities linked to the given entity (traverses the graph).
        
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
            current_level = next_level - result  # Don't revisit
        
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
