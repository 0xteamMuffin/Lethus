import json
import re
from typing import List, Dict, Set, Optional
from dataclasses import dataclass, field

from ..config import settings


@dataclass
class Entity:
    name: str
    entity_type: str
    aliases: Set[str] = field(default_factory=set)
    linked_entities: Set[str] = field(default_factory=set)


class GhostGraph:
    TECH_PATTERNS = [
        (r'\b[A-Z][A-Z0-9_]{2,}\b', 'CONFIG'),
        (r'\b[a-z_][a-z0-9_]*\([^)]*\)', 'FUNCTION'),
        (r'https?://[^\s]+', 'URL'),
        (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', 'IP'),
        (r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', 'EMAIL'),
        (r'\b(?:localhost|127\.0\.0\.1)(?::\d+)?\b', 'HOST'),
        (r'\b\d+\.\d+\.\d+(?:\.\d+)?\b', 'VERSION'),
        (r'\bport\s*[:=]?\s*(\d{2,5})\b', 'PORT'),
    ]
    
    SPACY_LABELS = {
        "PERSON", "ORG", "GPE", "PRODUCT", "WORK_OF_ART", "EVENT",
        "FAC", "LOC", "NORP", "LAW", "LANGUAGE", "DATE", "TIME",
        "MONEY", "QUANTITY", "CARDINAL", "ORDINAL"
    }
    
    _spacy_checked = False
    _spacy_available = False
    
    def __init__(self, use_spacy: bool = None):
        self.entities: Dict[str, Entity] = {}
        self.use_spacy = use_spacy if use_spacy is not None else settings.use_spacy
        self._nlp = None
        
        if self.use_spacy:
            self._ensure_spacy()
    
    @classmethod
    def _ensure_spacy(cls) -> bool:
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
        if self._nlp is None and self.use_spacy:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
        return self._nlp
    
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        entities = []
        
        for pattern, etype in self.TECH_PATTERNS:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[-1]
                if match and len(match) > 1:
                    entities.append({"name": match, "type": etype})
        
        if self.use_spacy:
            nlp = self._get_nlp()
            if nlp:
                doc = nlp(text)
                for ent in doc.ents:
                    if ent.label_ in self.SPACY_LABELS:
                        if len(ent.text.strip()) > 1:
                            entities.append({"name": ent.text.strip(), "type": ent.label_})
                
                for chunk in doc.noun_chunks:
                    if (2 < len(chunk.text) < 50 and 
                        chunk.root.pos_ in ("NOUN", "PROPN") and
                        chunk.root.text.lower() not in ("it", "this", "that", "we", "they", "i")):
                        entities.append({"name": chunk.text.strip(), "type": "CONCEPT"})
        
        seen_lower = {}
        for e in entities:
            key = e["name"].lower()
            if key not in seen_lower:
                seen_lower[key] = e
            else:
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
        entity_names = [e["name"] for e in entities]
        all_context = set(entity_names + (context_entities or []))
        
        for e in entities:
            name = e["name"]
            if name not in self.entities:
                self.entities[name] = Entity(name=name, entity_type=e["type"])
            
            for other in all_context:
                if other != name:
                    self.entities[name].linked_entities.add(other)
    
    def get_linked_entities(self, entity_name: str, depth: int = 1) -> Set[str]:
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
        text_lower = text.lower()
        mentions = []
        
        for name in self.entities:
            if name.lower() in text_lower:
                mentions.append(name)
        
        return mentions
    
    SEMANTIC_TYPE_MAPPINGS = {
        "environment variable": ["CONFIG"],
        "env var": ["CONFIG"],
        "config": ["CONFIG"],
        "setting": ["CONFIG"],
        "secret": ["CONFIG"],
        "api key": ["CONFIG"],
        "credential": ["CONFIG"],
        "database": ["HOST", "PORT", "CONFIG", "IP"],
        "db": ["HOST", "PORT", "CONFIG", "IP"],
        "port": ["HOST", "PORT", "IP"],
        "connection": ["HOST", "PORT", "URL", "CONFIG"],
        "who": ["PERSON"],
        "person": ["PERSON"],
        "team": ["PERSON", "ORG"],
        "colleague": ["PERSON"],
        "when": ["DATE", "TIME"],
        "timeline": ["DATE", "TIME"],
        "deadline": ["DATE", "TIME"],
        "schedule": ["DATE", "TIME"],
        "where": ["GPE", "LOC", "HOST", "URL"],
        "server": ["HOST", "IP", "URL"],
        "url": ["URL", "HOST"],
        "endpoint": ["URL", "FUNCTION"],
    }
    
    def _get_query_type_boosts(self, query: str) -> Set[str]:
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
        import numpy as np
        import json as json_module
        
        boost = boost_factor or settings.ghost_graph_boost
        boosted = similarities.copy()
        
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
        
        types_to_boost = self._get_query_type_boosts(query)
        if types_to_boost:
            additive_boost = settings.ghost_graph_additive_boost
            for i, turn in enumerate(turns):
                turn_entities_raw = turn.get("entities", "[]")
                if isinstance(turn_entities_raw, str):
                    turn_entities = json_module.loads(turn_entities_raw)
                else:
                    turn_entities = turn_entities_raw
                
                type_matches = 0
                for ent in turn_entities:
                    ent_type = ent.get("type") if isinstance(ent, dict) else None
                    if ent_type is None:
                        ent_name = ent if isinstance(ent, str) else ent.get("name", "")
                        if ent_name in self.entities:
                            ent_type = self.entities[ent_name].entity_type
                    
                    if ent_type in types_to_boost:
                        type_matches += 1
                
                if type_matches > 0:
                    boosted[i] += additive_boost * min(type_matches, 3)
        
        query_words = set(query.lower().split())
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "how", "do", "did", "we", "i", "you", "to", "on", "in", "for"}
        query_keywords = query_words - stop_words
        
        for i, turn in enumerate(turns):
            content_lower = turn.get("content", "").lower()
            matches = sum(1 for kw in query_keywords if kw in content_lower and len(kw) > 2)
            if matches >= 2:
                boosted[i] *= (1 + 0.1 * matches)
        
        return boosted
    
    def to_json(self) -> str:
        data = {}
        for name, entity in self.entities.items():
            data[name] = {
                "type": entity.entity_type,
                "aliases": list(entity.aliases),
                "links": list(entity.linked_entities)
            }
        return json.dumps(data)
    
    def from_json(self, json_str: str):
        data = json.loads(json_str)
        for name, info in data.items():
            self.entities[name] = Entity(
                name=name,
                entity_type=info["type"],
                aliases=set(info.get("aliases", [])),
                linked_entities=set(info.get("links", []))
            )
    
    def clear(self):
        self.entities.clear()
    
    def get_state_summary(self, limit: int = 20) -> str:
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
