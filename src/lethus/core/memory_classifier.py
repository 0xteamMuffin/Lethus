"""
Memory Classifier: Extracts structured memories (rules, facts, experiences) from conversations.
Transforms raw DYCP spans into actionable memory categories.
"""
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class MemoryType(str, Enum):
    RULE = "rule"           # User preferences, constraints, instructions
    FACT = "fact"           # Stated information, data, assertions
    EXPERIENCE = "experience"  # Past events, interactions, outcomes
    ENTITY = "entity"       # Named entities (people, orgs, configs)


@dataclass
class ClassifiedMemory:
    """A classified memory entry"""
    memory_type: MemoryType
    content: str
    source_turn: Optional[int] = None
    confidence: float = 1.0
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            "type": self.memory_type.value,
            "content": self.content,
            "source_turn": self.source_turn,
            "confidence": self.confidence,
            "metadata": self.metadata or {}
        }


# Pattern-based classification for common statements
RULE_PATTERNS = [
    "i don't like", "i hate", "don't use", "never", "always use",
    "i prefer", "please don't", "i want", "make sure to", "remember to",
    "don't forget", "i need", "i require", "avoid", "only use"
]

FACT_PATTERNS = [
    "my name is", "i am", "i work at", "i live in", "i have",
    "the password is", "the key is", "it's located", "the url is",
    "it costs", "there are", "it uses", "it requires"
]


class MemoryClassifier:
    """
    Classifies conversation turns into structured memory types.
    Uses pattern matching for fast classification, with optional LLM enhancement.
    """
    
    def __init__(self, use_llm: bool = False, api_key: Optional[str] = None):
        self.use_llm = use_llm
        self.api_key = api_key
    
    def classify_turn(self, role: str, content: str, turn_number: int = 0) -> List[ClassifiedMemory]:
        """
        Classify a single turn into memory types.
        User messages are primary source of rules/facts.
        """
        memories = []
        content_lower = content.lower()
        
        # Only classify user messages for rules/facts (assistant responses are not preferences)
        if role.lower() == "user":
            # Check for rules (preferences, constraints)
            for pattern in RULE_PATTERNS:
                if pattern in content_lower:
                    memories.append(ClassifiedMemory(
                        memory_type=MemoryType.RULE,
                        content=content,
                        source_turn=turn_number,
                        confidence=0.85,
                        metadata={"pattern": pattern}
                    ))
                    break
            
            # Check for facts (only if not already a rule)
            if not memories:
                for pattern in FACT_PATTERNS:
                    if pattern in content_lower:
                        memories.append(ClassifiedMemory(
                            memory_type=MemoryType.FACT,
                            content=content,
                            source_turn=turn_number,
                            confidence=0.85,
                            metadata={"pattern": pattern}
                        ))
                        break
        
        return memories
    
    def classify_spans(
        self, 
        spans: List[Dict], 
        entities: List[str] = None
    ) -> Dict[str, List[Dict]]:
        """
        Classify retrieved spans into structured memory categories.
        
        Returns dict with keys: rules, facts, experiences, entities
        """
        result = {
            "rules": [],
            "facts": [],
            "experiences": [],
            "entities": []
        }
        
        for span in spans:
            for turn in span.get("turns", []):
                role = turn.get("role", "")
                content = turn.get("content", "")
                
                classified = self.classify_turn(role, content)
                
                if classified:
                    for mem in classified:
                        if mem.memory_type == MemoryType.RULE:
                            result["rules"].append({
                                "content": mem.content,
                                "relevance": span.get("relevance_score", 0.8)
                            })
                        elif mem.memory_type == MemoryType.FACT:
                            result["facts"].append({
                                "content": mem.content,
                                "relevance": span.get("relevance_score", 0.8)
                            })
                else:
                    # Default: classify as experience (past interaction)
                    result["experiences"].append({
                        "role": role,
                        "content": content,
                        "relevance": span.get("relevance_score", 0.8),
                        "span_label": f"Span {spans.index(span) + 1}"
                    })
        
        # Add entities
        if entities:
            for entity in entities:
                result["entities"].append({
                    "name": entity,
                    "type": "entity"
                })
        
        return result
    
    def extract_with_llm(self, content: str, api_key: str) -> List[ClassifiedMemory]:
        """
        Use LLM to extract structured memories from content.
        More accurate but slower than pattern matching.
        """
        if not api_key:
            return []
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """Extract structured memories from the user message.
Return JSON array with objects containing:
- type: "rule" (preference/constraint), "fact" (information), or "experience" (past event)
- content: the extracted memory statement
- confidence: 0.0-1.0

Only extract clear, actionable memories. Return [] if none found."""},
                    {"role": "user", "content": content}
                ],
                temperature=0,
                max_tokens=500
            )
            
            result = json.loads(response.choices[0].message.content)
            memories = []
            for item in result:
                memories.append(ClassifiedMemory(
                    memory_type=MemoryType(item["type"]),
                    content=item["content"],
                    confidence=item.get("confidence", 0.8)
                ))
            return memories
            
        except Exception:
            return []
