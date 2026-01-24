from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from openai import OpenAI
from .database import Turn, PinnedMemory, Conversation
from .milvus_client import milvus_client
from .importance import ImportanceDetector, EmbeddingGenerator
from .retrieval import TurnPairer, RelevanceScorer, DYCPSpanSelector
from .context_processing import ContextTrimmer, ConfidenceChecker
from .config import settings


class MemoryOrchestrator:
    """
    Main orchestrator for the memory flow pipeline
    Implements the complete flow from new turn to LLM response
    """
    
    def __init__(self, openai_api_key: str, db: Session):
        self.db = db
        self.openai_api_key = openai_api_key
        
        # Initialize components
        self.importance_detector = ImportanceDetector(openai_api_key)
        self.embedding_gen = EmbeddingGenerator(openai_api_key)
        self.relevance_scorer = RelevanceScorer(self.embedding_gen)
        self.dycp_selector = DYCPSpanSelector()
        self.context_trimmer = ContextTrimmer()
        self.confidence_checker = ConfidenceChecker()
        self.openai_client = OpenAI(api_key=openai_api_key)
    
    async def process_message(
        self,
        conversation_id: int,
        user_message: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Process a new user message through the complete memory pipeline
        
        Flow:
        1. New turn creation
        2. Importance detection → pinned memory
        3. Turn pairing
        4. Query embedding (+ variants)
        5. Weighted relevance scoring
        6. DYCP span selection
        7. Span merging & gap filling
        8. Token-aware trimming
        9. Confidence check & fallback
        10. Prompt assembly
        11. LLM response
        """
        
        # Step 1: Get or create conversation
        conversation = self.db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        
        if not conversation:
            conversation = Conversation(
                user_id=user_id,
                title=user_message[:50] + "..." if len(user_message) > 50 else user_message
            )
            self.db.add(conversation)
            self.db.commit()
            self.db.refresh(conversation)
            conversation_id = conversation.id
        
        # Get current turn number
        last_turn = self.db.query(Turn).filter(
            Turn.conversation_id == conversation_id
        ).order_by(Turn.turn_number.desc()).first()
        
        turn_number = (last_turn.turn_number + 1) if last_turn else 1
        
        # Step 2-3: Retrieve pinned memories
        pinned_memories = self._get_pinned_memories(conversation_id)
        
        # Step 4: Query embedding with variants
        query_variants = self.embedding_gen.generate_query_variants(user_message)
        
        # Step 5: Weighted relevance scoring (multi-query retrieval)
        relevant_turns = self.relevance_scorer.multi_query_retrieval(
            queries=query_variants,
            conversation_id=conversation_id,
            top_k=15
        )
        
        # Step 6-7: DYCP span selection and merging
        spans = self.dycp_selector.select_spans(
            relevant_turns=relevant_turns,
            db=self.db,
            conversation_id=conversation_id
        )
        merged_spans = self.dycp_selector.merge_spans(spans)
        
        # Step 8: Token-aware trimming
        system_prompt = self._build_system_prompt()
        trimmed_spans = self.context_trimmer.trim_spans(
            spans=merged_spans,
            pinned_memories=pinned_memories,
            system_prompt=system_prompt,
            current_message=user_message
        )
        
        # Step 9: Confidence check & fallback
        confidence = self.confidence_checker.check_confidence(
            spans=trimmed_spans,
            pinned_memories=pinned_memories
        )
        
        if not confidence["confident"] and confidence["fallback"] == "recent":
            fallback_turns = self.confidence_checker.get_fallback_context(
                db=self.db,
                conversation_id=conversation_id,
                num_turns=5
            )
            # Add fallback as a span
            if fallback_turns:
                trimmed_spans = [{
                    "start_turn": fallback_turns[0]["turn_number"],
                    "end_turn": fallback_turns[-1]["turn_number"],
                    "relevance_score": 0.5,
                    "turns": fallback_turns,
                    "is_fallback": True
                }]
        
        # Step 10: Prompt assembly
        messages = self._assemble_prompt(
            system_prompt=system_prompt,
            pinned_memories=pinned_memories,
            spans=trimmed_spans,
            user_message=user_message
        )
        
        # Step 11: LLM response
        assistant_response = self._get_llm_response(messages)
        
        # Save the turn
        turn = Turn(
            conversation_id=conversation_id,
            turn_number=turn_number,
            user_message=user_message,
            assistant_message=assistant_response,
            importance_score=0.0,  # Will be calculated below
            metadata={
                "confidence": confidence,
                "num_spans": len(trimmed_spans),
                "num_pinned": len(pinned_memories)
            }
        )
        self.db.add(turn)
        self.db.commit()
        self.db.refresh(turn)
        
        # Calculate importance and potentially pin
        importance_score = self.importance_detector.calculate_importance(
            user_message, assistant_response
        )
        turn.importance_score = importance_score
        
        if self.importance_detector.should_pin(importance_score):
            turn.is_pinned = True
            pinned_memory = PinnedMemory(
                conversation_id=conversation_id,
                turn_id=turn.id,
                content=TurnPairer.create_turn_pair(user_message, assistant_response),
                importance_score=importance_score
            )
            self.db.add(pinned_memory)
        
        self.db.commit()
        
        # Store embedding in Milvus
        turn_text = TurnPairer.create_turn_pair(user_message, assistant_response)
        embedding = self.embedding_gen.generate_embedding(turn_text)
        
        milvus_client.insert_embedding(
            turn_id=turn.id,
            conversation_id=conversation_id,
            embedding=embedding,
            text=turn_text,
            turn_number=turn_number
        )
        
        return {
            "conversation_id": conversation_id,
            "turn_id": turn.id,
            "message": assistant_response,
            "retrieved_context": {
                "spans": trimmed_spans,
                "pinned_memories": pinned_memories,
                "confidence": confidence
            },
            "metadata": {
                "importance_score": importance_score,
                "is_pinned": turn.is_pinned,
                "turn_number": turn_number
            }
        }
    
    def _get_pinned_memories(self, conversation_id: int) -> List[Dict[str, Any]]:
        """Retrieve pinned memories for the conversation"""
        pinned = self.db.query(PinnedMemory).filter(
            PinnedMemory.conversation_id == conversation_id
        ).order_by(PinnedMemory.importance_score.desc()).all()
        
        return [
            {
                "id": p.id,
                "content": p.content,
                "importance_score": p.importance_score,
                "turn_id": p.turn_id
            }
            for p in pinned
        ]
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt"""
        return """You are a helpful AI assistant with access to conversation history. 
Use the provided context from previous conversations to give relevant and personalized responses.
If the context contains important information about the user's preferences or past discussions, 
incorporate that naturally into your response."""
    
    def _assemble_prompt(
        self,
        system_prompt: str,
        pinned_memories: List[Dict[str, Any]],
        spans: List[Dict[str, Any]],
        user_message: str
    ) -> List[Dict[str, str]]:
        """Assemble the final prompt for the LLM"""
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add pinned memories as context
        if pinned_memories:
            pinned_context = "Important information to remember:\n\n"
            for memory in pinned_memories[:5]:  # Limit to top 5
                pinned_context += f"- {memory['content']}\n\n"
            
            messages.append({
                "role": "system",
                "content": pinned_context
            })
        
        # Add relevant conversation spans
        if spans:
            context = "Relevant conversation history:\n\n"
            for span in spans:
                context += f"--- Conversation context ---\n"
                for turn in span["turns"]:
                    context += f"User: {turn['user_message']}\n"
                    context += f"Assistant: {turn['assistant_message']}\n\n"
            
            messages.append({
                "role": "system",
                "content": context
            })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        return messages
    
    def _get_llm_response(self, messages: List[Dict[str, str]]) -> str:
        """Get response from OpenAI"""
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
