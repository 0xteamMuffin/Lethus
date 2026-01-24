"""
Integration tests for Lethus REST API.
Tests the full chat flow, DYCP retrieval, and database operations.

Requires:
- PostgreSQL running on localhost:5432
- Milvus running on localhost:19530
- OpenAI API key in .env or LETHUS_OPENAI_API_KEY
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Load environment
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

import time
import json
from typing import List, Dict

# Test infrastructure first
def test_database_connectivity():
    """Test PostgreSQL connection."""
    print("\n" + "="*60)
    print("TEST: PostgreSQL Connectivity")
    print("="*60)
    
    from lethus.storage.postgres import get_postgres_storage, init_db
    from sqlalchemy import text
    
    try:
        storage = get_postgres_storage()
        init_db()
        
        # Test session
        session = storage.create_session()
        result = session.execute(text("SELECT 1")).fetchone()
        session.close()
        
        print("  PostgreSQL: CONNECTED")
        print(f"  URL: {storage.url}")
        return True
    except Exception as e:
        print(f"  PostgreSQL: FAILED - {e}")
        return False


def test_milvus_connectivity():
    """Test Milvus connection."""
    print("\n" + "="*60)
    print("TEST: Milvus Connectivity")
    print("="*60)
    
    from lethus.storage.milvus import get_milvus_storage
    
    try:
        storage = get_milvus_storage()
        count = storage.get_turn_count()
        
        print("  Milvus: CONNECTED")
        print(f"  URI: {storage.uri}")
        print(f"  Collection: {storage.collection_name}")
        print(f"  Current turn count: {count}")
        return True
    except Exception as e:
        print(f"  Milvus: FAILED - {e}")
        return False


def test_embeddings():
    """Test embedding provider."""
    print("\n" + "="*60)
    print("TEST: Embedding Provider")
    print("="*60)
    
    from lethus.core.embeddings import get_embedding_provider
    
    try:
        embedder = get_embedding_provider()
        
        test_text = "This is a test message for embeddings."
        embedding = embedder.embed_text(test_text)
        
        print(f"  Provider: {type(embedder).__name__}")
        print(f"  Embedding dimension: {len(embedding)}")
        print(f"  Sample values: [{embedding[0]:.4f}, {embedding[1]:.4f}, ...]")
        return True
    except Exception as e:
        print(f"  Embeddings: FAILED - {e}")
        return False


def test_ghost_graph():
    """Test Ghost Graph entity extraction."""
    print("\n" + "="*60)
    print("TEST: Ghost Graph Entity Extraction")
    print("="*60)
    
    from lethus.core.ghost_graph import GhostGraph
    
    try:
        graph = GhostGraph(use_spacy=True)
        
        test_texts = [
            "Set DATABASE_URL to postgresql://localhost:5432/mydb",
            "Talk to John about the Kubernetes deployment",
            "The API_KEY expires next month",
        ]
        
        for text in test_texts:
            entities = graph.extract_entities(text)
            print(f"  '{text[:40]}...'")
            print(f"    -> {[e['name'] for e in entities]}")
        
        print(f"\n  Total entities tracked: {len(graph.entities)}")
        return True
    except Exception as e:
        print(f"  Ghost Graph: FAILED - {e}")
        return False


def test_dycp_core():
    """Test DYCP algorithm."""
    print("\n" + "="*60)
    print("TEST: DYCP Core Algorithm")
    print("="*60)
    
    import numpy as np
    from lethus.core.dycp import DYCPCore
    
    try:
        dycp = DYCPCore(tau=0.6, theta=1.0)
        
        # Simulate similarity scores
        similarities = np.array([0.3, 0.3, 0.85, 0.82, 0.3, 0.2, 0.2])
        spans = dycp.get_pruned_indices(similarities)
        
        print(f"  Similarities: {similarities}")
        print(f"  Selected spans: {spans}")
        print(f"  Expected: [(2, 3)]")
        
        if spans == [(2, 3)]:
            print("  PASS")
            return True
        else:
            print("  FAIL - incorrect span selection")
            return False
    except Exception as e:
        print(f"  DYCP: FAILED - {e}")
        return False


def test_full_chat_flow():
    """Test the complete chat API flow."""
    print("\n" + "="*60)
    print("TEST: Full Chat API Flow")
    print("="*60)
    
    from lethus.storage.postgres import get_postgres_storage, init_db, Conversation, Turn
    from lethus.storage.milvus import get_milvus_storage
    from lethus.core.dycp import DYCPCore
    from lethus.core.ghost_graph import GhostGraph
    from lethus.core.embeddings import get_embedding_provider
    from lethus.config import settings
    import numpy as np
    
    try:
        # Initialize
        init_db()
        storage_pg = get_postgres_storage()
        storage_milvus = get_milvus_storage()
        dycp = DYCPCore()
        ghost = GhostGraph(use_spacy=True)
        embedder = get_embedding_provider()
        
        # Create test conversation
        session = storage_pg.create_session()
        
        # Clean up any existing test conversation
        test_conv = session.query(Conversation).filter(
            Conversation.title == "API_TEST_CONVERSATION"
        ).first()
        if test_conv:
            session.query(Turn).filter(Turn.conversation_id == test_conv.id).delete()
            session.delete(test_conv)
            session.commit()
            storage_milvus.clear_conversation(test_conv.id)
        
        # Create new conversation
        conv = Conversation(user_id="test_user", title="API_TEST_CONVERSATION")
        session.add(conv)
        session.commit()
        session.refresh(conv)
        print(f"  Created conversation ID: {conv.id}")
        
        # Simulate conversation turns
        test_turns = [
            ("user", "Let's set up a PostgreSQL database on port 5432"),
            ("assistant", "I'll configure DATABASE_URL=postgresql://localhost:5432/myapp"),
            ("user", "Now add Redis caching"),
            ("assistant", "I'll set REDIS_HOST=localhost:6379 for caching"),
            ("user", "What about authentication?"),
            ("assistant", "I recommend JWT tokens with python-jose. Set JWT_SECRET_KEY in your environment"),
        ]
        
        print(f"\n  Storing {len(test_turns)} turns...")
        for i, (role, content) in enumerate(test_turns):
            # Extract entities
            entities = ghost.extract_entities(content)
            ghost.register_entities(entities)
            entity_names = [e["name"] for e in entities]
            
            # Generate embedding
            embedding = embedder.embed_text(content)
            
            # Store in Milvus
            storage_milvus.add_turn(
                role=role,
                content=content,
                embedding=embedding,
                conversation_id=conv.id,
                entities=json.dumps(entity_names)
            )
            
            # Store in PostgreSQL (simplified)
            if role == "user" and i + 1 < len(test_turns):
                turn = Turn(
                    conversation_id=conv.id,
                    turn_number=(i // 2) + 1,
                    user_message=content,
                    assistant_message=test_turns[i + 1][1],
                    entities_json=json.dumps(entity_names)
                )
                session.add(turn)
        
        session.commit()
        print(f"  Stored turns in PostgreSQL and Milvus")
        
        # Test retrieval
        query = "What port is the database running on?"
        print(f"\n  Query: '{query}'")
        
        turns, history_embs = storage_milvus.get_all_turns_ordered(conv.id)
        print(f"  Retrieved {len(turns)} turns from Milvus")
        
        query_emb = embedder.embed_text(query)
        turn_indices = np.array([t["turn_index"] for t in turns])
        
        # Compute relevance
        similarities = dycp.compute_relevance(
            query_emb,
            history_embs,
            turn_indices=turn_indices,
            current_turn=len(turns),
            apply_decay=True
        )
        
        # Ghost Graph boost
        boosted = ghost.boost_similarities(query, turns, similarities)
        
        # Get spans
        spans = dycp.get_pruned_indices(boosted)
        
        print(f"\n  DYCP Results:")
        print(f"    Selected spans: {spans}")
        
        # Check if database port turns are retrieved
        retrieved_content = []
        for start, end in spans:
            for i in range(start, end + 1):
                if i < len(turns):
                    retrieved_content.append(turns[i]["content"])
        
        print(f"    Retrieved content:")
        for c in retrieved_content[:3]:
            print(f"      - {c[:60]}...")
        
        # Verify correct retrieval
        found_5432 = any("5432" in c for c in retrieved_content)
        print(f"\n  Database port 5432 in retrieval: {found_5432}")
        
        # Cleanup
        session.query(Turn).filter(Turn.conversation_id == conv.id).delete()
        session.delete(conv)
        session.commit()
        storage_milvus.clear_conversation(conv.id)
        session.close()
        print(f"  Cleaned up test data")
        
        if found_5432:
            print("\n  PASS - Full chat flow works correctly")
            return True
        else:
            print("\n  FAIL - DYCP did not retrieve the correct context")
            return False
            
    except Exception as e:
        import traceback
        print(f"  Full Chat Flow: FAILED")
        print(f"  Error: {e}")
        traceback.print_exc()
        return False


def test_api_endpoints():
    """Test FastAPI endpoints directly using TestClient."""
    print("\n" + "="*60)
    print("TEST: FastAPI Endpoints")
    print("="*60)
    
    try:
        from fastapi.testclient import TestClient
        from lethus.api.rest import app
        
        client = TestClient(app)
        
        # Test health check
        print("  Testing / (health check)...")
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        print(f"    Status: {response.json()['status']}")
        
        # Test stats endpoint
        print("  Testing /api/stats...")
        response = client.get("/api/stats")
        assert response.status_code == 200
        stats = response.json()
        print(f"    Total turns: {stats['total_turns']}")
        print(f"    Decay lambda: {stats['decay_lambda']}")
        
        # Test conversation creation
        print("  Testing POST /api/conversations...")
        response = client.post("/api/conversations", json={
            "user_id": "test_api_user",
            "title": "API Test Conversation"
        })
        assert response.status_code == 200
        conv = response.json()
        conv_id = conv["id"]
        print(f"    Created conversation ID: {conv_id}")
        
        # Test get conversation
        print(f"  Testing GET /api/conversations/{conv_id}...")
        response = client.get(f"/api/conversations/{conv_id}")
        assert response.status_code == 200
        print(f"    Title: {response.json()['title']}")
        
        # Test delete conversation (cleanup)
        print(f"  Testing DELETE /api/conversations/{conv_id}...")
        response = client.delete(f"/api/conversations/{conv_id}")
        assert response.status_code == 200
        print(f"    Deleted successfully")
        
        print("\n  PASS - All API endpoints working")
        return True
        
    except Exception as e:
        import traceback
        print(f"  FastAPI Endpoints: FAILED - {e}")
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("="*70)
    print("LETHUS API INTEGRATION TEST SUITE")
    print("="*70)
    
    results = {}
    
    # Infrastructure tests
    results["PostgreSQL"] = test_database_connectivity()
    results["Milvus"] = test_milvus_connectivity()
    results["Embeddings"] = test_embeddings()
    results["Ghost Graph"] = test_ghost_graph()
    results["DYCP Core"] = test_dycp_core()
    
    # Only run full tests if infrastructure is working
    if all([results["PostgreSQL"], results["Milvus"], results["Embeddings"]]):
        results["Full Chat Flow"] = test_full_chat_flow()
        results["API Endpoints"] = test_api_endpoints()
    else:
        print("\n  SKIPPING full tests - infrastructure not ready")
        results["Full Chat Flow"] = None
        results["API Endpoints"] = None
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, result in results.items():
        if result is True:
            status = "PASS"
            passed += 1
        elif result is False:
            status = "FAIL"
            failed += 1
        else:
            status = "SKIP"
            skipped += 1
        print(f"  {name:<20} {status}")
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0 and skipped == 0:
        print("\nALL TESTS PASSED")
        return 0
    else:
        print("\nSOME TESTS FAILED OR SKIPPED")
        return 1


if __name__ == "__main__":
    exit(main())
