"""
DYCP Benchmark Test Suite
Tests the core DYCP algorithm against known-correct answers.

Validates:
1. Span selection accuracy (does it find the right context?)
2. Token reduction (how much context is saved?)
3. Entity extraction (does Ghost Graph work?)
4. End-to-end retrieval quality
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from typing import List, Tuple, Dict
import json

from lethus.core.dycp import DYCPCore
from lethus.core.ghost_graph import GhostGraph
from lethus.core.embeddings import get_embedding_provider


class BenchmarkDataset:
    """
    Test dataset with known relevant spans for each query.
    
    Each test case has:
    - conversation: List of turns
    - query: User's question
    - expected_spans: Turn indices that SHOULD be retrieved
    - expected_answer_keywords: Keywords that should appear in good context
    """
    
    @staticmethod
    def get_database_scenario() -> Dict:
        """Scenario: Long conversation, query about database config."""
        return {
            "name": "database_port_query",
            "description": "User asks about database port after many unrelated turns",
            "conversation": [
                {"role": "user", "content": "Hey, let's set up the project database"},
                {"role": "assistant", "content": "Sure! What database would you prefer? PostgreSQL, MySQL, or MongoDB?"},
                {"role": "user", "content": "Let's go with PostgreSQL on port 5432"},
                {"role": "assistant", "content": "Great choice. I'll configure DATABASE_URL=postgresql://localhost:5432/myapp"},
                {"role": "user", "content": "Perfect. Now let's work on the API"},
                {"role": "assistant", "content": "I'll set up FastAPI. What endpoints do you need?"},
                {"role": "user", "content": "We need /users, /products, and /orders endpoints"},
                {"role": "assistant", "content": "Got it. I'll create RESTful CRUD endpoints for each resource."},
                {"role": "user", "content": "Also add authentication with JWT tokens"},
                {"role": "assistant", "content": "I'll use python-jose for JWT. Set JWT_SECRET_KEY in your environment."},
                {"role": "user", "content": "Now let's discuss caching strategies"},
                {"role": "assistant", "content": "We should use Redis. I'll configure REDIS_HOST=localhost:6379"},
                {"role": "user", "content": "What about the frontend framework?"},
                {"role": "assistant", "content": "Next.js would be great for SSR and the API routes integration."},
                {"role": "user", "content": "Sounds good. Let's deploy to AWS later"},
                {"role": "assistant", "content": "I'll set up EC2 instances and RDS for the PostgreSQL database."},
            ],
            "query": "What port is the database running on?",
            "expected_spans": [(2, 3)],  # Turns 2-3 mention port 5432
            "required_keywords": ["5432", "PostgreSQL", "port"],
            "noise_keywords": ["Redis", "JWT", "Next.js", "frontend"],  # Should NOT dominate
        }
    
    @staticmethod
    def get_person_reference_scenario() -> Dict:
        """Scenario: Query about a person mentioned earlier."""
        return {
            "name": "person_reference",
            "description": "User asks about a colleague mentioned in early conversation",
            "conversation": [
                {"role": "user", "content": "I was talking to Sarah from the DevOps team yesterday"},
                {"role": "assistant", "content": "What did Sarah say about the infrastructure?"},
                {"role": "user", "content": "She mentioned we need to upgrade to Kubernetes 1.28"},
                {"role": "assistant", "content": "That's a significant upgrade. Does Sarah have a timeline in mind?"},
                {"role": "user", "content": "She said Q2 2026, but it depends on the budget approval"},
                {"role": "assistant", "content": "Got it. I'll prepare the migration plan for review."},
                {"role": "user", "content": "Now let's talk about the frontend redesign"},
                {"role": "assistant", "content": "Sure, what aspects of the UI need updating?"},
                {"role": "user", "content": "The dashboard needs a complete overhaul"},
                {"role": "assistant", "content": "I'll create wireframes for the new dashboard layout."},
                {"role": "user", "content": "Make sure to include dark mode support"},
                {"role": "assistant", "content": "Will do. I'll use CSS custom properties for theming."},
            ],
            "query": "When did Sarah say we need to complete the Kubernetes upgrade?",
            "expected_spans": [(0, 5)],  # Sarah + Kubernetes + Q2 timeline
            "required_keywords": ["Sarah", "Q2", "2026", "Kubernetes"],
            "noise_keywords": ["dashboard", "wireframes", "dark mode"],
        }
    
    @staticmethod
    def get_multi_topic_scenario() -> Dict:
        """Scenario: Query touches multiple past topics."""
        return {
            "name": "multi_topic_query",
            "description": "Query about config that spans multiple discussion points",
            "conversation": [
                {"role": "user", "content": "What environment variables do we need?"},
                {"role": "assistant", "content": "For the API, you'll need API_KEY and API_SECRET"},
                {"role": "user", "content": "What about database connections?"},
                {"role": "assistant", "content": "Add DATABASE_URL with the PostgreSQL connection string"},
                {"role": "user", "content": "And for caching?"},
                {"role": "assistant", "content": "Set REDIS_URL=redis://localhost:6379/0"},
                {"role": "user", "content": "Now let's discuss the deployment pipeline"},
                {"role": "assistant", "content": "I'll set up GitHub Actions for CI/CD"},
                {"role": "user", "content": "Add staging and production environments"},
                {"role": "assistant", "content": "Each environment will have its own secrets in GitHub"},
            ],
            "query": "List all the environment variables we configured",
            "expected_spans": [(0, 5)],  # All the env var discussions
            "required_keywords": ["API_KEY", "DATABASE_URL", "REDIS_URL"],
            "noise_keywords": ["GitHub Actions", "staging"],
        }

    @staticmethod
    def all_scenarios() -> List[Dict]:
        return [
            BenchmarkDataset.get_database_scenario(),
            BenchmarkDataset.get_person_reference_scenario(),
            BenchmarkDataset.get_multi_topic_scenario(),
        ]


class DYCPBenchmark:
    """Benchmark runner for DYCP algorithm."""
    
    def __init__(self, use_real_embeddings: bool = True, debug: bool = False):
        # Use tuned parameters for benchmark (more inclusive than paper defaults)
        # Lower tau = include more turns, higher theta = longer spans
        self.dycp = DYCPCore(tau=0.3, theta=1.5, decay_lambda=0.98)
        self.ghost = GhostGraph(use_spacy=True)
        self.use_real_embeddings = use_real_embeddings
        self.debug = debug
        
        if use_real_embeddings:
            print("Initializing embedding provider...")
            self.embedder = get_embedding_provider()
            print(f"Using embeddings: {type(self.embedder).__name__}")
        else:
            self.embedder = None
    
    def embed_text(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        if self.use_real_embeddings:
            return self.embedder.embed_text(text)
        else:
            # Deterministic fake embeddings for testing
            np.random.seed(hash(text) % 2**32)
            return np.random.randn(384).astype(np.float32)
    
    def run_scenario(self, scenario: Dict) -> Dict:
        """Run a single benchmark scenario."""
        conversation = scenario["conversation"]
        query = scenario["query"]
        expected_spans = scenario["expected_spans"]
        required_keywords = scenario["required_keywords"]
        noise_keywords = scenario.get("noise_keywords", [])
        
        # Reset ghost graph for this scenario
        self.ghost.clear()
        
        # Step 1: Extract entities and embed all turns
        history_embs = []
        for turn in conversation:
            # Extract entities
            entities = self.ghost.extract_entities(turn["content"])
            self.ghost.register_entities(entities)
            # Store full entity objects (with type) for semantic boosting
            turn["entities"] = entities
            
            # Embed
            emb = self.embed_text(turn["content"])
            history_embs.append(emb)
        
        history_embs = np.stack(history_embs)
        
        # Step 2: Embed query and compute relevance
        query_emb = self.embed_text(query)
        current_turn = len(conversation)
        turn_indices = np.arange(current_turn)
        
        similarities = self.dycp.compute_relevance(
            query_emb,
            history_embs,
            turn_indices=turn_indices,
            current_turn=current_turn,
            apply_decay=True
        )
        
        # Step 3: Ghost Graph boosting (use higher boost for better recall)
        boosted = self.ghost.boost_similarities(query, conversation, similarities, boost_factor=1.5)
        
        # Debug: show similarity scores
        if self.debug:
            print(f"\n  Similarity scores (base -> boosted):")
            for i, (sim, boost) in enumerate(zip(similarities, boosted)):
                if sim > 0.2 or boost > 0.2:
                    ents = [e.get("name", e) if isinstance(e, dict) else e 
                            for e in conversation[i].get("entities", [])][:3]
                    marker = " <--" if boost > sim else ""
                    print(f"    [{i:2d}] {sim:.3f} -> {boost:.3f}{marker} {conversation[i]['content'][:40]}... {ents}")
        
        # Step 4: Run Kadane's Algorithm
        selected_spans = self.dycp.get_pruned_indices(boosted)
        
        # Step 5: Format context
        context = self.dycp.format_context(conversation, selected_spans)
        
        # Evaluation metrics
        results = {
            "scenario": scenario["name"],
            "query": query,
            "selected_spans": selected_spans,
            "expected_spans": expected_spans,
            "context": context,
        }
        
        # Metric 1: Span overlap (did we get the right turns?)
        selected_turns = set()
        for start, end in selected_spans:
            selected_turns.update(range(start, end + 1))
        
        expected_turns = set()
        for start, end in expected_spans:
            expected_turns.update(range(start, end + 1))
        
        overlap = selected_turns & expected_turns
        results["span_precision"] = len(overlap) / len(selected_turns) if selected_turns else 0
        results["span_recall"] = len(overlap) / len(expected_turns) if expected_turns else 0
        
        # Metric 2: Keyword coverage
        context_lower = context.lower()
        required_hits = sum(1 for k in required_keywords if k.lower() in context_lower)
        results["keyword_recall"] = required_hits / len(required_keywords)
        
        noise_hits = sum(1 for k in noise_keywords if k.lower() in context_lower)
        results["noise_ratio"] = noise_hits / len(noise_keywords) if noise_keywords else 0
        
        # Metric 3: Token reduction
        full_tokens = sum(len(t["content"].split()) for t in conversation)
        selected_tokens = 0
        for start, end in selected_spans:
            for i in range(start, end + 1):
                selected_tokens += len(conversation[i]["content"].split())
        
        results["full_tokens"] = full_tokens
        results["selected_tokens"] = selected_tokens
        results["token_reduction"] = 1 - (selected_tokens / full_tokens) if full_tokens else 0
        
        # Metric 4: Entity extraction quality
        results["entities_tracked"] = len(self.ghost.entities)
        
        return results
    
    def run_all(self) -> List[Dict]:
        """Run all benchmark scenarios."""
        all_results = []
        
        for scenario in BenchmarkDataset.all_scenarios():
            print(f"\n{'='*60}")
            print(f"Running: {scenario['name']}")
            print(f"{'='*60}")
            
            results = self.run_scenario(scenario)
            all_results.append(results)
            
            print(f"\nQuery: \"{results['query']}\"")
            print(f"Expected spans: {results['expected_spans']}")
            print(f"Selected spans: {results['selected_spans']}")
            print(f"\nMetrics:")
            print(f"  Span Precision: {results['span_precision']:.2%}")
            print(f"  Span Recall:    {results['span_recall']:.2%}")
            print(f"  Keyword Recall: {results['keyword_recall']:.2%}")
            print(f"  Noise Ratio:    {results['noise_ratio']:.2%}")
            print(f"  Token Reduction:{results['token_reduction']:.2%}")
            print(f"  Entities Found: {results['entities_tracked']}")
            print(f"\nRetrieved Context:")
            print(results['context'][:500] + "..." if len(results['context']) > 500 else results['context'])
        
        return all_results
    
    def summarize(self, results: List[Dict]) -> Dict:
        """Compute aggregate metrics."""
        avg_precision = np.mean([r["span_precision"] for r in results])
        avg_recall = np.mean([r["span_recall"] for r in results])
        avg_keyword_recall = np.mean([r["keyword_recall"] for r in results])
        avg_noise = np.mean([r["noise_ratio"] for r in results])
        avg_reduction = np.mean([r["token_reduction"] for r in results])
        
        return {
            "total_scenarios": len(results),
            "avg_span_precision": avg_precision,
            "avg_span_recall": avg_recall,
            "avg_keyword_recall": avg_keyword_recall,
            "avg_noise_ratio": avg_noise,
            "avg_token_reduction": avg_reduction,
            "passed": avg_keyword_recall >= 0.8 and avg_recall >= 0.5,
        }


def test_ghost_graph_entity_extraction():
    """Test that Ghost Graph extracts entities correctly."""
    print("\n" + "="*60)
    print("TEST: Ghost Graph Entity Extraction")
    print("="*60)
    
    ghost = GhostGraph(use_spacy=True)
    
    test_cases = [
        {
            "text": "Set DATABASE_URL to postgresql://localhost:5432/mydb",
            "expected_types": ["CONFIG", "HOST"],
        },
        {
            "text": "Talk to Sarah from DevOps about the Kubernetes upgrade",
            "expected_types": ["PERSON"],  # Sarah should be detected
        },
        {
            "text": "Configure REDIS_HOST=127.0.0.1:6379 for caching",
            "expected_types": ["CONFIG", "IP"],
        },
        {
            "text": "John said the API_KEY expires next week",
            "expected_types": ["PERSON", "CONFIG", "DATE"],
        },
    ]
    
    all_passed = True
    for case in test_cases:
        entities = ghost.extract_entities(case["text"])
        entity_types = {e["type"] for e in entities}
        
        print(f"\nText: \"{case['text'][:60]}...\"")
        print(f"  Extracted: {entities}")
        print(f"  Types found: {entity_types}")
        print(f"  Expected types: {case['expected_types']}")
        
        # Check if at least some expected types are found
        found_expected = any(t in entity_types for t in case["expected_types"])
        if found_expected:
            print("  PASS")
        else:
            print("  FAIL - Missing expected entity types")
            all_passed = False
    
    return all_passed


def test_dycp_span_selection():
    """Test that DYCP correctly selects relevant spans."""
    print("\n" + "="*60)
    print("TEST: DYCP Span Selection (Unit Test)")
    print("="*60)
    
    dycp = DYCPCore(tau=0.6, theta=1.0)
    
    # Test case: Clear signal with one relevant block
    similarities = np.array([
        0.3, 0.3, 0.85, 0.82, 0.3,  # Turns 2-3 are highly relevant
        0.2, 0.2, 0.2, 0.2, 0.2,     # Noise
    ])
    
    spans = dycp.get_pruned_indices(similarities)
    
    print(f"Similarities: {similarities}")
    print(f"Selected spans: {spans}")
    
    # Should select turns 2-3 (the high-relevance block)
    expected_in_span = {2, 3}
    selected_turns = set()
    for start, end in spans:
        selected_turns.update(range(start, end + 1))
    
    if expected_in_span.issubset(selected_turns):
        print("PASS - Correctly identified relevant span")
        return True
    else:
        print(f"FAIL - Expected {expected_in_span} in selection, got {selected_turns}")
        return False


def main():
    """Run all benchmark tests."""
    print("="*70)
    print("LETHUS DYCP BENCHMARK SUITE")
    print("="*70)
    
    # Test 1: Entity extraction
    entity_test_passed = test_ghost_graph_entity_extraction()
    
    # Test 2: Span selection unit test
    span_test_passed = test_dycp_span_selection()
    
    # Test 3: Full benchmark with real embeddings
    print("\n" + "="*70)
    print("FULL BENCHMARK (with real embeddings)")
    print("="*70)
    
    try:
        benchmark = DYCPBenchmark(use_real_embeddings=True, debug=True)
        results = benchmark.run_all()
        summary = benchmark.summarize(results)
        
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)
        print(f"Total Scenarios:     {summary['total_scenarios']}")
        print(f"Avg Span Precision:  {summary['avg_span_precision']:.2%}")
        print(f"Avg Span Recall:     {summary['avg_span_recall']:.2%}")
        print(f"Avg Keyword Recall:  {summary['avg_keyword_recall']:.2%}")
        print(f"Avg Noise Ratio:     {summary['avg_noise_ratio']:.2%}")
        print(f"Avg Token Reduction: {summary['avg_token_reduction']:.2%}")
        print(f"\nOverall: {'PASS' if summary['passed'] else 'NEEDS WORK'}")
        
        benchmark_passed = summary['passed']
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
        benchmark_passed = False
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Entity Extraction: {'PASS' if entity_test_passed else 'FAIL'}")
    print(f"Span Selection:    {'PASS' if span_test_passed else 'FAIL'}")
    print(f"Full Benchmark:    {'PASS' if benchmark_passed else 'FAIL'}")
    
    all_passed = entity_test_passed and span_test_passed and benchmark_passed
    print(f"\n{'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
