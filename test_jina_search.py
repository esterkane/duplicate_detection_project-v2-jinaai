#!/usr/bin/env python3
"""
Demo script to test the enhanced Jina AI search with reranking.

This script demonstrates:
1. Basic search with Jina AI embeddings
2. A/B comparison between baseline and reranked results
3. Performance evaluation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.es_client import get_es_client
from src.search_jina import knn_search_with_reranking, SearchPipeline
from src.embeddings_jina import load_jina_embedding_model
from src.config import INDEX_NAME, EMBEDDING_FIELD, JINA_MODEL_NAME
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_basic_search():
    """Demonstrate basic search functionality with Jina AI."""
    print("=" * 60)
    print("🚀 JINA AI SEARCH DEMO - Basic Search")
    print("=" * 60)
    
    # Initialize components
    print("Initializing Elasticsearch client and Jina AI model...")
    es_client = get_es_client()
    jina_model = load_jina_embedding_model()
    
    # Test query
    user_query = "How to configure Elasticsearch cluster settings"
    print(f"\n📝 Test Query: '{user_query}'")
    
    # Generate query embedding
    print("🔄 Generating query embedding...")
    query_vector = jina_model.encode([user_query], task='retrieval.query')[0].tolist()
    print(f"✅ Query embedding generated: {len(query_vector)} dimensions")
    
    # Search with reranking
    print("\n🔍 Searching with Jina AI reranking...")
    start_time = time.time()
    
    hits = knn_search_with_reranking(
        es_client=es_client,
        query_vector=query_vector,
        user_query=user_query,
        index_name=INDEX_NAME,
        embedding_field=EMBEDDING_FIELD,
        k=10,
        num_candidates=200,
        use_reranker=True
    )
    
    search_time = time.time() - start_time
    print(f"⏱️ Search completed in {search_time:.2f} seconds")
    print(f"📊 Found {len(hits)} results")
    
    # Display top results
    print("\n🏆 Top 5 Results:")
    for i, hit in enumerate(hits[:5], 1):
        source = hit['_source']
        score = hit['_score']
        rerank_score = hit.get('_rerank_score', 'N/A')
        original_score = hit.get('_original_score', 'N/A')
        
        print(f"\n{i}. {source.get('content_title', 'N/A')}")
        print(f"   Score: {score:.4f} (original: {original_score}, rerank: {rerank_score})")
        print(f"   Summary: {source.get('content_summary', '')[:100]}...")
        print(f"   Products: {source.get('metadata_products', [])}")

def demo_ab_comparison():
    """Demonstrate A/B comparison between baseline and reranked results."""
    print("\n" + "=" * 60)
    print("📊 JINA AI SEARCH DEMO - A/B Comparison")
    print("=" * 60)
    
    # Initialize search pipeline
    es_client = get_es_client()
    jina_model = load_jina_embedding_model()
    
    search_pipeline = SearchPipeline(
        es_client=es_client,
        index_name=INDEX_NAME,
        use_reranker=True
    )
    
    # Test queries
    test_queries = [
        "Elasticsearch cluster configuration",
        "Kibana dashboard setup",
        "Logstash pipeline configuration",
        "Beats data collection setup"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing Query: '{query}'")
        
        # Generate embedding
        query_vector = jina_model.encode([query], task='retrieval.query')[0].tolist()
        
        # Get comparison results
        start_time = time.time()
        comparison = search_pipeline.compare_with_baseline(
            query_vector=query_vector,
            user_query=query,
            k=5
        )
        comparison_time = time.time() - start_time
        
        baseline_hits = comparison['baseline']
        reranked_hits = comparison['reranked']
        
        print(f"⏱️ Comparison completed in {comparison_time:.2f} seconds")
        
        # Show top 3 results side by side
        print("\n📈 Baseline vs Reranked (Top 3):")
        print("-" * 80)
        print(f"{'Baseline':<35} | {'Reranked':<35}")
        print("-" * 80)
        
        for i in range(min(3, len(baseline_hits), len(reranked_hits))):
            baseline_title = baseline_hits[i]['_source'].get('content_title', 'N/A')[:30]
            reranked_title = reranked_hits[i]['_source'].get('content_title', 'N/A')[:30]
            baseline_score = baseline_hits[i]['_score']
            reranked_score = reranked_hits[i]['_score']
            
            print(f"{baseline_title:<30} ({baseline_score:.3f}) | {reranked_title:<30} ({reranked_score:.3f})")
        
        # Check for differences in top results
        baseline_ids = [hit['_source']['article_id'] for hit in baseline_hits[:3]]
        reranked_ids = [hit['_source']['article_id'] for hit in reranked_hits[:3]]
        
        if baseline_ids != reranked_ids:
            print("✨ Reranking changed the order of results!")
        else:
            print("🔄 Results order remained the same")

def demo_performance_analysis():
    """Analyze performance differences between baseline and reranked search."""
    print("\n" + "=" * 60)
    print("⚡ JINA AI SEARCH DEMO - Performance Analysis")
    print("=" * 60)
    
    es_client = get_es_client()
    jina_model = load_jina_embedding_model()
    
    query = "Elasticsearch performance optimization"
    query_vector = jina_model.encode([query], task='retrieval.query')[0].tolist()
    
    # Test baseline search
    print("🔍 Testing baseline search (no reranking)...")
    baseline_times = []
    for i in range(3):
        start_time = time.time()
        baseline_hits = knn_search_with_reranking(
            es_client=es_client,
            query_vector=query_vector,
            user_query=query,
            index_name=INDEX_NAME,
            embedding_field=EMBEDDING_FIELD,
            k=10,
            use_reranker=False
        )
        baseline_times.append(time.time() - start_time)
    
    # Test reranked search
    print("🎯 Testing reranked search...")
    reranked_times = []
    for i in range(3):
        start_time = time.time()
        reranked_hits = knn_search_with_reranking(
            es_client=es_client,
            query_vector=query_vector,
            user_query=query,
            index_name=INDEX_NAME,
            embedding_field=EMBEDDING_FIELD,
            k=10,
            use_reranker=True
        )
        reranked_times.append(time.time() - start_time)
    
    # Performance comparison
    avg_baseline = sum(baseline_times) / len(baseline_times)
    avg_reranked = sum(reranked_times) / len(reranked_times)
    overhead = ((avg_reranked - avg_baseline) / avg_baseline) * 100
    
    print(f"\n📊 Performance Results:")
    print(f"   Baseline avg: {avg_baseline:.3f}s")
    print(f"   Reranked avg: {avg_reranked:.3f}s")
    print(f"   Overhead: {overhead:.1f}%")
    
    if overhead < 50:
        print("✅ Reranking overhead is acceptable!")
    else:
        print("⚠️ Consider optimizing reranking parameters")

def main():
    """Run all demo functions."""
    print("🎯 Starting Jina AI Search Enhancement Demo")
    print(f"📍 Using index: {INDEX_NAME}")
    print(f"🤖 Using model: {JINA_MODEL_NAME}")
    
    try:
        # Run demos
        demo_basic_search()
        demo_ab_comparison()
        demo_performance_analysis()
        
        print("\n" + "=" * 60)
        print("🎉 Demo completed successfully!")
        print("=" * 60)
        print("\n💡 Next steps:")
        print("1. Run 'streamlit run run_pipeline.py' to test the UI")
        print("2. Try different queries and compare baseline vs reranked results")
        print("3. Adjust reranking parameters based on your use case")
        print("4. Monitor performance with real workloads")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())