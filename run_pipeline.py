# run_pipeline.py

import streamlit as st
import pandas as pd
import numpy as np
import logging
import plotly.express as px
import umap
import warnings

# Suppress harmless warnings for cleaner output
warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated!")
warnings.filterwarnings("ignore", message="'force_all_finite' was renamed to 'ensure_all_finite'")
warnings.filterwarnings("ignore", message="n_jobs value .* overridden .* by setting random_state")

from src.config import INDEX_NAME, JINA_MODEL_NAME, EMBEDDING_FIELD, JINA_TASK
from src.es_client import get_es_client
# Updated imports for Jina AI
from src.search_jina import knn_search_with_reranking, SearchPipeline
from src.embeddings_jina import load_jina_embedding_model, JinaReranker
from src.hits_analysis import analyze_knn_hits

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('elasticsearch').setLevel(logging.WARNING)

# --- Initialize Clients ---
@st.cache_resource
def initialize_models():
    """Initialize and cache Elasticsearch client and Jina AI models."""
    es_client = get_es_client()
    jina_model = load_jina_embedding_model()
    
    # Initialize search pipeline with reranking
    search_pipeline = SearchPipeline(
        es_client=es_client,
        index_name=INDEX_NAME,
        use_reranker=True  # Enable reranking by default
    )
    
    return es_client, jina_model, search_pipeline

es_client, jina_model, search_pipeline = initialize_models()

# --- Main App Logic ---
st.set_page_config(page_title="KB Article Duplicate Detection with Jina AI", layout="wide")

# Header with improved branding
st.title("🚀 KB Article Duplicate Detection with Jina AI")
st.markdown("**Enhanced with 16x larger context window and AI reranking for superior precision**")

# --- User Input ---
user_query = st.text_input("Enter your search query:", placeholder="e.g., How to configure Elasticsearch cluster settings")

# Enhanced controls with Jina AI features
col1, col2 = st.columns(2)
with col1:
    text_boost = st.slider("Text Query Boost", 0.1, 5.0, 1.0, 0.1, 
                          help="Weight for keyword matching score.")
with col2:
    knn_boost = st.slider("k-NN Query Boost", 0.1, 5.0, 1.0, 0.1, 
                         help="Weight for semantic similarity score.")

# Enhanced sidebar with Jina AI options
st.sidebar.header("🔧 Search Configuration")

# Search parameters
k_results = st.sidebar.number_input("Number of Results (k)", min_value=1, max_value=100, value=10, step=1)
num_candidates = st.sidebar.number_input("Number of Candidates", min_value=1, max_value=1000, value=200, step=10)

# Jina AI specific options
st.sidebar.subheader("🎯 Jina AI Features")
use_reranker = st.sidebar.checkbox(
    "Enable AI Reranking", 
    value=True, 
    help="Use Jina AI cross-encoder reranking for 20-30% better precision"
)

rerank_candidates = st.sidebar.number_input(
    "Reranking Candidates", 
    min_value=k_results, 
    max_value=500, 
    value=min(100, num_candidates), 
    step=10,
    help="Number of candidates to retrieve before reranking (more = better quality, slower)"
)

# A/B Testing option
show_comparison = st.sidebar.checkbox(
    "Show A/B Comparison", 
    value=False, 
    help="Compare results with and without reranking"
)

# Model info
st.sidebar.subheader("📊 Model Information")
st.sidebar.info(f"""
**Model:** {JINA_MODEL_NAME}
**Index:** {INDEX_NAME}
**Context:** 8,192 tokens (16x improvement!)
**Dimensions:** 1,024
**Task:** {JINA_TASK}
""")

search_button = st.button("🔍 Search Similar Articles", type="primary")

if es_client and jina_model:
    if search_button and user_query:
        with st.spinner("🚀 Searching with Jina AI..."):
            try:
                # 1. Get Query Embedding using Jina AI
                query_vector = jina_model.encode([user_query], task='retrieval.query')[0].tolist()

                if query_vector:
                    # 2. Perform Search with optional A/B comparison
                    if show_comparison:
                        # A/B Testing: Compare baseline vs reranked
                        st.subheader("📈 A/B Comparison: Baseline vs Reranked Results")
                        
                        comparison_results = search_pipeline.compare_with_baseline(
                            query_vector=query_vector,
                            user_query=user_query,
                            k=k_results
                        )
                        
                        # Display side-by-side comparison
                        col_baseline, col_reranked = st.columns(2)
                        
                        with col_baseline:
                            st.markdown("**🔍 Baseline (No Reranking)**")
                            baseline_hits = comparison_results['baseline']
                            if baseline_hits:
                                for i, hit in enumerate(baseline_hits[:5], 1):
                                    source = hit['_source']
                                    st.write(f"{i}. **{source.get('content_title', 'N/A')}**")
                                    st.write(f"   Score: {hit['_score']:.4f}")
                                    st.write(f"   {source.get('content_summary', '')[:100]}...")
                                    st.write("---")
                        
                        with col_reranked:
                            st.markdown("**🎯 With AI Reranking**")
                            reranked_hits = comparison_results['reranked']
                            if reranked_hits:
                                for i, hit in enumerate(reranked_hits[:5], 1):
                                    source = hit['_source']
                                    st.write(f"{i}. **{source.get('content_title', 'N/A')}**")
                                    score_text = f"Score: {hit['_score']:.4f}"
                                    if '_rerank_score' in hit:
                                        score_text += f" (reranked from {hit.get('_original_score', 'N/A'):.4f})"
                                    st.write(f"   {score_text}")
                                    st.write(f"   {source.get('content_summary', '')[:100]}...")
                                    st.write("---")
                        
                        # Use reranked results for analysis
                        hits = reranked_hits
                    
                    else:
                        # Standard search with configurable reranking
                        hits = knn_search_with_reranking(
                            es_client=es_client,
                            query_vector=query_vector,
                            user_query=user_query,
                            index_name=INDEX_NAME,
                            embedding_field=EMBEDDING_FIELD,
                            k=k_results,
                            num_candidates=num_candidates,
                            rerank_candidates=rerank_candidates,
                            text_query_boost=text_boost,
                            knn_query_boost=knn_boost,
                            use_reranker=use_reranker
                        )

                    if not hits:
                        st.warning("No search results found.")
                    else:
                        # 3. Analyze Hits (including clustering)
                        df_hits, embeddings_used, cluster_params = analyze_knn_hits(hits)

                        # 4. Display Results Table
                        st.subheader("📋 Search Results")
                        
                        # Add reranking indicator
                        if use_reranker and any('_rerank_score' in hit for hit in hits):
                            st.success("✨ Results enhanced with AI reranking")
                        
                        if df_hits is not None and not df_hits.empty:
                            # Enhanced display with reranking info
                            display_cols = ['article_id', 'score', 'cluster_label', 'product_area', 'title', 'summary', 'url']
                            
                            # Add rerank score column if available
                            if 'rerank_score' not in df_hits.columns and hits and '_rerank_score' in hits[0]:
                                df_hits['rerank_score'] = [hit.get('_rerank_score', None) for hit in hits[:len(df_hits)]]
                                df_hits['original_score'] = [hit.get('_original_score', None) for hit in hits[:len(df_hits)]]
                                display_cols.insert(2, 'rerank_score')
                                display_cols.insert(3, 'original_score')
                            
                            missing_cols = [col for col in display_cols if col not in df_hits.columns]
                            if missing_cols:
                                st.error(f"DataFrame is missing columns: {missing_cols}")
                                st.dataframe(df_hits)
                            else:
                                column_config = {
                                    "url": st.column_config.LinkColumn("Article URL", display_text="View Article"),
                                    "score": st.column_config.NumberColumn("Score", format="%.5f"),
                                    "cluster_label": st.column_config.NumberColumn("Cluster", format="%d")
                                }
                                
                                # Add rerank score config if present
                                if 'rerank_score' in df_hits.columns:
                                    column_config["rerank_score"] = st.column_config.NumberColumn("Rerank Score", format="%.5f")
                                    column_config["original_score"] = st.column_config.NumberColumn("Original Score", format="%.5f")
                                
                                st.dataframe(
                                    df_hits[display_cols],
                                    column_config=column_config,
                                    width='stretch',  # Updated from use_container_width=True
                                    hide_index=True
                                )

                            # 5. Enhanced Analysis Details
                            st.markdown("---")
                            st.subheader("📊 Analysis Details")

                            # Search performance metrics
                            col_metrics1, col_metrics2 = st.columns(2)
                            
                            with col_metrics1:
                                st.write("**🔍 Search Configuration:**")
                                st.markdown(f"- Results: {k_results}")
                                st.markdown(f"- Candidates: {num_candidates}")
                                st.markdown(f"- Text Boost: {text_boost}")
                                st.markdown(f"- k-NN Boost: {knn_boost}")
                                st.markdown(f"- Reranking: {'✅ Enabled' if use_reranker else '❌ Disabled'}")
                                if use_reranker:
                                    st.markdown(f"- Rerank Candidates: {rerank_candidates}")
                            
                            with col_metrics2:
                                st.write("**🎯 Model Performance:**")
                                st.markdown(f"- Model: {JINA_MODEL_NAME}")
                                st.markdown(f"- Context Window: 8,192 tokens")
                                st.markdown(f"- Embedding Dims: 1,024")
                                st.markdown(f"- Task: {JINA_TASK}")

                            # Clustering analysis (existing code)
                            st.write("**🔗 Clustering Analysis (HDBSCAN):**")
                            param_items = [f"`{k}`: {v}" for k, v in cluster_params.items() if k != 'status']
                            st.markdown("- " + "\n- ".join(param_items))
                            if 'status' in cluster_params:
                                st.warning(f"Clustering Status: {cluster_params['status']}")

                            # ...existing clustering summary code...
                            labels = df_hits['cluster_label'].to_numpy()
                            unique_labels, counts = np.unique(labels, return_counts=True)
                            n_clusters = len(unique_labels[unique_labels >= 0])
                            n_outliers = np.sum(labels == -1)
                            n_unclustered = np.sum(labels == -2)
                            
                            st.write("**📈 Clustering Summary:**")
                            st.write(f"- Clusters found: {n_clusters}")
                            st.write(f"- Outliers (noise points): {n_outliers}")
                            if n_unclustered > 0:
                                st.write(f"- Not clustered (missing embedding/error): {n_unclustered}")

                            if n_clusters > 0:
                                st.write("**📝 Articles per Cluster:**")
                                for label in unique_labels:
                                    if label >= 0:
                                        cluster_articles = df_hits[df_hits['cluster_label'] == label]['article_id'].tolist()
                                        st.write(f"  - Cluster {label} ({counts[unique_labels == label][0]} items): {', '.join(cluster_articles)}")

                            # 6. Enhanced Visualization
                            if embeddings_used is not None and embeddings_used.shape[0] > 1:
                                st.write("**🎨 Cluster Visualization (UMAP Projection):**")
                                try:
                                    reducer = umap.UMAP(
                                        n_neighbors=max(2, min(15, embeddings_used.shape[0]-1)),
                                        min_dist=0.1, 
                                        n_components=2, 
                                        random_state=42, 
                                        metric='cosine'
                                    )
                                    embedding_2d = reducer.fit_transform(embeddings_used)

                                    plot_df = pd.DataFrame(embedding_2d, columns=['UMAP_1', 'UMAP_2'])
                                    valid_labels = df_hits.loc[df_hits['cluster_label'] != -2, 'cluster_label'].astype(str)
                                    plot_df['Cluster'] = valid_labels.values
                                    valid_indices = df_hits[df_hits['cluster_label'] != -2].index
                                    plot_df['Article ID'] = df_hits.loc[valid_indices, 'article_id'].values
                                    plot_df['Title'] = df_hits.loc[valid_indices, 'title'].values

                                    fig = px.scatter(
                                        plot_df, x='UMAP_1', y='UMAP_2', color='Cluster',
                                        hover_data=['Article ID', 'Title'],
                                        title="2D UMAP projection of Jina AI embeddings by cluster",
                                        color_discrete_map={"-1": "grey"}
                                    )
                                    fig.update_traces(marker=dict(size=8, opacity=0.8))
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception as viz_e:
                                    st.error(f"Failed to create visualization: {viz_e}")
                                    logging.exception("Error during visualization:")

                        elif df_hits is not None and df_hits.empty:
                            st.warning("Analysis resulted in an empty DataFrame (check logs).")
                        else:
                            st.error("Analysis failed to produce a DataFrame (check logs).")
                else:
                    st.error("Failed to get embedding for the query.")
            except Exception as e:
                st.error(f"An error occurred during the search process: {e}")
                logging.exception("Error during search process:")
    elif search_button and not user_query:
        st.warning("Please enter a search query.")
else:
    st.error("Application cannot start because initialization failed.")

# --- Enhanced Footer ---
st.markdown("---")
st.markdown(f"""
**🚀 Powered by Jina AI v3** | Index: `{INDEX_NAME}` | Model: `{JINA_MODEL_NAME}`  
✨ **16x larger context** • 🎯 **AI reranking** • 🔗 **Hybrid search with RRF** • 📊 **HDBSCAN clustering**
""")

# Performance tips
with st.expander("💡 Performance Tips"):
    st.markdown("""
    - **Enable Reranking**: Get 20-30% better precision with minimal performance impact
    - **Increase Rerank Candidates**: More candidates = better quality, but slower
    - **Use A/B Comparison**: Compare baseline vs reranked results side-by-side
    - **Adjust Boosts**: Fine-tune text vs semantic similarity balance
    - **Context Window**: Jina AI now understands 16x more content per document!
    """)

