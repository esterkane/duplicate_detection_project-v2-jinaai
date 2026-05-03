import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import ast
import re  # Keep re for potential use in summary parsing
import logging  # Make sure logging is imported
from typing import List, Dict, Any, Tuple  # <<< Import Tuple >>>
from .deduplication import cluster_embeddings_hdbscan  # Fixed: import correct function name
from .config import KB_BASE_URL, EMBEDDING_FIELD  # Ensure import is present

# --- DEFINE YOUR ACTUAL BASE URL HERE ---
# KB_BASE_URL is now imported from config
# ---

# --- DEFINE TARGET PRODUCT AREAS ---
TARGET_PRODUCT_AREAS = [
    "APM", "App Search", "AutoOps Monitoring", "Beats", "Elastic Agent & Fleet",
    "Elastic Cloud Platform (ECE)", "Elastic Cloud Platform (ECK)",
    "Elastic Cloud Platform (Hosted Deployment / ESS)", "Elastic Cloud Platform (Serverless Project)",
    "Elastic Distributions of OpenTelemetry", "Elastic Endgame", "Elastic Integration for AWS Firehose",
    "Elastic Managed Ingest Service", "Elasticsearch",
    "Enterprise Search",  # <<< ADD ENTERPRISE SEARCH
    "Elastic Serverless Forwarder",
    "Elastic Universal Profiling", "Kibana", "Logstash", "Machine Learning", "Non-Product",
    "Observability AI Assistant", "Observability Alerts, Rules & Cases",
    "Observability Service-level Objectives", "Observability UI", "OSQuery Management UI",
    "Partner Product", "Security Alerting & Rules", "Security UI", "Site Search", "Synthetics",
    "Usage/Metering (Serverless Project - Elasticsearch)",
    "Usage/Metering (Serverless Project - Observability)",
    "Usage/Metering (Serverless Project - Security)",
    "Other",  # Add a default category
    "Unknown"  # Add category for parsing errors or empty lists
]
# ---

# --- Helper function _safe_literal_eval ---
def _safe_literal_eval(s):
    """Safely evaluates a string that looks like a list, falling back to splitting."""
    if not isinstance(s, str):
        return []  # Return empty list if not a string
    s_cleaned = s.strip()
    try:
        # Check if it looks like a list representation
        if s_cleaned.startswith('[') and s_cleaned.endswith(']'):
            evaluated = ast.literal_eval(s_cleaned)
            if isinstance(evaluated, list):
                # Ensure all items are strings and strip quotes/spaces
                return [str(item).strip().strip('"') for item in evaluated if item]
        # Fallback: If not a list string or eval fails, try splitting by comma
        return [item.strip().strip('"') for item in s_cleaned.split(',') if item.strip()]
    except (ValueError, SyntaxError, TypeError):
        # Final fallback: Treat the whole string as a single item list if non-empty
        return [s.strip().strip('"')] if s.strip() else []


# --- Product Area Mapping Function map_to_product_area ---
def map_to_product_area(product_list_input, title: str, summary: str) -> str:
    """
    Maps products to a target product area. Prioritizes combining all areas
    found in metadata_products before falling back to title/summary keywords
    or 'Unknown'.
    """
    products = []
    # --- Handle list or string input ---
    if isinstance(product_list_input, list):
        products = product_list_input
    elif isinstance(product_list_input, str):
        products = _safe_literal_eval(product_list_input)

    title_lower = title.lower() if title else ""
    summary_lower = summary.lower() if summary else ""
    default_area = "Unknown"  # Default if nothing else matches

    # --- PRIORITY CHECK: Enterprise Search in Title/Summary ---
    # This check remains as it's a specific override based on title/summary
    ent_search_mapped = None
    ent_search_terms = ["enterprise search", "app search", "site search", "workplace search"]
    if any(term in title_lower for term in ent_search_terms) or \
       any(term in summary_lower for term in ent_search_terms):
        if "app search" in title_lower or "app search" in summary_lower:
            ent_search_mapped = "App Search"
        elif "site search" in title_lower or "site search" in summary_lower:
            ent_search_mapped = "Site Search"
        else:
            ent_search_mapped = "Enterprise Search"
        # If Ent Search is mapped, return it immediately
        return ent_search_mapped

    # --- Stage 1: Map ALL products from metadata_products ---
    metadata_mapped_areas = set()
    if products:
        for product_name in products:
            product_lower = product_name.lower()
            area = None
            # --- Mapping Logic for each product ---
            if "apm" in product_lower: area = "APM"
            elif "app search" in product_lower: area = "App Search"
            elif "beats" in product_lower or product_lower in ["filebeat", "metricbeat", "winlogbeat", "heartbeat", "packetbeat", "auditbeat", "journalbeat", "functionbeat"]: area = "Beats"
            elif "elastic agent" in product_lower or "fleet" in product_lower: area = "Elastic Agent & Fleet"
            elif "ece" in product_lower: area = "Elastic Cloud Platform (ECE)"
            elif "eck" in product_lower: area = "Elastic Cloud Platform (ECK)"
            elif "ess" in product_lower or "elastic cloud" in product_lower or "hosted" in product_lower: area = "Elastic Cloud Platform (Hosted Deployment / ESS)"
            elif "opentelemetry" in product_lower: area = "Elastic Distributions of OpenTelemetry"
            elif "endgame" in product_lower: area = "Elastic Endgame"
            elif "firehose" in product_lower: area = "Elastic Integration for AWS Firehose"
            elif "elasticsearch" in product_lower: area = "Elasticsearch"
            elif "enterprise search" in product_lower: area = "Enterprise Search"
            elif "universal profiling" in product_lower: area = "Elastic Universal Profiling"
            elif "kibana" in product_lower: area = "Kibana"
            elif "logstash" in product_lower: area = "Logstash"
            elif "machine learning" in product_lower or "ml" == product_lower: area = "Machine Learning"
            elif "observability" in product_lower: area = "Observability UI"
            elif "osquery" in product_lower: area = "OSQuery Management UI"
            elif "security" in product_lower: area = "Security UI"
            elif "site search" in product_lower: area = "Site Search"
            elif "synthetics" in product_lower: area = "Synthetics"
            elif "non-product" in product_lower: area = "Non-Product"
            # --- End Mapping Logic ---
            if area:
                metadata_mapped_areas.add(area)

    # --- Stage 2: Return combined metadata areas if found ---

    if metadata_mapped_areas:
        return ", ".join(sorted(list(metadata_mapped_areas)))

    # --- Stage 3: Fallback to summary keywords ONLY if metadata was empty ---
    if summary_lower:
        # --- Mapping Logic based on summary keywords ---
        if "elastic agent" in summary_lower or "fleet" in summary_lower: return "Elastic Agent & Fleet"
        if "app search" in summary_lower: return "App Search"
        if "site search" in summary_lower: return "Site Search"
        if "filebeat" in summary_lower or "metricbeat" in summary_lower or "winlogbeat" in summary_lower or "heartbeat" in summary_lower or "packetbeat" in summary_lower or "auditbeat" in summary_lower: return "Beats"
        if "logstash" in summary_lower: return "Logstash"
        if "apm" in summary_lower: return "APM"
        if "kibana" in summary_lower: return "Kibana"
        if "elasticsearch" in summary_lower: return "Elasticsearch"  # Ent Search already handled
        if "machine learning" in summary_lower or " ml " in summary_lower: return "Machine Learning"
        if "observability" in summary_lower: return "Observability UI"
        if "security" in summary_lower: return "Security UI"
        if "synthetics" in summary_lower: return "Synthetics"
        if "endgame" in summary_lower: return "Elastic Endgame"
        if " ece" in summary_lower: return "Elastic Cloud Platform (ECE)"
        if " eck" in summary_lower: return "Elastic Cloud Platform (ECK)"
        if "elastic cloud" in summary_lower: return "Elastic Cloud Platform (Hosted Deployment / ESS)"  # Ent Search already handled

    # --- Final Stage: Return default if nothing else matched ---
    return default_area


# --- Main Analysis Function ---
def analyze_knn_hits(hits: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    """
    Processes Elasticsearch hits, extracts fields, maps product areas,
    constructs URLs, performs clustering on embeddings, and returns results.

    Returns:
        A tuple containing:
        - pd.DataFrame: Processed hit data with cluster labels.
        - np.ndarray: Array of embeddings used for clustering (or None).
        - Dict: Parameters used for clustering.
    """
    processed_hits = []
    embeddings_list = []
    logging.info(f"Analyzing {len(hits)} hits received from search.")

    # --- Loop through hits (extract data and embeddings) ---
    for i, hit in enumerate(hits):
        source = hit.get('_source', {})
        score = hit.get('_score')
        article_id = source.get('article_id', None)
        title = source.get('content_title', '')
        summary = source.get('content_summary', '')
        content_body = source.get('content_body', '')  # Extract content body
        metadata_products_input = source.get('metadata_products')

        # <<< SIMPLIFY EMBEDDING EXTRACTION >>>
        embedding = source.get(EMBEDDING_FIELD)  # Get directly from _source

        # Check if embedding is valid
        if embedding and isinstance(embedding, list):
            embeddings_list.append(embedding)
        else:
            embeddings_list.append(None)
            logging.warning(f"Hit {i} (ID: {article_id}) missing or has invalid embedding. Value from source: {embedding}.")  # Simplified warning

        metadata_products_list = []
        if isinstance(metadata_products_input, list):
            metadata_products_list = [str(item).strip().strip('"') for item in metadata_products_input if item]
        elif isinstance(metadata_products_input, str):
            metadata_products_list = _safe_literal_eval(metadata_products_input)
        product_area = map_to_product_area(metadata_products_list, title, summary)
        url = f"{KB_BASE_URL}{article_id}" if article_id else None
        # Store processed data
        hit_data = {
            'article_id': article_id, 'score': score, 'product_area': product_area,
            'title': title, 'summary': summary, 'content_body': content_body,  # Add content_body here
            'url': url
        }
        processed_hits.append(hit_data)
    # --- End Loop ---

    logging.info(f"Finished processing hits. Created {len(processed_hits)} records.")

    # --- Perform Clustering ---
    cluster_labels = np.array([-2] * len(processed_hits))  # Initialize with -2 (unprocessed)
    valid_embeddings_indices = [idx for idx, emb in enumerate(embeddings_list) if emb is not None]
    valid_embeddings = [embeddings_list[idx] for idx in valid_embeddings_indices]
    embeddings_array = None

    # <<< Adjust clustering parameters MORE AGGRESSIVELY >>>
    num_valid = len(valid_embeddings)
    # Try smaller min_cluster_size for fewer points. e.g., 2 if < 10 points, 3 if < 15, etc.
    if num_valid < 4:
        hdbscan_min_cluster_size = 2 # Need at least 2 for a cluster
    elif num_valid < 10:
        hdbscan_min_cluster_size = 2 # More aggressive for small k
    elif num_valid < 15:
        hdbscan_min_cluster_size = 3
    else:
        hdbscan_min_cluster_size = 5 # Default for larger k

    hdbscan_min_samples = 1  # Keep min_samples low for small datasets
    # <<< TRY A NON-ZERO EPSILON >>>
    hdbscan_epsilon = 0.1 # Default: 0.0. Try small values like 0.05 or 0.1 if needed.

    clustering_params = {
        'method': 'HDBSCAN',
        'min_cluster_size': hdbscan_min_cluster_size,
        'min_samples': hdbscan_min_samples,
        'metric': 'cosine',
        'cluster_selection_epsilon': hdbscan_epsilon # Pass the epsilon value
    }
    # <<< End parameter adjustment >>>

    # 3. Perform clustering if we have valid embeddings
    if valid_embeddings and len(valid_embeddings) > 1:
        embeddings_array = np.array(valid_embeddings)
        logging.info(f"Attempting to cluster {embeddings_array.shape[0]} valid embeddings using HDBSCAN with min_cluster_size={hdbscan_min_cluster_size}, min_samples={hdbscan_min_samples}, epsilon={hdbscan_epsilon}.")
        
        # Use the correct function name and pass parameters properly
        valid_cluster_labels = cluster_embeddings_hdbscan(
            embeddings_array, 
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
            cluster_selection_epsilon=hdbscan_epsilon
        )
        
        # Map the cluster labels back to the original indices
        for i, valid_idx in enumerate(valid_embeddings_indices):
            cluster_labels[valid_idx] = valid_cluster_labels[i]
        
        logging.info(f"Clustering complete. Found labels: {np.unique(valid_cluster_labels)}")
    else:
        logging.warning("Not enough valid embeddings found to perform clustering.")
        clustering_params['status'] = 'Skipped: Not enough valid embeddings.'
        embeddings_array = None

    # --- Add Cluster Labels to Processed Hits ---
    for i, hit_data in enumerate(processed_hits):
        hit_data['cluster_label'] = cluster_labels[i]

    # --- Create DataFrame ---
    expected_columns = ['article_id', 'score', 'cluster_label', 'product_area', 'title', 'summary', 'content_body', 'url']
    df = pd.DataFrame(processed_hits)
    for col in expected_columns:
        if col not in df.columns:
            df[col] = None
    df = df[expected_columns]

    logging.debug(f"Created DataFrame with columns: {df.columns.tolist()}")
    return df, embeddings_array, clustering_params
