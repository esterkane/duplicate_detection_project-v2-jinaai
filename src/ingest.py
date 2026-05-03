# src/ingest.py

import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, BulkIndexError
import time
import sys
import os
import ast
import re

# Add the project directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embeddings_jina import load_jina_embedding_model, compute_jina_embeddings, combine_text_with_metadata
from src.config import ES_MODEL_ID, INDEX_NAME, JINA_DIMENSIONS, JINA_MODEL_NAME, JINA_TASK, JINA_MAX_LENGTH
from src.es_client import get_es_client


def create_index(
    es: Elasticsearch,
    name: str,
    dims: int,
    mapping: dict = None
):
    """
    Create an Elasticsearch index for KB articles with metadata,
    an 'article_id' field, and a dense_vector field named 'embedding'.
    Includes additional metadata fields.
    """
    props = mapping or {
        # Existing fields
        "article_id": {"type": "keyword"},
        "content_title": {"type": "text"},
        "content_summary": {"type": "text"},
        "content_body": {"type": "text"},
        "category": {"type": "keyword"},
        "metadata_products": {"type": "keyword"},
        "embedding": {
            "type": "dense_vector",
            "dims": dims,
            "index": True,
            "similarity": "cosine"
        },
        "duplicate_cluster_id": {"type": "keyword"},

        # Newly added metadata fields
        "metadata_product_versions": {"type": "keyword"}, # Assuming list-like strings, map as keyword
        "metadata_introduced_version": {"type": "version"}, # Use 'version' type if format is consistent
        "metadata_fixed_version": {"type": "version"},      # Use 'version' type if format is consistent
        "metadata_components": {"type": "keyword"},       # Assuming list-like strings
        "metadata_platforms": {"type": "keyword"},        # Assuming list-like strings
        "metadata_solutions": {"type": "keyword"},        # Assuming list-like strings
        "metadata_deployments": {"type": "keyword"},       # Assuming list-like strings
        "metadata_deployment_versions": {"type": "keyword"} # Assuming list-like strings
    }
    body = {"mappings": {"properties": props}}

    if es.indices.exists(index=name):
        print(f"Index '{name}' already exists. Attempting to update mapping...")
        try:
            # Try to update mapping (might fail for incompatible changes)
            es.indices.put_mapping(index=name, properties=props)
            print(f"✅ Index '{name}' mapping updated.")
        except Exception as e:
            print(f"⚠️ Could not update mapping for index '{name}'. Error: {e}")
            print("Proceeding with existing mapping. New fields might not be indexed correctly unless index is recreated.")
    else:
        es.indices.create(index=name, body=body)
        print(f"✅ Index '{name}' created with full metadata mapping.")


def clean_text_field(raw_value):
    """
    Helper function to clean text fields, potentially handling simple stringified lists.
    """
    if pd.isna(raw_value):
        return ""
    text = str(raw_value).strip().strip('"')

    if text.startswith('["') and text.endswith('"]'):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                if len(parsed) == 1:
                    return str(parsed[0]).strip().strip('"')
                elif len(parsed) == 0:
                    return ""
                else:
                    return text
        except (ValueError, SyntaxError):
            return text
    if text == '[]':
        return ''
    text = text.replace('""', '"')
    return text.strip('"')


def clean_metadata_products(raw_value):
    """
    Helper function specifically for metadata_products (expecting list of strings).
    """
    if pd.isna(raw_value):
        return []
    text = str(raw_value).strip().strip('"')
    products = []

    if text.startswith('[') and text.endswith(']'):
        try:
            cleaned_for_eval = text.replace('""', '"')
            parsed = ast.literal_eval(cleaned_for_eval)

            if isinstance(parsed, list):
                products = [str(item).strip().strip('"') for item in parsed if str(item).strip()]
                products = [p for p in products if p]
                return products
            else:
                if text != '[]':
                    products = [text.strip().strip('"')]

        except (ValueError, SyntaxError):
            cleaned_text = text.strip().strip('"')
            if cleaned_text and cleaned_text != '[]':
                products = [cleaned_text]

    elif text:
        products = [text.strip().strip('"')]

    return [p for p in products if p]


def ingest_data_to_es(
    es: Elasticsearch,
    df: pd.DataFrame,
    index_name: str,
    vector_field: str = "embedding",
    chunk_size: int = 500,
    max_retries: int = 3
):
    """
    Bulk-index a DataFrame, storing original ID in 'article_id',
    including all metadata fields, and letting ES generate _id.
    """
    actions = []
    skipped_count = 0
    processed_count = 0
    # Check for essential columns
    required_columns = ["id", vector_field]
    if not all(col in df.columns for col in required_columns):
        # Check if the vector field name might be the issue
        if vector_field == "embedding" and "embeddings" in df.columns:
            print(f"Note: Found 'embeddings' column, using it as vector field instead of default '{vector_field}'.")
            vector_field = "embeddings"  # Auto-correct common typo
            required_columns = ["id", vector_field]
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"DataFrame must contain columns: {required_columns}")
        else:
            raise ValueError(f"DataFrame must contain columns: {required_columns}")

    # Define columns that contain list-like strings and need list cleaning
    list_like_cols = [
        "metadata_products", "metadata_product_versions", "metadata_components",
        "metadata_platforms", "metadata_solutions", "metadata_deployments",
        "metadata_deployment_versions"
    ]
    # Define columns that are single text/version fields
    single_text_cols = [
        "content_title", "content_summary", "content_body", "category",
        "metadata_introduced_version", "metadata_fixed_version"
    ]

    for _, row in df.iterrows():
        try:
            processed_count += 1
            original_id = str(row.get("id", "")).strip().strip('"')
            if not original_id:
                skipped_count += 1
                continue

            # Prepare the _source document
            doc_source = {
                "article_id": original_id,
                # Add vector field
                vector_field: row[vector_field].tolist() if hasattr(row[vector_field], 'tolist') else list(row[vector_field])
            }

            # Add single text fields using clean_text_field
            for col in single_text_cols:
                if col in row:  # Check if column exists in the row/DataFrame
                    doc_source[col] = clean_text_field(row.get(col, ""))

            # Add list-like fields using clean_metadata_products
            for col in list_like_cols:
                if col in row:  # Check if column exists in the row/DataFrame
                    doc_source[col] = clean_metadata_products(row.get(col, ""))

            # Add optional duplicate_cluster_id if present
            if "duplicate_cluster_id" in row and pd.notna(row["duplicate_cluster_id"]):
                try:
                    doc_source["duplicate_cluster_id"] = str(int(row["duplicate_cluster_id"]))
                except ValueError:
                    print(f"⚠️ Could not convert duplicate_cluster_id '{row['duplicate_cluster_id']}' to int for original ID {original_id}. Skipping field.")

            actions.append({
                "_op_type": "index",
                "_index": index_name,
                "_source": doc_source
            })

        except Exception as row_error:
            print(f"🚨 Error processing row {processed_count} (Original ID: {row.get('id', 'N/A')}): {row_error}")
            skipped_count += 1
            continue

    if skipped_count > 0:
        print(f"⚠️ Skipped {skipped_count} rows during processing.")

    if not actions:
        print("No documents to index.")
        return

    print(f"Indexing {len(actions)} docs with precomputed embeddings in batches of {chunk_size}…")
    for i in range(0, len(actions), chunk_size):
        batch = actions[i: i + chunk_size]
        for attempt in range(1, max_retries + 1):
            try:
                response = bulk(es, batch)
                errors_in_response = False
                if response and isinstance(response, tuple) and len(response) > 1:
                    for item in response[1]:
                        op_type = list(item.keys())[0]
                        if item[op_type].get('status', 200) >= 300:
                            error_detail = item[op_type].get('error', {})
                            print(f"❌ Error in bulk response item (ID: {item[op_type].get('_id')}): Status={item[op_type].get('status')}, Type={error_detail.get('type')}, Reason={error_detail.get('reason')}")
                            errors_in_response = True

                if not errors_in_response:
                    print(f"✅ Batch {i}-{i + len(batch)} indexed (or processed).")
                    break
                else:
                    print(f"⚠️ Batch {i}-{i + len(batch)} attempt {attempt} had item errors.")
                    if attempt == max_retries:
                        print(f"🚨 Giving up on batch {i}-{i + len(batch)} after {max_retries} attempts due to item errors.")
                    else:
                        time.sleep(2 * attempt)

            except BulkIndexError as e:
                print(f"❌ BulkIndexError on batch {i}-{i + len(batch)} attempt {attempt} ({len(e.errors)} errors).")
                print("First few errors:")
                for idx, error_info in enumerate(e.errors[:5]):
                    item_error = error_info.get('index', {}).get('error', {})
                    print(f"  Error {idx + 1}: Status={error_info.get('index', {}).get('status')}, Type={item_error.get('type')}, Reason={item_error.get('reason')}")
                if attempt < max_retries:
                    time.sleep(2 * attempt)
                else:
                    print(f"🚨 Giving up on batch {i}-{i + len(batch)} after {max_retries} attempts due to BulkIndexError.")
            except Exception as e:
                print(f"⚠️ Unexpected error on batch {i}-{i + len(batch)} attempt {attempt}: {e}")
                if attempt < max_retries:
                    time.sleep(5)
                else:
                    print(f"🚨 Giving up on batch {i}-{i + len(batch)} after {max_retries} attempts due to unexpected error.")


def create_index_with_embeddings(es: Elasticsearch, name: str, dims: int):
    """
    Create an Elasticsearch index for KB articles with embeddings field.
    """
    return create_index(es, name, dims)

def ingest_df_to_es(es: Elasticsearch, df: pd.DataFrame, index_name: str):
    """
    Wrapper for ingest_data_to_es with correct vector field name.
    """
    return ingest_data_to_es(es, df, index_name, vector_field="embeddings")


def main():
    """
    Main function to load data, generate embeddings, create index, and ingest data
    by processing the source CSV in chunks.
    """
    # Connect to Elasticsearch
    es_client = get_es_client()

    # Define parameters - UPDATED FOR JINA AI MIGRATION
    index_name = INDEX_NAME  # Now uses "kb_articles_metadata_jina_v3"
    csv_file = os.getenv("SOURCE_CSV_PATH", "data/input.csv")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(
            f"Source CSV not found: {csv_file}. Set SOURCE_CSV_PATH or place data at data/input.csv."
        )
    dims = JINA_DIMENSIONS  # 1024 dimensions for Jina AI
    
    chunk_count = 0
    total_ingested = 0
    
    # Load Jina AI model once for all chunks
    jina_model = load_jina_embedding_model()
    print(f"✅ Jina AI model loaded: {JINA_MODEL_NAME}")
    
    # Reduce CSV chunk size for memory management
    csv_chunk_size = 200  # Reduced from 500

    for df_chunk in pd.read_csv(csv_file, chunksize=csv_chunk_size):
        chunk_count += 1
        print(f"\n--- Processing Chunk {chunk_count} (rows {total_ingested + 1} to {total_ingested + len(df_chunk)}) ---")
        
        # Clean and prepare data
        df_chunk = df_chunk.dropna(subset=['content_title', 'content_summary'])
        df_chunk = df_chunk.reset_index(drop=True)
        
        if df_chunk.empty:
            print(f"Chunk {chunk_count} is empty after cleaning, skipping...")
            continue

        # Step 3: Generate embeddings using Jina AI with smaller batches
        print(f"Generating Jina AI embeddings for {len(df_chunk)} documents...")
        df_chunk["text_for_embedding"] = df_chunk.apply(combine_text_with_metadata, axis=1)
        
        # Use smaller batch size for memory efficiency
        texts = df_chunk["text_for_embedding"].tolist()
        embeddings_array = compute_jina_embeddings(
            model=jina_model,
            texts=texts,
            task=JINA_TASK,
            batch_size=8,  # Reduced from 32 to avoid memory issues
            normalize=True
        )
        
        # Clear any cached tensors to free memory
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # Add embeddings to dataframe
        df_chunk["embeddings"] = embeddings_array.tolist()
        print(f"✅ Generated {len(embeddings_array)} embeddings with shape {embeddings_array.shape}")

        # Step 4: Ingest the chunk into Elasticsearch
        try:
            ingest_df_to_es(es_client, df_chunk, index_name)
            total_ingested += len(df_chunk)
            print(f"✅ Successfully ingested chunk {chunk_count}. Total documents: {total_ingested}")
        except Exception as e:
            print(f"❌ Failed to ingest chunk {chunk_count}: {e}")
            break

    print(f"\n🎉 Migration complete! Total documents ingested: {total_ingested}")
    print(f"📊 New index: {index_name}")
    print(f"🤖 Using Jina AI model: {JINA_MODEL_NAME}")
    print(f"📈 Context length: {JINA_MAX_LENGTH} tokens (16x improvement!)")
    
    # Verify the index
    try:
        count = es_client.count(index=index_name)
        print(f"✅ Index verification: {count['count']} documents in {index_name}")
    except Exception as e:
        print(f"⚠️ Could not verify index: {e}")

if __name__ == "__main__":
    main()
