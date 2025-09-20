import os
import sys
import json
from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch, RequestsHttpConnection, OpenSearchException
from opensearchpy.helpers import bulk

# --- Configurations ---
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
LOCAL_MODEL_PATH = "all-MiniLM-L6-v2"

# --- OpenSearch Connection Details ---
HOST = 'localhost'
PORT = 9200
OPENSEARCH_PASSWORD = "OpenSearch!2025"
AUTH = ('admin', OPENSEARCH_PASSWORD)
INDEX_NAME = 'products'

def main():
    """
    Main function to connect to OpenSearch, process data from multiple ndjson
    formats, generate combined embeddings, and index the unified documents.
    """
    # --- Initialize Models and Clients ---
    print("Loading local embedding model...")
    try:
        embedding_model = SentenceTransformer(LOCAL_MODEL_PATH)
        print("Embedding model loaded successfully.")
    except Exception as e:
        print(f"Fatal Error loading SentenceTransformer model: {e}", file=sys.stderr)
        sys.exit(1)

    print("Connecting to OpenSearch...")
    client = initialize_opensearch_client()

    # --- Process and Consolidate Data ---
    print("Processing and consolidating data from all files...")
    products_data = process_data_files('output_opensearch_bulk')
    
    # --- Generate Embeddings and Prepare for Bulk Indexing ---
    print("Generating combined embeddings for each product...")
    bulk_data = []
    for part_number, data in products_data.items():
        # Combine all relevant text into one string for a single, powerful embedding
        combined_text = " ".join(filter(None, [
            data.get('productDescription', ''),
            " ".join(f"{f.get('feature', '')} {f.get('comment', '')}" for f in data.get('features', [])),
            " ".join(f"{d.get('designBlocks', '')} {d.get('description', '')}" for d in data.get('designIPs', []))
        ]))

        # Generate the single embedding
        doc_embedding = embedding_model.encode(combined_text).tolist()

        # Create the final, unified document for OpenSearch
        final_doc = {
            **data,
            'doc_embedding': doc_embedding
        }
        
        # Add to bulk data list
        bulk_data.append({
            "_index": INDEX_NAME,
            "_id": part_number,
            "_source": final_doc
        })

    # --- Create Index and Ingest Data ---
    setup_index(client)
    
    print(f"Indexing {len(bulk_data)} consolidated product documents...")
    try:
        success, failed = bulk(client, bulk_data, refresh=True)
        print(f"Successfully indexed {success} documents.")
        if failed:
            print(f"Failed to index {len(failed)} documents.", file=sys.stderr)
    except OpenSearchException as e:
        print(f"An error occurred during bulk indexing: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nData indexing complete!")

def initialize_opensearch_client():
    """Initializes and returns the OpenSearch client."""
    try:
        client = OpenSearch(
            hosts=[{'host': HOST, 'port': PORT}],
            http_auth=AUTH,
            use_ssl=True,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
            connection_class=RequestsHttpConnection
        )
        if not client.ping():
            raise ConnectionError("Could not connect to OpenSearch.")
        return client
    except Exception as e:
        print(f"Error connecting to OpenSearch: {e}", file=sys.stderr)
        sys.exit(1)

def process_data_files(data_folder):
    """
    Reads all .ndjson files, handles different schemas, and groups data by part number.
    """
    products = {}
    for filename in os.listdir(data_folder):
        if filename.endswith(".ndjson"):
            filepath = os.path.join(data_folder, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                # Try loading as a single, large JSON object first (like CDQ23815LX)
                try:
                    f.seek(0)
                    single_json_doc = json.load(f)
                    if 'productInfo' in single_json_doc:
                        part_number = single_json_doc['productInfo']['partNumber']
                        products[part_number] = {**single_json_doc['productInfo'], **single_json_doc}
                        del products[part_number]['productInfo'] # Flatten structure
                    continue # Move to next file
                except json.JSONDecodeError:
                    # If it fails, process as a line-by-line ndjson file
                    f.seek(0)

                for line in f:
                    if line.strip().startswith('{"index":'):
                        continue
                    
                    doc = json.loads(line)
                    part_number = doc.get('partNumber')
                    if not part_number:
                        continue
                    
                    if part_number not in products:
                        products[part_number] = {'features': [], 'designIPs': []}

                    doc_type = doc.get('type') or doc.get('docType')
                    if doc_type in ['productInfo', 'productInfoEmbedding']:
                        products[part_number].update(doc)
                    elif doc_type in ['feature', 'featureEmbedding']:
                        products[part_number]['features'].append(doc)
                    elif doc_type in ['designIP', 'designIPEmbedding']:
                        products[part_number]['designIPs'].append(doc)
    return products


def setup_index(client):
    """Deletes the old index and creates a new one with the simplified vector mapping."""
    index_mapping = {
        "settings": {"index.knn": True},
        "mappings": {
            "properties": {
                "doc_embedding": {
                    "type": "knn_vector",
                    "dimension": 384,
                    "method": {"name": "hnsw", "space_type": "l2", "engine": "lucene"}
                }
            }
        }
    }
    try:
        if client.indices.exists(index=INDEX_NAME):
            print(f"Deleting existing index '{INDEX_NAME}'...")
            client.indices.delete(index=INDEX_NAME)
        
        print(f"Creating new index '{INDEX_NAME}' with a unified vector field...")
        client.indices.create(index=INDEX_NAME, body=index_mapping)
    except OpenSearchException as e:
        print(f"Error setting up index: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()

