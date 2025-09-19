import os
import sys
from opensearchpy import OpenSearch, RequestsHttpConnection, OpenSearchException

# --- Connection Details ---
host = 'localhost'
port = 9200
OPENSEARCH_PASSWORD = "OpenSearch!2025" # The password you set in docker-compose.yml
auth = ('admin', OPENSEARCH_PASSWORD)
index_name = 'products'

# --- Create a client instance ---
try:
    client = OpenSearch(
        hosts=[{'host': host, 'port': port}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        connection_class=RequestsHttpConnection
    )
    if not client.ping():
        raise ConnectionError("Could not connect to OpenSearch.")
except Exception as e:
    print(f"Error connecting to OpenSearch: {e}", file=sys.stderr)
    print("Please ensure your OpenSearch cluster is running and accessible.", file=sys.stderr)
    sys.exit(1)

# --- Index Mapping for Vector Search ---
# This mapping tells OpenSearch how to handle the data, specifically identifying
# the fields that contain vector embeddings for k-NN search.
# IMPORTANT: The 'dimension' must match the size of your actual embeddings.
# Common models produce vectors of size 384, 768, or 1536. Please verify this value.
index_mapping = {
    "settings": {
        "index.knn": True
    },
    "mappings": {
        "properties": {
            "productDescriptionEmbedding": {
                "type": "knn_vector",
                "dimension": 384,
                "method": {
                    "name": "hnsw",
                    "space_type": "l2",
                    "engine": "lucene" # MODIFIED: Changed deprecated 'nmslib' to 'lucene'
                }
            },
            "featureEmbedding": {
                "type": "knn_vector",
                "dimension": 384,
                "method": {
                    "name": "hnsw",
                    "space_type": "l2",
                    "engine": "lucene" # MODIFIED: Changed deprecated 'nmslib' to 'lucene'
                }
            },
            "descriptionEmbedding": {
                "type": "knn_vector",
                "dimension": 384,
                "method": {
                    "name": "hnsw",
                    "space_type": "l2",
                    "engine": "lucene" # MODIFIED: Changed deprecated 'nmslib' to 'lucene'
                }
            }
        }
    }
}

# --- Indexing ---
try:
    # Delete the index if it exists, to apply the new mapping
    if client.indices.exists(index=index_name):
        print(f"Deleting existing index '{index_name}' to apply new mapping...")
        client.indices.delete(index=index_name)
    
    # Create the index with the new mapping
    print(f"Creating index '{index_name}' with vector search mapping...")
    client.indices.create(index=index_name, body=index_mapping)

    # Ingest the data from ndjson files
    data_folder = 'output_opensearch_bulk'
    for filename in os.listdir(data_folder):
        if filename.endswith(".ndjson"):
            filepath = os.path.join(data_folder, filename)
            print(f"Indexing data from {filename}...")
            with open(filepath, 'r') as f:
                data = f.read()
                client.bulk(body=data, index=index_name, refresh=True)

    print("\nData indexed successfully with vector support!")

except OpenSearchException as e:
    print(f"An error occurred during indexing: {e}", file=sys.stderr)
    sys.exit(1)

