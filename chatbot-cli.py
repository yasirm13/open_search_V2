import os
import sys
from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch, RequestsHttpConnection, OpenSearchException
import google.generativeai as genai
import json
from dotenv import load_dotenv # Import the dotenv library

# --- Configurations ---
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# --- LLM and Embedding Model Setup ---
# The API key will be loaded from the environment, not hardcoded.
LOCAL_MODEL_PATH = "all-MiniLM-L6-v2"

# --- OpenSearch Connection Details ---
HOST = 'localhost'
PORT = 9200
OPENSEARCH_PASSWORD = "OpenSearch!2025"
AUTH = ('admin', OPENSEARCH_PASSWORD)
INDEX_NAME = 'products'


def initialize_models():
    """Initializes and returns the Gemini and SentenceTransformer models."""
    print("Initializing Gemini model...")
    try:
        # Load environment variables from the .env file
        load_dotenv()
        # Get the API key from the environment
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            print("Fatal Error: GEMINI_API_KEY not found in the .env file.", file=sys.stderr)
            print("Please create a '.env' file and add your key (e.g., GEMINI_API_KEY='YOUR_KEY').", file=sys.stderr)
            sys.exit(1)

        genai.configure(api_key=gemini_api_key)
        llm_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        print("Gemini model initialized.")
    except Exception as e:
        print(f"Fatal Error: Could not configure Gemini model. Check your API key. Error: {e}", file=sys.stderr)
        sys.exit(1)

    print("Loading local embedding model (this may take a moment)...")
    try:
        if not os.path.isdir(LOCAL_MODEL_PATH):
            print(f"Error: The local model folder '{LOCAL_MODEL_PATH}' was not found.", file=sys.stderr)
            sys.exit(1)
        embedding_model = SentenceTransformer(LOCAL_MODEL_PATH)
        print("Embedding model loaded.")
    except Exception as e:
        print(f"Fatal Error: Could not load the SentenceTransformer model from '{LOCAL_MODEL_PATH}'. Error: {e}",
              file=sys.stderr)
        sys.exit(1)

    return llm_model, embedding_model


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


def _print_item(item, indent="  "):
    """Recursively prints nested dictionaries and lists for clear display."""
    for key, value in item.items():
        if 'embedding' in key.lower():  # Skip all embedding fields
            continue
        if value is None or value == '':
            continue

        if isinstance(value, dict):
            print(f"{indent}{key}:")
            _print_item(value, indent + "  ")
        elif isinstance(value, list):
            print(f"{indent}{key}:")
            for i, list_item in enumerate(value):
                if isinstance(list_item, dict):
                    print(f"{indent}  - Item {i+1}:")
                    _print_item(list_item, indent + "    ")
                else:
                    print(f"{indent}  - {list_item}")
        else:
            print(f"{indent}{key}: {value}")


def format_and_print_results(search_results):
    """Formats and prints the search results for clarity."""
    print("\n--- Data Retrieved from Database ---")
    if not search_results:
        print("No relevant documents found.")
        return

    for i, hit in enumerate(search_results, 1):
        print(f"\n[Result {i}] (Score: {hit['_score']:.2f})")
        _print_item(hit['_source'])
    print("------------------------------------\n")


def ask_chatbot(question, client, llm_model, embedding_model):
    """Handles chatbot logic: embedding, hybrid search, and answer generation."""
    try:
        print("Generating embedding for your question...")
        question_embedding = embedding_model.encode(question)

        # UPDATED: Simplified hybrid query with a single, powerful vector search
        search_body = {
            "query": {
                "hybrid": {
                    "queries": [
                        {
                            "multi_match": {
                                "query": question,
                                "fields": ["partNumber", "productDescription", "features.feature", "designIPs.designBlocks", "designIPs.description"]
                            }
                        },
                        {
                            "knn": {
                                "doc_embedding": {
                                    "vector": question_embedding.tolist(),
                                    "k": 5
                                }
                            }
                        }
                    ]
                }
            },
            "size": 5
        }

        print("Searching database...")
        response = client.search(index=INDEX_NAME, body=search_body)
        
        # Pass the full hits with scores to the display function
        search_hits = response['hits']['hits']
        format_and_print_results(search_hits)
        
        search_results = [hit['_source'] for hit in search_hits]

        if not search_results:
            return "I could not find any relevant information in the database to answer your question."

        prompt = f"""
        You are an expert engineering assistant. Your task is to answer the user's question based *only* on the provided search results from a product database.
        Synthesize the information from all the provided documents to form a comprehensive and accurate answer.
        - If the results contain a clear answer, provide it directly.
        - If the results contain multiple relevant parts, combine the information logically.
        - Always mention the specific 'partNumber' or 'designBlocks' you are referring to.
        - If the provided search results do not contain enough information to answer the question, state that clearly. Do not make up information.

        User's Question: {question}

        Search Results:
        {json.dumps(search_results, indent=2)}

        Answer:
        """

        llm_response = llm_model.generate_content(prompt)
        return llm_response.text

    except OpenSearchException as e:
        return f"An error occurred while querying the database: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def main():
    """Main function to run the CLI chatbot."""
    llm_model, embedding_model = initialize_models()
    opensearch_client = initialize_opensearch_client()

    print("\nProduct Chatbot (Vector Search Enabled) Initialized.")
    print("Type 'quit' or 'exit' to end the session.")
    print("------------------------------")

    while True:
        try:
            user_question = input("You: ")
            if user_question.lower() in ['quit', 'exit']:
                break

            bot_answer = ask_chatbot(user_question, opensearch_client, llm_model, embedding_model)
            print(f"\nBot: {bot_answer}\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}\n", file=sys.stderr)

    print("\nChat session ended. Goodbye!")

if __name__ == '__main__':
    main()

