import os
import google.generativeai as genai
from opensearchpy import OpenSearch, RequestsHttpConnection
from sentence_transformers import SentenceTransformer
import sys

# --- Configuration ---
GEMINI_API_KEY = "AIzaSyDHeGlRZOU4Z311enKGFyH5m7LV3-NHqco" # Use your actual Gemini API key
OPENSEARCH_PASSWORD = "OpenSearch!2025" # The password you set in docker-compose.yml

# --- Configure Models ---
try:
    # LLM for generating answers
    genai.configure(api_key=GEMINI_API_KEY)
    llm_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    # Embedding model to match your data (all-MiniLM-L6-v2)
    # UPDATED: Load the model from the user-specified local directory.
    print("Loading embedding model from local files...")
    local_model_path = "./all-MiniLM-L6-v2"  # Path to the folder with model files
    if not os.path.isdir(local_model_path):
        print(f"Error: The local model directory was not found at '{local_model_path}'", file=sys.stderr)
        print("Please follow the instructions in manual_model_download.md to download the model files.", file=sys.stderr)
        sys.exit(1)
        
    embedding_model = SentenceTransformer(local_model_path) 
    print("Embedding model loaded.")

except Exception as e:
    print(f"Error configuring models: {e}", file=sys.stderr)
    print("Please ensure your API key is valid.", file=sys.stderr)
    sys.exit(1)


# --- OpenSearch Connection ---
host = 'localhost'
port = 9200
auth = ('admin', OPENSEARCH_PASSWORD)
index_name = 'products'

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


def get_embedding(text):
    """Generates a vector embedding for the given text using SentenceTransformer."""
    try:
        # The encode function returns a NumPy array, which we convert to a list
        return embedding_model.encode(text).tolist()
    except Exception as e:
        print(f"Error generating embedding: {e}", file=sys.stderr)
        return None


def ask_chatbot(question):
    """
    Performs a hybrid search (vector + keyword) and uses the LLM to generate an answer.
    """
    try:
        # 1. Generate an embedding for the user's question
        print("Generating embedding for your question...")
        query_vector = get_embedding(question)
        if query_vector is None:
            return "Sorry, I couldn't process your question to generate an embedding."

        # 2. Perform a Hybrid Search in OpenSearch
        search_body = {
            "size": 5, 
            "query": {
                "hybrid": {
                    "queries": [
                        {
                            "multi_match": {
                                "query": question,
                                "fields": ["productDescription", "feature", "designBlock", "ipReference", "partNumber", "description"]
                            }
                        },
                        {
                            "knn": {
                                "featureEmbedding": { "vector": query_vector, "k": 5 }
                            }
                        },
                        {
                            "knn": {
                                "productDescriptionEmbedding": { "vector": query_vector, "k": 5 }
                            }
                        },
                        {
                            "knn": {
                                "descriptionEmbedding": { "vector": query_vector, "k": 5 }
                            }
                        }
                    ]
                }
            }
        }
        
        print("Searching database...")
        response = client.search(index=index_name, body=search_body)
        search_results = [hit['_source'] for hit in response['hits']['hits']]

        if not search_results:
            return "I couldn't find any information related to your question in the database."

        # 3. Augment with LLM for final answer generation
        prompt = f"""
        You are a helpful assistant for a product database.
        Answer the following question based ONLY on the provided search results.
        If the information is not in the results, state that you cannot find the answer in the provided data.
        Be concise and clear in your answer.

        Question: "{question}"

        Search Results:
        {search_results}

        Answer:
        """

        print("Generating answer...")
        llm_response = llm_model.generate_content(prompt)
        return llm_response.text

    except Exception as e:
        return f"An error occurred: {e}"

def main():
    """Main function to run the chatbot in a loop."""
    print("\nProduct Chatbot (Vector Search Enabled) Initialized.")
    print("Type 'quit' or 'exit' to end the session.")
    print("-" * 30)

    while True:
        user_question = input("You: ")
        if user_question.lower() in ['quit', 'exit']:
            print("Bot: Goodbye!")
            break
        
        if not user_question.strip():
            continue

        answer = ask_chatbot(user_question)
        print(f"\nBot: {answer}\n")


if __name__ == '__main__':
    main()

