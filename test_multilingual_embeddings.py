# Installation instructions:
# pip install --upgrade google-cloud-aiplatform numpy

import vertexai # Use Vertex AI SDK
from vertexai.preview.language_models import TextEmbeddingModel # Specific model class
import os
import sys
import numpy as np # Added for similarity calculation
# Removed google.genai imports
# --- Configuration ---
# Option 1: Set Environment Variables (Recommended)
# Ensure these environment variables are set before running the script:
# export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
# export GOOGLE_CLOUD_LOCATION="your-gcp-location" # e.g., us-central1
# export GOOGLE_GENAI_USE_VERTEXAI=True

# Option 2: Set Project ID and Location directly in the script (Less Secure)
# If you prefer not to use environment variables, uncomment and set these variables:
os.environ["GOOGLE_CLOUD_PROJECT"] = "roo-code-459017"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"
# Removed GOOGLE_GENAI_USE_VERTEXAI as it's not needed for google-cloud-aiplatform SDK

# --- Source Document ---
source_document = """
R. Wes Lawrence, a seasoned professional with over 20 years of experience in the music and entertainment industry, is the founder and driving force behind I AM HIM Management. His journey began in tour management, where he honed his skills working with renowned artists like 50 Cent, G-Unit, Ciara, Chris Brown, Lil Wayne, Drake, Nicki Minaj, T.I., Young Jeezy, Gucci Mane, Three 6 Mafia, Waka Flocka Flame, Soulja Boy, Trey Songz, and many others. This extensive background provided him with invaluable insights into the complexities of artist development, brand building, and navigating the ever-evolving entertainment landscape.

Recognizing the need for a more holistic and artist-centric approach, Wes established I AM HIM Management. The company offers a comprehensive suite of services designed to empower artists and athletes. These services include artist management, tour management, publishing administration, record label services, and strategic consulting. I AM HIM Management distinguishes itself through its deep industry connections, personalized strategies, and unwavering commitment to its clients' long-term success.

In tour management, the company leverages Wes's vast experience to ensure seamless execution, handling logistics, scheduling, budgeting, and personnel management for tours of all scales. For publishing, they assist artists in managing their copyrights, collecting royalties, and exploring synchronization opportunities in film, TV, and advertising. The record label services division supports artists through the entire music release process, from production and distribution to marketing and promotion.

Beyond music, I AM HIM Management extends its expertise to athletes, providing representation, contract negotiation, brand endorsements, and financial planning guidance. The company's philosophy centers on building strong, transparent relationships with clients, acting as trusted partners in their careers. With a proven track record and a forward-thinking approach, I AM HIM Management is poised to shape the future of talent representation in the entertainment and sports industries.
"""

# --- Sample Queries ---
queries = [
    # Hebrew
    "מה הרקע של ר. ווס לורנס?",
    "אילו שירותים מציעה I AM HIM Management?",
    "עם אילו אמנים מפורסמים עבד ווס לורנס?",
    "מהי המומחיות של I AM HIM Management בניהול סיבובי הופעות?",
    "כיצד I AM HIM Management מסייעת לספורטאים?",
    # Russian
    "Расскажите о Р. Уэсе Лоуренсе.",
    "Какие услуги по управлению турами предоставляет I AM HIM Management?",
    "С кем сотрудничал Уэс Лоуренс в музыкальной индустрии?",
    "Что такое I AM HIM Management?",
    "Как I AM HIM Management помогает артистам с издательскими правами?",
    # Amharic
    "ስለ አር. ዌስ ላውረንስ ማንነት ይንገሩኝ?",
    "I AM HIM Management ምን አይነት የአስተዳደር አገልግሎቶችን ይሰጣል?",
    "ዌስ ላውረንስ ከማን ጋር ነው የሰራው?",
    "የጉብኝት አስተዳደር ምንድን ነው?",
    "I AM HIM Management ለአትሌቶች ምን ያደርጋል?"
]

# --- Embedding Model ---
MODEL_NAME = "text-embedding-large-exp-03-07" # Using the target experimental model
TOP_N_RESULTS = 3 # Number of top results to display for each query

# --- Helper Functions ---
def calculate_cosine_similarity(vec1, vec2):
    """Calculates the cosine similarity between two numpy vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def generate_embeddings(texts, task_type, model_name=MODEL_NAME):
    """
    Generates embeddings using the Vertex AI SDK TextEmbeddingModel,
    handling batch size limit of 1 by iterating.
    """
    print(f"\nGenerating embeddings for {len(texts)} texts (one by one) using model: {model_name} with task type: {task_type}...")
    all_embeddings = []
    model = None # Initialize model outside the loop for efficiency
    task_type_supported = True # Flag to track if task_type arg works

    for i, text in enumerate(texts):
        print(f"  Processing text {i+1}/{len(texts)}...")
        try:
            if model is None: # Lazy initialization
                 model = TextEmbeddingModel.from_pretrained(model_name)

            # Prepare the single-item list for the API call
            text_list = [text]

            if task_type_supported:
                try:
                    # Attempt with task_type
                    embeddings_result = model.get_embeddings(text_list, task_type=task_type)
                except TypeError as te:
                    if "unexpected keyword argument 'task_type'" in str(te):
                        print("  Warning: task_type argument not supported. Proceeding without it.")
                        task_type_supported = False # Don't try task_type again
                        embeddings_result = model.get_embeddings(text_list)
                    else:
                        raise te # Re-raise other TypeErrors
            else:
                 # task_type already known to be unsupported
                 embeddings_result = model.get_embeddings(text_list)

            # Extract the single embedding vector
            if embeddings_result and len(embeddings_result) == 1:
                all_embeddings.append(embeddings_result[0].values)
            else:
                 # Should not happen with batch size 1, but good to check
                 print(f"Warning: Unexpected result structure for text {i+1}: {embeddings_result}")
                 # Decide how to handle: append None, raise error, etc.
                 # For now, let's raise to be safe
                 raise ValueError(f"Unexpected embedding result for text: {text}")

        except Exception as e:
            print(f"An error occurred during embedding generation for text {i+1}: {e}")
            # Option: continue to next text or re-raise the exception
            raise e # Re-raise to stop the process on error

    print(f"Successfully generated {len(all_embeddings)} embeddings.")
    return all_embeddings

# --- Main Execution ---
def main():
    """Main function to run the multilingual retrieval test."""
    try:
        # --- Initialization ---
        # Configuration is now set directly above, so the check below is less critical
        # but kept for robustness in case the direct setting fails or is commented out later.
        if not os.getenv("GOOGLE_CLOUD_PROJECT") or not os.getenv("GOOGLE_CLOUD_LOCATION"):
            print("Error: GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION are not set.")
            print("Attempted to set them directly in the script, but failed or they are commented out.")
            sys.exit(1)
        # GOOGLE_GENAI_USE_VERTEXAI is removed. Initialize Vertex AI SDK instead.
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION")
        print(f"Initializing Vertex AI for project {project_id} in {location}...")
        vertexai.init(project=project_id, location=location)
        print("Vertex AI initialized.")
        # Removed genai.Client() initialization

        # --- Document Processing ---
        print("\n--- Processing Source Document ---")
        document_chunks = [chunk.strip() for chunk in source_document.strip().split('\n\n') if chunk.strip()]
        print(f"Split document into {len(document_chunks)} chunks.")
        if not document_chunks:
            print("Error: No document chunks found after splitting.")
            sys.exit(1)

        # --- Embedding Generation ---
        # Generate embeddings for document chunks
        # Removed client argument from calls
        chunk_embeddings = generate_embeddings(document_chunks, task_type="RETRIEVAL_DOCUMENT")

        # Generate embeddings for queries
        query_embeddings = generate_embeddings(queries, task_type="RETRIEVAL_QUERY")

        # --- Retrieval and Ranking ---
        print(f"\n--- Performing Retrieval (Top {TOP_N_RESULTS} Results) ---")
        for i, query in enumerate(queries):
            query_embedding = query_embeddings[i]
            similarities = []

            # Calculate similarity between the query and each chunk
            for j, chunk_embedding in enumerate(chunk_embeddings):
                similarity = calculate_cosine_similarity(query_embedding, chunk_embedding)
                similarities.append((similarity, document_chunks[j]))

            # Sort results by similarity (descending)
            similarities.sort(key=lambda x: x[0], reverse=True)

            # Print results for the current query
            print(f"\nQuery: {query}")
            print("-" * 30)
            if not similarities:
                print("No relevant chunks found.")
            else:
                for k in range(min(TOP_N_RESULTS, len(similarities))):
                    score, chunk_text = similarities[k]
                    print(f"Rank {k+1} (Score: {score:.4f}):")
                    print(f"  Chunk: {chunk_text[:150]}...") # Print first 150 chars
                    print("-" * 10)

        print("\nMultilingual retrieval test completed successfully.")

    except Exception as e:
        print(f"\nAn unexpected error occurred in the main execution: {e}")
        print("Please ensure:")
        print("1. You have run 'pip install --upgrade google-cloud-aiplatform numpy'.")
        print("2. Environment variables (GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION) are correctly set OR defined in the script.")
        print("3. You have authenticated with Google Cloud (e.g., using 'gcloud auth application-default login').")
        print("4. The Vertex AI API (aiplatform.googleapis.com) is enabled for your project.")
        sys.exit(1)


if __name__ == "__main__":
    main()