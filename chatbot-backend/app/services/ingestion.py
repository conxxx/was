# app/services/ingestion.py

# --- Imports ---
import os
import json
import time
import traceback
import hashlib
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from datetime import datetime
from pdfminer.high_level import extract_text as extract_text_from_pdf
from docx import Document
import chardet
from celery.utils.log import get_task_logger
import random # Added for jitter calculation

# --- ADD LANGCHAIN SPLITTER IMPORT ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
# ------------------------------------

# GCP imports
# GCP imports
import vertexai # Use Vertex AI SDK
from google.cloud import storage
# from vertexai.language_models import TextEmbeddingModel # Replaced with google.genai
from google import genai # Added for new embedding model
from google.genai.types import EmbedContentConfig # Added for new embedding model config
from google.cloud import aiplatform # Keep for ME client
from google.api_core.exceptions import GoogleAPICallError, ServiceUnavailable, ResourceExhausted, InternalServerError, DeadlineExceeded # Added more specific exceptions
from requests.exceptions import Timeout, ConnectionError, HTTPError, RequestException # Added requests exceptions

# Flask/App specific imports
from flask import current_app
from sqlalchemy.exc import OperationalError # Example DB error (adjust if needed)
from celery import current_app as celery_current_app # Use celery's current_app for logger
from app import db  # Import db session
from celery_worker import celery_app # Import celery_app instance from its definition file
from app.models import Chatbot, VectorIdMapping # Import models

# --- Import shared SSE utility ---
# SSE push is now handled within Chatbot model methods, so this might be removable if not used elsewhere
try:
    from app.sse_utils import push_status_update
except ImportError:
    print("WARNING: app.sse_utils not found. SSE status updates will not be sent.")
    def push_status_update(chatbot_id, status, client_id): pass # Dummy function

# --- End Imports ---


# --- Constants ---
MAX_PAGES_INGEST = 100
REQUEST_TIMEOUT = 20    # Increased timeout slightly
USER_AGENT = "ChatbotDataIngestionBot/1.0 (requests)"
CHUNK_SIZE = 1000       # Size of text chunks
CHUNK_OVERLAP = 100     # Overlap between chunks
EMBEDDING_BATCH_SIZE = 5 # Reduced batch size for potentially better reliability
# UPSERT_BATCH_SIZE removed - not used for trigger method
# Use Vertex AI model name convention
EMBEDDING_MODEL_NAME = "text-embedding-large-exp-03-07" # Default Vertex AI embedding model name
# GEMINI_EMBEDDING_MODEL removed
# --- ADJUST CHUNKING PARAMS (tune these as needed) ---
# Smaller chunk size to aim for more focused context, reduce LLM input size
# Cost Constraint Focus: Smaller chunks can increase embedding calls slightly,
# but better relevance should allow reducing RAG_TOP_K later, saving LLM cost.
RECURSIVE_CHUNK_SIZE = 700 # Target size in characters (adjust based on testing)
RECURSIVE_CHUNK_OVERLAP = 100 # Overlap (adjust based on testing)
# Using default separators: ["\n\n", "\n", " ", ""] - prioritizes paragraphs/lines
# --------------------------------------------------

# --- GCP Config (Ideally load from app.config which reads .env) ---
PROJECT_ID = os.environ.get('PROJECT_ID', "roo-code-459017")
REGION = os.environ.get('REGION', "us-central1")
BUCKET_NAME = os.environ.get('BUCKET_NAME', "was-bucket4")
# *** INDEX_ENDPOINT_ID Needed to get Index client object ***
INDEX_ENDPOINT_ID = os.environ.get('INDEX_ENDPOINT_ID', "7916908131475521536") # Updated endpoint ID
INDEX_ID = os.environ.get('INDEX_ID', "6453695649417265152") # Updated index ID
# --- End Constants ---

# Define specific retryable exceptions for ingestion based on doc.md
INGESTION_RETRYABLE_EXCEPTIONS = (
    GoogleAPICallError,     # General GCP API errors
    ServiceUnavailable,     # Specific 503 type
    ResourceExhausted,      # Specific 429 type (Rate Limiting)
    InternalServerError,    # Specific 500 type from GCP
    DeadlineExceeded,       # Request timed out on GCP side
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
    # Add specific DB errors if needed and if they are transient
    # OperationalError,
    # Note: HTTPError from requests is NOT included here by default.
    # If needed, it should be caught explicitly in the task to check status code.
)

# --- REMOVE OLD chunk_text HELPER ---
# def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
#    """Simple text chunking."""
#    if not text: return []
#    chunks = []; start = 0
#    while start < len(text):
#        end = start + size
#        chunks.append(text[start:end])
#        start += size - overlap
#        if start < 0: start = 0
#    return chunks
# ------------------------------------

# --- Helper: Initialize GCP Clients (Includes Index Client for Trigger) ---
# No longer needs 'app' passed explicitly, uses current_app
def initialize_gcp_clients():
    """Initializes Storage, Vertex AI Embedding Model, and Matching Engine Index clients."""
    logger = current_app.logger
    logger.info("Attempting to initialize GCP clients (Storage, Google GenAI Client, ME Index)...") # Updated log
    project_id = current_app.config.get('PROJECT_ID', PROJECT_ID)
    region = current_app.config.get('REGION', REGION)
    bucket_name = current_app.config.get('BUCKET_NAME', BUCKET_NAME)
    embedding_model_name = current_app.config.get('EMBEDDING_MODEL_NAME', EMBEDDING_MODEL_NAME) # Get model name for logging
    index_endpoint_id = current_app.config.get('INDEX_ENDPOINT_ID', INDEX_ENDPOINT_ID)

    if not index_endpoint_id:
        logger.error("Ingestion Service: FATAL - INDEX_ENDPOINT_ID is not configured.")
        raise ValueError("INDEX_ENDPOINT_ID configuration is missing.")

    storage_client, bucket, genai_client, index_client = None, None, None, None # Changed embedding_model to genai_client

    # --- Explicitly set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION ---
    # This ensures genai.Client() picks up the correct project/location,
    # overriding any potentially empty or incorrect values from the worker's environment.
    if project_id:
        os.environ['GOOGLE_CLOUD_PROJECT'] = project_id
        logger.info(f"Ingestion Service: Explicitly set GOOGLE_CLOUD_PROJECT env var to: {project_id}")
    else:
        logger.warning("Ingestion Service: PROJECT_ID from app config is empty. genai.Client() might fail if GOOGLE_CLOUD_PROJECT is not already correctly set.")

    if region:
        os.environ['GOOGLE_CLOUD_LOCATION'] = region
        logger.info(f"Ingestion Service: Explicitly set GOOGLE_CLOUD_LOCATION env var to: {region}")
    else:
        logger.warning("Ingestion Service: REGION from app config is empty. genai.Client() might fail if GOOGLE_CLOUD_LOCATION is not already correctly set.")
    # --------------------------------------------------------------------

    try:
        # Initialize Vertex AI SDK
        logger.info(f"Initializing Vertex AI SDK: Project={project_id}, Region={region}")
        vertexai.init(project=project_id, location=region)
        logger.info("Vertex AI SDK initialized.")

        # Initialize GCS Client
        logger.info(f"Initializing GCS Client for bucket: {bucket_name}...")
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_name)
        logger.info("GCS Client initialized.")

        # Initialize Google GenAI Client (for Embeddings)
        logger.info(f"Initializing Google GenAI Client for Embeddings (using model name: '{embedding_model_name}')...")
        # Ensure GOOGLE_GENAI_USE_VERTEXAI=True is set in the environment
        env_var_value_ingestion = os.getenv('GOOGLE_GENAI_USE_VERTEXAI')
        logger.info(f"Ingestion Service: Value of GOOGLE_GENAI_USE_VERTEXAI before genai.Client(): {env_var_value_ingestion}")
        if env_var_value_ingestion != 'True':
            logger.warning("Ingestion Service: GOOGLE_GENAI_USE_VERTEXAI environment variable is NOT 'True' when genai.Client() is called. This is required for google-genai SDK to target Vertex AI.")
        genai_client = genai.Client()
        logger.info(f"Google GenAI Client initialized. Embedding model '{embedding_model_name}' will be called via this client.")

        # Initialize Matching Engine Endpoint client
        logger.info(f"Initializing Matching Engine Endpoint client for ID: {index_endpoint_id}...")
        endpoint_name = f"projects/{project_id}/locations/{region}/indexEndpoints/{index_endpoint_id}"
        index_endpoint_client = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=endpoint_name)
        logger.info(f"Connected to ME Endpoint: {index_endpoint_client.resource_name}")

        index_client = None
        if index_endpoint_client.deployed_indexes:
            deployed_index_info = index_endpoint_client.deployed_indexes[0]
            index_resource_name = deployed_index_info.index
            logger.info(f"Initializing Matching Engine Index client for: {index_resource_name}")
            # Use the aiplatform.MatchingEngineIndex class for streaming updates
            index_client = aiplatform.MatchingEngineIndex(index_name=index_resource_name)
            logger.info(f"Initialized ME Index client: {index_client.resource_name}")
        else:
            logger.error(f"FATAL: No deployed index found on endpoint {endpoint_name}")
            raise RuntimeError(f"No deployed index found on endpoint {endpoint_name}")

        logger.info("All required GCP clients initialized successfully.")
        return storage_client, bucket, genai_client, index_client # Return genai_client

    except Exception as e:
        logger.error(f"Failed to initialize one or more GCP clients: {e}", exc_info=True)
        raise


# --- Helper: Save Chunk to GCS ---
# No longer needs 'app' passed explicitly, uses current_app
def save_chunk_to_gcs(bucket, client_id, chatbot_id, source_identifier, chunk_index, chunk, task_instance=None, max_attempts=3): # Added chatbot_id
    """
    Saves text chunk to GCS, returns blob name and unique vector ID.
    Includes retry logic for GCS upload failures.

    Args:
        bucket: GCS bucket object
        client_id: Client identifier
        source_identifier: Source identifier for the chunk
        chunk_index: Index of the chunk
        chunk: Text content of the chunk
        task_instance: The Celery task instance for retries (self)
        max_attempts: Maximum number of local retry attempts
    """
    logger = current_app.logger
    log_source_id = (source_identifier[:75] + '...') if len(source_identifier) > 78 else source_identifier
    logger.debug(f"Attempting save chunk {chunk_index} for '{log_source_id}'")

    # Prepare the blob name and vector ID outside the retry loop
    hashed_source = hashlib.sha256(source_identifier.encode()).hexdigest()[:16]
    # Use chatbot_id for GCS path and vector ID
    gcs_blob_name = f"chatbot_{chatbot_id}/source_{hashed_source}/{chunk_index}.txt"
    vector_id = f"chatbot_{chatbot_id}_source_{hashed_source}_chunk_{chunk_index}" # Added 'source' and 'chunk' for clarity
    # Ensure vector_id does not exceed Vertex AI limits (typically 64 chars, but check specific limits)
    # If it might exceed, consider a shorter hash or different ID scheme.
    if len(vector_id) > 64:
        # Option 1: Truncate (potential for collisions if not careful)
        # vector_id = vector_id[:64]
        # Option 2: Use a shorter hash or a different ID generation method
        logger.warning(f"Generated Vector ID exceeds 64 characters: {vector_id}. Consider shortening scheme.")
        # For now, we'll let it pass but log the warning. Vertex AI might truncate or reject it.

    # Local retry loop for GCS uploads
    for attempt in range(1, max_attempts + 1):
        try:
            blob = bucket.blob(gcs_blob_name)
            blob.upload_from_string(chunk, content_type='text/plain; charset=utf-8')
            logger.info(f"Saved chunk: gs://{bucket.name}/{gcs_blob_name}")
            return gcs_blob_name, vector_id

        except GoogleAPICallError as gcs_err:
            # Check if the error is retryable based on the global definition
            if isinstance(gcs_err, INGESTION_RETRYABLE_EXCEPTIONS):
                # If retryable, let it propagate up to the main task handler
                logger.warning(f"Chatbot {chatbot_id}: Retryable GCS error saving chunk {chunk_index} for '{log_source_id}': {gcs_err}. Propagating for Celery retry.")
                raise gcs_err # Propagate up
            else:
                # If not retryable according to the list, log as error and fail chunk
                logger.error(f"Chatbot {chatbot_id}: Non-retryable GCS error saving chunk {chunk_index} for '{log_source_id}': {gcs_err}", exc_info=True)
                return None, None # Indicate failure for this chunk

        except Exception as e:
            # Non-retryable errors
            logger.error(f"Failed GCS upload chunk {chunk_index} for '{log_source_id}' with non-retryable error: {e}", exc_info=True)
            return None, None

    # This should not be reached due to the return statements in the loop
    return None, None


# --- Embeddings Generation (STREAM UPSERT VERSION) ---
# Modified to use google-genai SDK and include retry logic via the bound task instance ('self')
# Updated signature to accept genai_client instead of embedding_model
def generate_and_trigger_batch_update(task_instance, chatbot_id, client_id, genai_client, bucket, index_client, chunks_data: list, storage_client=None):
    """
    Generates embeddings using google-genai and directly upserts them to the index using streaming updates.
    Includes retry logic for API calls.
    Args:
        task_instance: The bound Celery task instance (self) for retrying.
        genai_client: Initialized google.genai.Client instance.
        ... other args ...
    """
    logger = current_app.logger
    embedding_model_name = current_app.config.get('EMBEDDING_MODEL_NAME', EMBEDDING_MODEL_NAME) # Get model name for logging/use
    logger.info(f"Using embedding model via GenAI Client: {embedding_model_name}")

    # Validate inputs
    if not chunks_data:
        logger.warning(f"Chatbot {chatbot_id}: No chunks to process.")
        return False
    if not bucket or not index_client or not genai_client: # Check genai_client
        logger.error(f"Chatbot {chatbot_id}: Invalid bucket, index client, or GenAI client.")
        return False

    try:
        # Step 1: Generate Embeddings and Upsert to Index
        logger.info(f"Chatbot {chatbot_id}: Starting embedding generation and indexing for {len(chunks_data)} chunks...") # Removed model name log here
        embeddings_generated_count = 0
        embeddings_upserted_count = 0
        ids_for_batch = [data['id'] for data in chunks_data]
        texts_to_embed = [data['text'] for data in chunks_data]
        processed_chunks = []  # Store chunks with their embeddings

        from google.cloud.aiplatform_v1.types import index as index_types

        all_datapoints_to_upsert = []
        all_mappings_to_save = []

        logger.info(f"Chatbot {chatbot_id}: Processing {len(chunks_data)} chunks individually for embedding...")

        for index, chunk in enumerate(chunks_data):
            vec_id = chunk['id']
            text = chunk['text']
            source_id = chunk['source'] # Get source identifier from chunk data

            logger.debug(f"Chatbot {chatbot_id}: Processing chunk {index + 1}/{len(chunks_data)}, ID: {vec_id}")

            try:
                # --- Generate embedding for ONE chunk using GenAI Client ---
                task_type_for_embedding = "RETRIEVAL_DOCUMENT" # Use DOCUMENT type for ingestion

                api_response = genai_client.models.embed_content(
                    model=embedding_model_name,
                    contents=[text], # List containing single text chunk
                    config=EmbedContentConfig(
                        task_type=task_type_for_embedding,
                        # output_dimensionality=3072, # Optional: default for text-embedding-large-exp-03-07
                        # auto_truncate=False # Optional: default is True (silent truncation)
                    )
                )

                # Check response structure (expecting response.embeddings list)
                if api_response and api_response.embeddings:
                    embedding_values = api_response.embeddings[0].values
                    if embedding_values:
                        embedding_vector = embedding_values
                        embeddings_generated_count += 1
                    else:
                        logger.warning(f"Chatbot {chatbot_id}: Received no/empty embedding values for chunk id {vec_id}. Skipping.")
                        continue # Skip to next chunk if embedding is empty
                else:
                    logger.error(f"Chatbot {chatbot_id}: Unexpected response structure from embed_content for chunk id {vec_id}. Response: {api_response}")
                    # Decide how to handle: skip chunk or raise error? Raising allows retry.
                    raise ValueError(f"Unexpected embedding response structure for chunk {vec_id}")

                # Store chunk data for reference (optional, if needed later)
                processed_chunks.append({
                    "id": vec_id,
                    "text": text,
                    "embedding": embedding_vector
                })

                # Create datapoint for this chunk
                datapoint = index_types.IndexDatapoint(
                    datapoint_id=vec_id,
                    feature_vector=embedding_vector,
                    restricts=[
                        # Use ONLY chatbot_id for primary data isolation
                        index_types.IndexDatapoint.Restriction(
                            namespace="chatbot_id",
                            allow_list=[str(chatbot_id)] # Ensure chatbot_id is a string
                        )
                        # Removed client_id restriction
                    ]
                )
                all_datapoints_to_upsert.append(datapoint)

                # Prepare mapping for this chunk
                if not source_id:
                    logger.warning(f"Chatbot {chatbot_id}: Missing source identifier for vector_id {vec_id} during mapping preparation.")
                all_mappings_to_save.append(VectorIdMapping(
                    chatbot_id=chatbot_id,
                    vector_id=vec_id,
                    source_identifier=source_id
                ))

            except (GoogleAPICallError, ServiceUnavailable, ResourceExhausted, InternalServerError, DeadlineExceeded) as retryable_error:
                # Let specific retryable errors propagate up to the main task handler
                logger.warning(f"Chatbot {chatbot_id}: Retryable API error during embedding chunk {index + 1}/{len(chunks_data)} (ID: {vec_id}): {retryable_error}. Propagating for Celery retry.")
                raise retryable_error # Propagate up

            except Exception as embedding_error:
                logger.error(f"Chatbot {chatbot_id}: Error generating embedding for chunk id {vec_id}: {embedding_error}", exc_info=True)
                # Re-raise the exception to be caught by the outer try/except, potentially triggering Celery retry
                raise embedding_error

        # === Upsert and Save Mappings AFTER processing all chunks ===
        if all_datapoints_to_upsert:
            logger.info(f"Chatbot {chatbot_id}: Upserting {len(all_datapoints_to_upsert)} datapoints to index in bulk...")
            try:
                # --- Upsert ALL datapoints ---
                index_client.upsert_datapoints(datapoints=all_datapoints_to_upsert)
                embeddings_upserted_count = len(all_datapoints_to_upsert)
                logger.info(f"Chatbot {chatbot_id}: Successfully upserted {embeddings_upserted_count} datapoints")
                # --- End upsert ---

                # --- Save ALL Vector ID Mappings to DB (only after successful upsert) ---
                if all_mappings_to_save:
                    logger.info(f"Chatbot {chatbot_id}: Saving {len(all_mappings_to_save)} vector ID mappings to DB...")
                    try:
                        db.session.bulk_save_objects(all_mappings_to_save)
                        db.session.commit()
                        logger.info(f"Chatbot {chatbot_id}: Saved {len(all_mappings_to_save)} vector ID mappings to DB.")
                    except Exception as mapping_db_error:
                        db.session.rollback() # Rollback DB changes if mapping save fails
                        logger.error(f"Chatbot {chatbot_id}: Failed to save vector ID mappings to DB: {mapping_db_error}", exc_info=True)
                        # If upsert worked but DB save failed, this is problematic.
                        # Consider adding logic to potentially delete the upserted vectors or mark them as orphaned.
                        # For now, re-raise to indicate a critical failure.
                        raise RuntimeError("Failed to save mappings after successful upsert") from mapping_db_error
                else:
                    logger.warning(f"Chatbot {chatbot_id}: No mappings to save despite successful upsert (this shouldn't happen).")
                # --- End DB Mapping Save ---

            except (GoogleAPICallError, ServiceUnavailable, ResourceExhausted, InternalServerError, DeadlineExceeded) as retryable_upsert_error:
                # Let specific retryable errors propagate up to the main task handler
                logger.warning(f"Chatbot {chatbot_id}: Retryable API error during bulk upsert: {retryable_upsert_error}. Propagating for Celery retry.")
                raise retryable_upsert_error # Propagate up

            except Exception as upsert_error:
                logger.error(f"Chatbot {chatbot_id}: Error during bulk upsert of {len(all_datapoints_to_upsert)} datapoints: {upsert_error}", exc_info=True)
                # Re-raise the exception to be caught by the outer try/except
                raise upsert_error
        else:
            logger.warning(f"Chatbot {chatbot_id}: No datapoints were generated to upsert.")

        # --- End of modified processing loop ---
        logger.info(f"Chatbot {chatbot_id}: Completed embedding generation and indexing loop. Total embeddings generated: {embeddings_generated_count}, upserted: {embeddings_upserted_count}")

        # Check counts *after* the loop finishes
        if embeddings_generated_count == 0 and len(chunks_data) > 0: # Check if input was non-empty
            logger.error(f"Chatbot {chatbot_id}: No embeddings were successfully generated after processing all batches.")
            return False # Fail the step if no embeddings were generated at all

        if embeddings_upserted_count < embeddings_generated_count:
            logger.warning(f"Chatbot {chatbot_id}: Not all generated embeddings were successfully upserted ({embeddings_upserted_count}/{embeddings_generated_count}). Check logs for batch errors.")
            # Consider it a success if *any* were upserted, but log warning.

        if embeddings_upserted_count == 0 and len(chunks_data) > 0: # Check if input was non-empty
            logger.error(f"Chatbot {chatbot_id}: No embeddings were successfully upserted to index after processing all batches.")
            return False # Fail the step if none were upserted

        # If we reach here, processing likely succeeded (or no input chunks).
        logger.info(f"Chatbot {chatbot_id}: Embedding/Indexing step completed.")
        return True # Indicate this step succeeded

    except Exception as e: # Catch non-retryable errors during batch processing loop
        logger.error(f"Chatbot {chatbot_id}: Non-retryable error during embedding/upsert batch processing loop: {e}", exc_info=True)
        raise e # Reraise the non-retryable error

# Removed update_chatbot_status helper function.
# Status updates are now handled directly in run_ingestion_task using Chatbot model methods.
# --- process_uploaded_files (BATCH + TRIGGER VERSION) ---
# No longer needs 'app' passed explicitly, uses current_app
def process_uploaded_files(chatbot_id, client_id, uploaded_file_paths, storage_client, bucket, task_instance=None): # Removed embedding_model, index_client
    logger = current_app.logger
    # No need for app.app_context() here, task runs within context
    logger.info(f"Chatbot {chatbot_id}: Starting process_uploaded_files for {len(uploaded_file_paths)} files...")
    # Status update handled by caller
    all_chunks_data = []
    file_errors = 0
    processed_files_count = 0 # Corrected indentation

    # --- Instantiate the text splitter here ---
    # Uses parameters defined in constants
    # length_function=len counts characters, which is simpler and avoids extra tokenizer dependency/cost
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=RECURSIVE_CHUNK_SIZE,
        chunk_overlap=RECURSIVE_CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False, # Use simple string separators
    )
    logger.info(f"Chatbot {chatbot_id}: Using RecursiveCharacterTextSplitter (Size: {RECURSIVE_CHUNK_SIZE}, Overlap: {RECURSIVE_CHUNK_OVERLAP})")
    # ---------------------------------------

    for file_index, file_path in enumerate(uploaded_file_paths):
         filename = os.path.basename(file_path)
         logger.info(f"Chatbot {chatbot_id}: Processing file {file_index + 1}/{len(uploaded_file_paths)}: '{filename}'")
         text = ""
         try:
             ext = os.path.splitext(filename)[1].lower()
             if ext == '.pdf': text = extract_text_from_pdf(file_path)
             elif ext == '.docx': doc = Document(file_path); text = '\n'.join([p.text for p in doc.paragraphs if p.text])
             elif ext == '.txt':
                 with open(file_path, 'rb') as f: raw = f.read(); enc = chardet.detect(raw)['encoding'] or 'utf-8'; text = raw.decode(enc, errors='ignore')
             else: logger.warning(f"Skipping unsupported file: {filename}"); continue
             processed_files_count += 1
             if not text or not text.strip(): logger.warning(f"No text extracted from file: {filename}"); continue

             # --- Replace old chunking with new splitter ---
             # chunks = chunk_text(text) # OLD
             chunks = text_splitter.split_text(text) # NEW
             # --------------------------------------------

             logger.info(f"Chatbot {chatbot_id}: Split '{filename}' into {len(chunks)} chunks.")
             for i, chunk in enumerate(chunks):
                 source_id = f"file://{filename}" # Use original filename as source ID
                 # Pass task_instance for potential GCS retry
                 # --- Start Inner Try-Except for chunk saving ---
                 try:
                     blob_name, vector_id = save_chunk_to_gcs(bucket, client_id, chatbot_id, source_id, i, chunk, task_instance=task_instance) # Pass chatbot_id
                     if blob_name and vector_id: all_chunks_data.append({"id": vector_id, "text": chunk, "source": source_id, "chunk_index": i})
                     else:
                         logger.warning(f"Chatbot {chatbot_id}: Failed to save chunk {i} for {source_id} (save_chunk_to_gcs returned None).")
                         file_errors += 1
                 except Exception as chunk_save_e:
                     logger.error(f"Chatbot {chatbot_id}: Error saving chunk {i} for {source_id}: {chunk_save_e}", exc_info=True)
                     file_errors += 1 # Increment file_errors for chunk-specific issues
                     # Continue to the next chunk within the same file
                 # --- End Inner Try-Except ---
         except Exception as e: logger.error(f"Chatbot {chatbot_id}: FAILED processing file '{filename}': {e}", exc_info=True); file_errors += 1

    logger.info(f"Chatbot {chatbot_id}: File parsing finished. Processed: {processed_files_count}, Chunks generated: {len(all_chunks_data)}, Errors: {file_errors}.")

    if file_errors > 0 and len(all_chunks_data) == 0 and processed_files_count > 0:
         logger.error(f"Chatbot {chatbot_id}: Errors processing files, no data extracted.")
         # Status update handled by caller
         return None # Return None on failure
    if not all_chunks_data: logger.warning(f"Chatbot {chatbot_id}: No file chunks generated. Skipping embedding step for files."); return [] # Return empty list if no chunks

    # Embedding/Indexing happens in the main task function now
    # Return the generated chunk data instead of True/False
    return all_chunks_data # Return list of dicts


# --- process_web_source (BATCH + TRIGGER VERSION) ---
# No longer needs 'app' passed explicitly, uses current_app
def process_web_source(chatbot_id, client_id, urls_to_process: list, source_type, storage_client, bucket, task_instance=None): # Removed embedding_model, index_client
    logger = current_app.logger
    # No need for app.app_context() here, task runs within context
    logger.info(f"Chatbot {chatbot_id}: Starting process_web_source for {len(urls_to_process)} URLs...")
    # Status update handled by caller
    all_chunks_data = []
    processed_count = 0
    fetch_errors = 0 # Corrected indentation
    session = requests.Session()
    session.headers.update({'User-Agent': USER_AGENT})

    # --- Instantiate the text splitter here ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=RECURSIVE_CHUNK_SIZE,
        chunk_overlap=RECURSIVE_CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    logger.info(f"Chatbot {chatbot_id}: Using RecursiveCharacterTextSplitter (Size: {RECURSIVE_CHUNK_SIZE}, Overlap: {RECURSIVE_CHUNK_OVERLAP})")
    # ---------------------------------------

    for url_index, url in enumerate(urls_to_process):
        try:
            logger.info(f"Chatbot {chatbot_id}: Processing URL {url_index+1}/{len(urls_to_process)}: {url}")

            # Fetch the web content with retry
            html_content = None
            try:
                logger.info(f"Chatbot {chatbot_id}: Fetching content from {url}")
                response = session.get(url, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()  # Raise exception for 4XX/5XX responses

                # Detect encoding
                if response.encoding.lower() == 'iso-8859-1':
                    detected_encoding = chardet.detect(response.content)
                    if detected_encoding['confidence'] > 0.7:
                        response.encoding = detected_encoding['encoding']

                html_content = response.text
                logger.info(f"Chatbot {chatbot_id}: Successfully fetched {len(html_content)} bytes from {url}")

            except (Timeout, ConnectionError) as net_err:
                 # Let specific network errors defined in INGESTION_RETRYABLE_EXCEPTIONS propagate
                 logger.warning(f"Chatbot {chatbot_id}: Retryable network error fetching {url}: {net_err}. Propagating for Celery retry.")
                 raise net_err # Propagate up
            except HTTPError as http_err:
                 # Handle HTTPError separately - fail non-5xx, propagate 5xx if desired (currently not in INGESTION_RETRYABLE_EXCEPTIONS)
                 status_code = http_err.response.status_code if http_err.response else 500
                 if 500 <= status_code <= 599:
                     # If we wanted to retry 5xx, we'd raise it here.
                     # Since it's not in INGESTION_RETRYABLE_EXCEPTIONS, we treat it as non-retryable for now.
                     logger.error(f"Chatbot {chatbot_id}: Non-retryable HTTP error {status_code} fetching {url}: {http_err}")
                     fetch_errors += 1
                     continue # Skip this URL
                 else:
                     # Non-5xx errors are definitely not retryable
                     logger.error(f"Chatbot {chatbot_id}: Non-retryable HTTP error {status_code} fetching {url}: {http_err}")
                     fetch_errors += 1
                     continue # Skip this URL
            except RequestException as req_err:
                 # Catch other non-retryable requests errors
                 logger.error(f"Chatbot {chatbot_id}: Non-retryable request error fetching {url}: {req_err}")
                 fetch_errors += 1
                 continue # Skip this URL

            # If fetch failed after retries or non-retryable error, html_content will be None
            if html_content is None:
                logger.warning(f"Chatbot {chatbot_id}: Skipping URL {url} due to fetch failure.")
                continue

            # Extract text from HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            # Remove script and style elements
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()
            # Get text, strip leading/trailing whitespace, separate lines
            text = soup.get_text(separator='\n', strip=True)

            if not text or not text.strip():
                logger.warning(f"No text extracted from URL: {url}")
                continue

            # --- Replace old chunking with new splitter ---
            # chunks = chunk_text(text) # OLD
            chunks = text_splitter.split_text(text) # NEW
            # --------------------------------------------

            logger.info(f"Chatbot {chatbot_id}: Split '{url}' into {len(chunks)} chunks.")

            # Save chunks to GCS
            file_errors = 0 # Track errors for this specific URL
            for chunk_index, chunk in enumerate(chunks):
                source_id = url # Use URL as source ID
                # Pass task_instance for potential GCS retry
                # --- Start Inner Try-Except for chunk saving ---
                try:
                    blob_name, vector_id = save_chunk_to_gcs(
                        bucket, client_id, chatbot_id, source_id, chunk_index, chunk, task_instance=task_instance # Pass chatbot_id
                    )

                    if blob_name and vector_id: # Check if save was successful
                        all_chunks_data.append({
                            'id': vector_id,
                            'text': chunk,
                            'source': source_id, # Store URL as source
                            'chunk_index': chunk_index
                        })
                    else:
                        logger.warning(f"Chatbot {chatbot_id}: Failed to save chunk {chunk_index} for {source_id} (save_chunk_to_gcs returned None).")
                        file_errors += 1 # Increment error count if save failed
                except Exception as chunk_save_e:
                    logger.error(f"Chatbot {chatbot_id}: Error saving chunk {chunk_index} for {source_id}: {chunk_save_e}", exc_info=True)
                    file_errors += 1 # Increment file_errors for chunk-specific issues
                    # Continue to the next chunk within the same URL
                # --- End Inner Try-Except ---

            processed_count += 1
            logger.info(f"Chatbot {chatbot_id}: Successfully processed {url}")

        except Exception as e:
            logger.error(f"Chatbot {chatbot_id}: Unexpected error processing {url}: {e}", exc_info=True)
            fetch_errors += 1
            continue

    logger.info(f"Chatbot {chatbot_id}: Web fetching finished. Processed: {processed_count}, Errors: {fetch_errors}, Chunks: {len(all_chunks_data)}.")

    if not all_chunks_data and processed_count == 0 and fetch_errors > 0 and len(urls_to_process) > 0:
         logger.error(f"Chatbot {chatbot_id}: Failed to fetch any web content.")
         # Status update handled by caller
         return None # Return None on failure
    if not all_chunks_data:
        logger.warning(f"Chatbot {chatbot_id}: No web chunks generated. Skipping embedding step for web.")
        return [] # Return empty list if no chunks

    # Embedding/Indexing happens in the main task function now
    # Return the generated chunk data instead of True/False
    return all_chunks_data # Return list of dicts


# --- Internal Logic for Ingestion ---
# --- Internal Logic for Ingestion ---
# Modified to handle specific source_details for updates
def _perform_ingestion(task_instance, chatbot_id, client_id, source_details):
    """
    Internal logic for the ingestion task. Separated for easier testing.
    Handles specific source additions if provided in source_details.
    This function should now let exceptions propagate up to run_ingestion_task.
    """ # Ensure this closing docstring marker is present
    logger = get_task_logger(__name__)
    # Removed duplicate function definition below (The actual 'def' line was removed previously, this comment is just a note)
    # The original definition starting below remains:
    # --- Removed erroneous lines from previous edit ---
    # Ensure correct indentation for the function body starts here
    is_specific_addition = 'urls_to_ingest' in source_details or 'files_to_ingest' in source_details
    logger.info(f"Task {task_instance.request.id}: Starting _perform_ingestion for Chatbot {chatbot_id} (Specific Addition: {is_specific_addition}, Update: {source_details.get('is_update', False)})")
    logger.debug(f"Task {task_instance.request.id}: Source Details: {source_details}")

    # --- Variables ---
    is_update = source_details.get('is_update', False)
    cleanup_ok = True
    index_client = None
    all_processed_chunks = []
    files_processed = False
    web_processed = False
    uploaded_file_basenames_for_cleanup = []
    task_arg_source_details = source_details.copy() # Use task_arg_source_details for specific additions

    # === STEP 1: Initialize Clients ===
    logger.info(f"Task {task_instance.request.id}: STEP 1: Initializing GCP Clients...")
    start_time_init = time.time()
    # Exceptions during init should propagate up
    storage_client, bucket, genai_client, index_client = initialize_gcp_clients() # Use genai_client
    logger.info(f"Task {task_instance.request.id}: STEP 1: GCP clients init OK ({time.time() - start_time_init:.2f}s).")
    # Initial status update handled in run_ingestion_task

    # === STEP 1b: Load Full Source Details from DB if NOT specific addition ===
    if not is_specific_addition:
        logger.info(f"Task {task_instance.request.id}: Not a specific addition. Loading full details from DB.")
        # DB interaction should be within the app context of run_ingestion_task
        chatbot = db.session.get(Chatbot, chatbot_id) # Assumes db session is available
        if not chatbot:
            logger.error(f"Task {task_instance.request.id}: Chatbot {chatbot_id} not found in _perform_ingestion.")
            raise ValueError(f"Chatbot {chatbot_id} not found") # Raise error to stop
        try:
            full_source_details_str = chatbot.source_details
            if not full_source_details_str:
                logger.error(f"Task {task_instance.request.id}: Chatbot record has no source_details for full ingestion.")
                raise ValueError("Missing source details in DB")
            source_details = json.loads(full_source_details_str)
            logger.debug(f"Task {task_instance.request.id}: Loaded full source_details from DB: {source_details}")
            is_update = source_details.get('is_update', False)
        except json.JSONDecodeError as json_err:
            logger.error(f"Task {task_instance.request.id}: Failed to parse source_details JSON from DB: {json_err}", exc_info=True)
            raise ValueError("Invalid source details format") from json_err
        except Exception as db_load_err:
            logger.error(f"Task {task_instance.request.id}: Error loading source_details from DB: {db_load_err}", exc_info=True)
            raise ValueError("DB error loading details") from db_load_err
    else:
        logger.info(f"Task {task_instance.request.id}: Processing specific sources provided in task arguments.")
        source_details = task_arg_source_details # Use the copied details for specific additions

    # === STEP 2: Cleanup Step (ONLY if is_update and NOT specific addition) ===
    logger.info(f"Task {task_instance.request.id}: STEP 2: Checking for data cleanup (is_update={is_update}, is_specific_addition={is_specific_addition})...")
    if is_update and not is_specific_addition:
        logger.info(f"Task {task_instance.request.id}: Running cleanup for update...")
        removed_files = source_details.get('removed_files_identifiers', [])
        removed_urls = source_details.get('removed_urls_identifiers', [])
        identifiers_to_remove = [f"file://{basename}" for basename in removed_files] + removed_urls

        if identifiers_to_remove:
            # Exceptions during cleanup should propagate up
            try:
                # Attempt to import get_rag_service safely
                from app.api.routes import get_rag_service
                rag_service = get_rag_service() # Call it to get instance
            except ImportError:
                 try:
                     from app.services.rag_service import RagService
                     rag_service = RagService() # Instantiate directly, removed logger arg
                 except ImportError:
                     logger.error(f"Task {task_instance.request.id}: Could not import get_rag_service or RagService. Cleanup cannot proceed.")
                     raise ImportError("RAG service unavailable for cleanup")

            logger.info(f"Task {task_instance.request.id}: Identifying vectors to delete for identifiers: {identifiers_to_remove}")
            mappings_to_delete = db.session.query(VectorIdMapping.vector_id)\
                .filter(VectorIdMapping.chatbot_id == chatbot_id)\
                .filter(VectorIdMapping.source_identifier.in_(identifiers_to_remove))\
                .all()
            vector_ids_to_delete = [m[0] for m in mappings_to_delete]

            if vector_ids_to_delete:
                logger.info(f"Task {task_instance.request.id}: Found {len(vector_ids_to_delete)} vector IDs to delete.")
                if not rag_service or not rag_service.clients_initialized:
                     logger.error(f"Task {task_instance.request.id}: RAG Service not available for cleanup.")
                     raise RuntimeError("RAG Service unavailable for cleanup.")

                cleanup_ok = rag_service.cleanup_stale_data(
                    vector_ids_to_delete=vector_ids_to_delete,
                    file_basenames_to_delete=removed_files,
                    chatbot_id=chatbot_id
                )

                if not cleanup_ok:
                    logger.error(f"Task {task_instance.request.id}: Cleanup of stale data failed. Aborting update ingestion.")
                    raise RuntimeError("Update failed during data cleanup.")
                else:
                    logger.info(f"Task {task_instance.request.id}: Stale data cleanup step completed successfully.")
            else:
                logger.info(f"Task {task_instance.request.id}: No existing vector mappings found for removed identifiers.")
        else:
             logger.info(f"Task {task_instance.request.id}: Skipping data cleanup (No identifiers to remove).") # Updated log message
    else: # Added else block for clarity
         logger.info(f"Task {task_instance.request.id}: Skipping data cleanup (Not an update OR is specific addition).")
    logger.info(f"Task {task_instance.request.id}: STEP 2: Data cleanup finished (or skipped). Cleanup OK: {cleanup_ok}")

    # === STEP 3: Process Uploaded Files ===
    files_to_process_basenames = source_details.get('files_to_ingest', []) if is_specific_addition else source_details.get('files_uploaded', [])
    logger.info(f"Task {task_instance.request.id}: STEP 3: Checking for file processing... Files: {len(files_to_process_basenames)}")
    if files_to_process_basenames:
        files_processed = True
        logger.info(f"Task {task_instance.request.id}: STEP 3a: Starting File Processing for {len(files_to_process_basenames)} files...")
        file_start = time.time()
        UPLOAD_FOLDER = 'uploads' # Corrected folder name case
        full_file_paths = [os.path.join(UPLOAD_FOLDER, basename) for basename in files_to_process_basenames if basename]

        existing_files = [fp for fp in full_file_paths if os.path.exists(fp)]
        if len(existing_files) != len(full_file_paths):
             missing_files = set(full_file_paths) - set(existing_files)
             logger.warning(f"Task {task_instance.request.id}: Some files listed do not exist in {UPLOAD_FOLDER}. Missing: {missing_files}")

        if not existing_files:
             logger.warning(f"Task {task_instance.request.id}: No existing files found to process.")
        else:
            # Exceptions from process_uploaded_files should propagate
            file_chunks_data = process_uploaded_files(chatbot_id, client_id, existing_files, storage_client, bucket, task_instance=task_instance)
            if file_chunks_data is None:
                logger.error(f"Task {task_instance.request.id}: File processing failed (returned None).")
                raise RuntimeError("File processing failed")
            else:
                all_processed_chunks.extend(file_chunks_data)
                logger.info(f"Task {task_instance.request.id}: STEP 3a: File Processing finished ({time.time()-file_start:.2f}s). Added {len(file_chunks_data)} chunks.")
                if not is_specific_addition:
                     uploaded_file_basenames_for_cleanup.extend(files_to_process_basenames)

        # --- File Cleanup Logic Removed - Moved to run_ingestion_task ---
        # The 'uploaded_file_basenames_for_cleanup' list is still populated above,
        # but the actual deletion will happen in the calling task upon success.
    else:
        logger.info(f"Task {task_instance.request.id}: STEP 3: No files to process.")
    logger.info(f"Task {task_instance.request.id}: STEP 3: File processing finished.")

    # === STEP 4: Process Web Sources ===
    urls_to_ingest = []
    if is_specific_addition:
        urls_to_ingest = source_details.get('urls_to_ingest', [])
    else:
        urls_to_ingest = source_details.get('selected_urls', [])
        if not urls_to_ingest and source_details.get('original_url'):
             urls_to_ingest = [source_details['original_url']]

    logger.info(f"Task {task_instance.request.id}: STEP 4: Checking for web processing... URLs: {len(urls_to_ingest)}")
    if urls_to_ingest:
        web_processed = True
        logger.info(f"Task {task_instance.request.id}: STEP 4a: Starting Web Processing ({len(urls_to_ingest)} URLs)...")
        web_start = time.time()
        web_source_type = 'web_specific' if is_specific_addition else 'web_filtered'
        # Exceptions from process_web_source should propagate
        web_chunks_data = process_web_source(chatbot_id, client_id, urls_to_ingest, web_source_type, storage_client, bucket, task_instance=task_instance)

        if web_chunks_data is None:
            logger.error(f"Task {task_instance.request.id}: Web processing failed (returned None).")
            raise RuntimeError("Web processing failed")
        else:
            all_processed_chunks.extend(web_chunks_data)
            logger.info(f"Task {task_instance.request.id}: STEP 4a: Web Processing finished ({time.time()-web_start:.2f}s). Added {len(web_chunks_data)} chunks.")
    else:
        logger.info(f"Task {task_instance.request.id}: STEP 4: No URLs selected or provided. Skipping web processing.")
    logger.info(f"Task {task_instance.request.id}: STEP 4: Web processing finished.")

    # === STEP 5: Embed and Index ALL Processed Chunks ===
    logger.info(f"Task {task_instance.request.id}: STEP 5: Checking for embedding/indexing... Total chunks: {len(all_processed_chunks)}")
    if all_processed_chunks:
         logger.info(f"Task {task_instance.request.id}: STEP 5a: Starting Embedding & Indexing for {len(all_processed_chunks)} chunks...")
         embedding_start = time.time()
         # Exceptions from generate_and_trigger_batch_update should propagate
         embedding_ok = generate_and_trigger_batch_update(
             task_instance=task_instance, chatbot_id=chatbot_id, client_id=client_id,
             genai_client=genai_client, # Pass the initialized GenAI client
             bucket=bucket, index_client=index_client,
             chunks_data=all_processed_chunks, storage_client=storage_client
         )
         logger.info(f"Task {task_instance.request.id}: STEP 5a: Embedding & Indexing finished ({time.time()-embedding_start:.2f}s). Result: {'OK' if embedding_ok else 'FAILED'}")
         if not embedding_ok:
             logger.error(f"Task {task_instance.request.id}: Embedding/indexing failed (returned False).")
             raise RuntimeError("Embedding/Indexing failed")
    elif files_processed or web_processed:
         # Sources processed but no chunks generated
         logger.warning(f"Task {task_instance.request.id}: Sources processed but no indexable chunks generated.")
         if is_specific_addition:
              logger.info(f"Task {task_instance.request.id}: Specific source addition resulted in no new data. Task considered successful.")
              # Return normally, success handled in run_ingestion_task
         else:
              logger.error(f"Task {task_instance.request.id}: Full ingestion failed to produce indexable chunks.")
              raise RuntimeError("Failed to process data into indexable chunks.")
    else:
         # No sources were even attempted
         logger.warning(f"Task {task_instance.request.id}: No data sources were provided or processed.")
         raise ValueError("No data sources provided")

    logger.info(f"Task {task_instance.request.id}: STEP 5: Embedding/Indexing finished.")

    # === STEP 6: Return total chunks processed for final status update ===
    logger.info(f"Task {task_instance.request.id}: --- _perform_ingestion COMPLETED Successfully ---")
    return len(all_processed_chunks), uploaded_file_basenames_for_cleanup # Return chunk count and files to cleanup on success

    # Removed the final try/except Exception block and finally block.
    # Errors should propagate to run_ingestion_task for handling.
    # File cleanup on critical error needs to be handled in run_ingestion_task's final except block.


# --- Main Ingestion Task Runner (Celery Task - REFACTORED) ---
@celery_app.task(
    bind=True,
    # autoretry_for=INGESTION_RETRYABLE_EXCEPTIONS, # REMOVED for manual interactive retry
    max_retries=5,
    retry_backoff=True,     # Enable exponential backoff
    retry_backoff_max=600,  # Max delay 10 minutes
    retry_jitter=True       # Add jitter to avoid thundering herd
)
def run_ingestion_task(self, chatbot_id, client_id, source_details):
    """
    Main Celery task for data ingestion with enhanced retry logic and DB status updates.
    """
    from app import create_app # Import create_app locally for context
    app = create_app()
    with app.app_context(): # Ensure all code runs within context
        logger = get_task_logger(__name__)
        chatbot = None # Initialize chatbot variable

        try:
            # --- Get Chatbot Instance ---
            chatbot = db.session.get(Chatbot, chatbot_id)
            if not chatbot:
                logger.error(f"Task {self.request.id}: Chatbot {chatbot_id} not found. Aborting.")
                self.update_state(state='FAILURE', meta={'error': 'Chatbot not found'})
                # No retry needed if chatbot doesn't exist
                return {'status': 'Failed', 'error': 'Chatbot not found'}

            # --- Initial Status Update ---
            # Use the model's method to set initial state if not already processing/retrying
            if chatbot.index_operation_state not in ['RUNNING', 'RETRYING']:
                 logger.info(f"Task {self.request.id}: Setting initial status to RUNNING for Chatbot {chatbot_id}")
                 operation_id = self.request.id # Use Celery task ID as operation ID
                 chatbot.start_index_operation(operation_id=operation_id)
                 # Note: start_index_operation commits the session

            logger.info(f"Task {self.request.id}: Starting ingestion for Chatbot {chatbot_id}, Attempt {self.request.retries + 1}/{self.max_retries + 1}")

            # --- Main Ingestion Logic ---
            # Call _perform_ingestion, passing 'self'. It now returns chunk count and files to cleanup.
            total_chunks_processed, files_to_cleanup = _perform_ingestion(self, chatbot_id, client_id, source_details)

            # --- Success ---
            logger.info(f"Task {self.request.id}: Ingestion successful for Chatbot {chatbot_id}. Processed {total_chunks_processed} chunks.")

            # --- Cleanup Processed Files on Success (ONLY for initial/full ingestion) ---
            is_specific_addition = 'urls_to_ingest' in source_details or 'files_to_ingest' in source_details
            if not is_specific_addition and files_to_cleanup:
                 logger.info(f"Task {self.request.id}: STEP 6: Cleaning up {len(files_to_cleanup)} successfully processed uploaded files..."); cleanup_start = time.time()
                 cleaned_count = 0
                 UPLOAD_FOLDER = 'uploads' # Ensure consistent folder name
                 full_paths_to_cleanup = [os.path.join(UPLOAD_FOLDER, basename) for basename in files_to_cleanup]
                 for path in full_paths_to_cleanup:
                     if path and os.path.exists(path):
                         try: os.remove(path); cleaned_count += 1
                         except Exception as cleanup_e: logger.warning(f"Task {self.request.id}: Failed to remove successfully processed file {path}: {cleanup_e}")
                 logger.info(f"Task {self.request.id}: STEP 6: Cleanup finished ({time.time()-cleanup_start:.2f}s). Removed {cleaned_count} files.")
            elif is_specific_addition:
                 logger.info(f"Task {self.request.id}: STEP 6: Skipping file cleanup for specific addition.")
            else:
                 logger.info(f"Task {self.request.id}: STEP 6: No processed files to cleanup.")

            # Update final status using model method
            chatbot.complete_index_operation(success=True, total_chunks=total_chunks_processed)
            # Note: complete_index_operation commits and pushes SSE

            return {'status': 'Success', 'chatbot_id': chatbot_id, 'chunks_processed': total_chunks_processed}

        except INGESTION_RETRYABLE_EXCEPTIONS as e:
            logger.warning(f"Task {self.request.id} (Attempt {self.request.retries + 1}/{self.max_retries + 1}): Retryable error encountered for Chatbot {chatbot_id}: {type(e).__name__} - {e}", exc_info=True)
            if self.request.retries < self.max_retries:
                user_choice = 'n' # Default to no retry if input fails
                try:
                    prompt_message = (
                        f"Task {self.request.id} (Attempt {self.request.retries + 1}/{self.max_retries + 1}) "
                        f"failed with {type(e).__name__}: {e}. Retry? (y/n): "
                    )
                    user_choice = input(prompt_message)
                except EOFError:
                    logger.warning(f"Task {self.request.id}: No TTY for input. Defaulting to no retry for {type(e).__name__}.")

                if user_choice.lower() == 'y':
                    logger.info(f"Task {self.request.id}: User chose to retry.")
                    if chatbot:
                        try:
                            chatbot.update_index_operation_status(
                                operation_id=self.request.id, state='RETRYING',
                                error=f"User approved retry for: {type(e).__name__}",
                                progress=chatbot.index_operation_progress
                            )
                            logger.info(f"Task {self.request.id}: Updated Chatbot {chatbot_id} status to RETRYING in DB.")
                        except Exception as db_err:
                            logger.error(f"Task {self.request.id}: FAILED to update Chatbot {chatbot_id} status to RETRYING: {db_err}", exc_info=True)
                            db.session.rollback()
                    raise self.retry(exc=e) # Celery handles backoff and jitter
                else:
                    logger.info(f"Task {self.request.id}: User chose not to retry for {type(e).__name__}. Failing task.")
                    raise e # Propagate to final failure handler
            else:
                logger.warning(f"Task {self.request.id}: Max retries reached for {type(e).__name__}. Failing task.")
                raise e # Propagate to final failure handler

        except HTTPError as http_err:
            status_code = http_err.response.status_code if http_err.response else 500
            logger.warning(f"Task {self.request.id} (Attempt {self.request.retries + 1}/{self.max_retries + 1}): HTTP error {status_code} for Chatbot {chatbot_id}", exc_info=True)
            if 500 <= status_code <= 599: # Only consider 5xx for this interactive retry
                if self.request.retries < self.max_retries:
                    user_choice = 'n' # Default
                    try:
                        prompt_message = (
                            f"Task {self.request.id} (Attempt {self.request.retries + 1}/{self.max_retries + 1}) "
                            f"failed with HTTP {status_code}. Retry? (y/n): "
                        )
                        user_choice = input(prompt_message)
                    except EOFError:
                        logger.warning(f"Task {self.request.id}: No TTY for input. Defaulting to no retry for HTTP {status_code}.")

                    if user_choice.lower() == 'y':
                        logger.info(f"Task {self.request.id}: User chose to retry HTTP {status_code}.")
                        if chatbot:
                            try:
                                chatbot.update_index_operation_status(
                                    operation_id=self.request.id, state='RETRYING',
                                    error=f"User approved retry for HTTP {status_code}",
                                    progress=chatbot.index_operation_progress
                                )
                                logger.info(f"Task {self.request.id}: Updated Chatbot {chatbot_id} status to RETRYING in DB.")
                            except Exception as db_err:
                                logger.error(f"Task {self.request.id}: FAILED to update Chatbot {chatbot_id} status to RETRYING: {db_err}", exc_info=True)
                                db.session.rollback()
                        raise self.retry(exc=http_err) # Celery handles backoff and jitter
                    else:
                        logger.info(f"Task {self.request.id}: User chose not to retry HTTP {status_code}. Failing task.")
                        raise http_err # Propagate to final failure handler
                else:
                    logger.warning(f"Task {self.request.id}: Max retries reached for HTTP {status_code}. Failing task.")
                    raise http_err # Propagate to final failure handler
            else: # Non-5xx HTTP errors
                logger.error(f"Task {self.request.id}: Non-retryable HTTP error {status_code}. Failing task.")
                raise http_err # Propagate to final failure handler

        except Exception as e:
            # --- Update DB Status to RETRYING ---
            if chatbot:
                try:
                    # Use 'RETRYING' state
                    chatbot.update_index_operation_status(
                        operation_id=self.request.id,
                        state='RETRYING',
                        error=f"Retrying due to: {type(e).__name__}",
                        progress=chatbot.index_operation_progress # Keep current progress
                    )
                    # Note: update_index_operation_status commits
                    logger.info(f"Task {self.request.id}: Updated Chatbot {chatbot_id} status to RETRYING in DB.")
                except Exception as db_err:
                    logger.error(f"Task {self.request.id}: FAILED to update Chatbot {chatbot_id} status to RETRYING: {db_err}", exc_info=True)
                    db.session.rollback() # Rollback if status update fails
            # --- Final Failure (Non-retryable or Max Retries Exceeded) ---
            logger.error(f"Task {self.request.id}: FINAL FAILURE for Chatbot {chatbot_id} after {self.request.retries} retries: {e}", exc_info=True)
            # Update final status using model method
            if chatbot:
                try:
                    # Format error with traceback
                    error_details = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                    chatbot.complete_index_operation(success=False, error=error_details[:1024]) # Limit error length
                    # Note: complete_index_operation commits and pushes SSE
                    logger.info(f"Task {self.request.id}: Updated Chatbot {chatbot_id} status to FAILED in DB.")
                except Exception as db_err:
                     logger.error(f"Task {self.request.id}: FAILED to update Chatbot {chatbot_id} status to FAILED: {db_err}", exc_info=True)
                     db.session.rollback()

            # Update Celery state for monitoring
            self.update_state(state='FAILURE', meta={'error': str(e), 'traceback': traceback.format_exc()})

            # --- Handle File Cleanup on Final Failure (if applicable) ---
            # Check if it was a full ingestion and files were processed
            is_specific_addition = 'urls_to_ingest' in source_details or 'files_to_ingest' in source_details
            # Need to determine if files were processed - this info isn't directly available here
            # We might need _perform_ingestion to return more status or check source_details
            files_were_processed_in_run = not is_specific_addition and source_details.get('files_uploaded')
            if files_were_processed_in_run:
                 logger.warning(f"Task {self.request.id}: Cleaning up uploaded files due to critical task failure during full ingestion.")
                 uploaded_file_basenames = source_details.get('files_uploaded', [])
                 if uploaded_file_basenames:
                     UPLOAD_FOLDER = 'uploads'
                     full_paths_to_cleanup = [os.path.join(UPLOAD_FOLDER, basename) for basename in uploaded_file_basenames if basename]
                     cleaned_count = 0
                     for path in full_paths_to_cleanup:
                         if path and os.path.exists(path):
                             try: os.remove(path); cleaned_count += 1
                             except Exception as cleanup_e: logger.warning(f"Failed remove {path}: {cleanup_e}")
                     logger.info(f"Task {self.request.id}: Cleanup finished ({cleaned_count} files).")

            # Reraise to ensure Celery marks it as failed if not handled by autoretry
            raise e

# --- END ingestion.py ---
