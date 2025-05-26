# app/services/rag_service.py

import hashlib
import os
import traceback
import json
import time
# import re # Removed as generate_rephrased_queries is removed
import functools
import uuid
import logging
from collections import defaultdict
import concurrent.futures
from google.api_core import exceptions as api_core_exceptions
from google.cloud import storage
from google.cloud import aiplatform
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import Namespace
from google.api_core.exceptions import GoogleAPICallError, NotFound as GoogleNotFound, ResourceExhausted
from flask import current_app
import vertexai
# from google.cloud import translate_v2 as translate # Removed as detection/translation is skipped here
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import select

# Vertex AI SDK imports
# from vertexai.language_models import TextEmbeddingModel # Replaced with google.genai
from google import genai # Added for new embedding model
from google.genai.types import EmbedContentConfig # Added for new embedding model
from vertexai.generative_models import (
    GenerativeModel,
    Part,
    GenerationConfig,
    FinishReason,
    HarmCategory,
    HarmBlockThreshold
)

# Application-specific imports
from app import db
from app.models import VectorIdMapping, Chatbot, ChatMessage, DetailedFeedback, UsageLog, User
from app.services import advanced_rag_service # Import moved to top level
# Safety settings for generative models
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
}

# --- Custom Exceptions for Deletion ---
class SourceNotFoundError(Exception):
    """Raised when the specified source identifier is not found for the chatbot."""
    pass

class VertexAIDeletionError(Exception):
    """Raised when deletion from Vertex AI fails."""
    pass

class DatabaseCleanupError(Exception):
    """Raised when database cleanup fails after successful Vertex AI deletion."""
    pass


class RagService:
    def __init__(self, logger=None): # Made logger optional for flexibility
        self.logger = logger if logger else logging.getLogger(__name__)
        self.storage_client = None
        self.bucket = None
        self.embedding_model = None # This will no longer store a model instance, genai_client will be used directly
        self.genai_client = None # Added for the new SDK
        self.generation_model = None
        # self.rephrase_model = None # Removed, not needed
        self.index_endpoint = None
        self.index_object = None
        # self.translate_client = None # Removed, not needed for this flow
        self.clients_initialized = False
        self.initialization_error = None
        self.index_resource_name = None
        self.deployed_index_id = None

    def _ensure_clients_initialized(self):
        """Ensure all clients are initialized before proceeding."""
        if self.clients_initialized:
            return True
        if not current_app:
            self.logger.error("RAG Service: Cannot initialize clients outside Flask app context.")
            self.initialization_error = "Application context not available."
            return False
        with current_app.app_context():
            return self._initialize_clients()

    def _initialize_clients(self):
        """Attempts to initialize all necessary Google Cloud clients."""
        if self.clients_initialized:
            return True

        self.logger.info("RAG Service: Attempting client initialization...")
        start_time = time.time()
        try:
            app_config = current_app.config
            project_id = app_config.get('PROJECT_ID')
            region = app_config.get('REGION')
            bucket_name = app_config.get('BUCKET_NAME')
            embedding_model_name = app_config.get('EMBEDDING_MODEL_NAME', "text-embedding-large-exp-03-07")
            generation_model_name = app_config.get('GENERATION_MODEL_NAME', "gemini-2.5-flash-preview-04-17")
            # rephrase_model_name = app_config.get('REPHRASE_MODEL_NAME', generation_model_name) # Removed
            index_endpoint_id = app_config.get('INDEX_ENDPOINT_ID')
            deployed_index_id = app_config.get('DEPLOYED_INDEX_ID')

            # --- Explicitly set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION ---
            # This ensures genai.Client() picks up the correct project/location,
            # overriding any potentially empty or incorrect values from the worker's environment.
            if project_id:
                os.environ['GOOGLE_CLOUD_PROJECT'] = project_id
                self.logger.info(f"RAG Service: Explicitly set GOOGLE_CLOUD_PROJECT env var to: {project_id}")
            else:
                self.logger.warning("RAG Service: PROJECT_ID from app config is empty. genai.Client() might fail if GOOGLE_CLOUD_PROJECT is not already correctly set.")
            if region:
                os.environ['GOOGLE_CLOUD_LOCATION'] = region
                self.logger.info(f"RAG Service: Explicitly set GOOGLE_CLOUD_LOCATION env var to: {region}")
            else:
                self.logger.warning("RAG Service: REGION from app config is empty. genai.Client() might fail if GOOGLE_CLOUD_LOCATION is not already correctly set.")
            # --------------------------------------------------------------------
            # --- Initialize Vertex AI ---
            self.logger.info(f"RAG Service: Initializing Vertex AI SDK (Project: {project_id}, Region: {region})...")
            vertexai.init(project=project_id, location=region)
            self.logger.info("RAG Service: Vertex AI SDK initialized.")

            # --- GCS Client ---
            self.logger.info(f"RAG Service: Initializing GCS Client for bucket '{bucket_name}'...")
            self.storage_client = storage.Client(project=project_id)
            self.bucket = self.storage_client.bucket(bucket_name)
            self.logger.info(f"RAG Service: GCS Client initialized for bucket '{bucket_name}'.")

            # --- Embedding Model (Switched to Google GenAI SDK) ---
            self.logger.info(f"RAG Service: Initializing Google GenAI Client for Embeddings (using model name: '{embedding_model_name}')...")
            # Ensure GOOGLE_GENAI_USE_VERTEXAI=True is set in the environment
            # Diagnostic log:
            env_var_value_rag = os.getenv('GOOGLE_GENAI_USE_VERTEXAI')
            self.logger.info(f"RAG Service: Value of GOOGLE_GENAI_USE_VERTEXAI before genai.Client(): {env_var_value_rag}")
            if env_var_value_rag != 'True':
                self.logger.warning("RAG Service: GOOGLE_GENAI_USE_VERTEXAI environment variable is NOT 'True' when genai.Client() is called. This is required for google-genai SDK to target Vertex AI.")


            self.genai_client = genai.Client()
            self.logger.info(f"RAG Service: Google GenAI Client initialized. Embedding model '{embedding_model_name}' will be called via this client.")
            # self.embedding_model is no longer used to store the model instance directly.

            # --- Generation Model ---
            self.logger.info(f"RAG Service: Initializing Generation Model '{generation_model_name}'...")
            self.generation_model = GenerativeModel(model_name=generation_model_name)
            self.logger.info(f"RAG Service: Generation Model '{generation_model_name}' OK.")

            # --- Rephrase Model Initialization Removed ---
            # self.rephrase_model = self.generation_model
            # self.logger.info(f"RAG Service: Using Generation Model '{generation_model_name}' for Rephrasing.")

            # --- Matching Engine Endpoint ---
            self.logger.info(f"RAG Service: Connecting to ME Endpoint '{index_endpoint_id}'...")
            endpoint_resource_name = f"projects/{project_id}/locations/{region}/indexEndpoints/{index_endpoint_id}"
            self.index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=endpoint_resource_name)

            # Verify endpoint and get deployed index info
            # ... (keep verification logic as is) ...
            deployed_indexes = self.index_endpoint.deployed_indexes
            if not deployed_indexes:
                raise RuntimeError(f"No deployed indexes found on endpoint {endpoint_resource_name}")

            target_deployed_index = None
            if deployed_index_id:
                for di in deployed_indexes:
                    if di.id == deployed_index_id:
                        target_deployed_index = di
                        break
                else: # Use else on the for loop
                    raise RuntimeError(f"Deployed index '{deployed_index_id}' not found on endpoint {endpoint_resource_name}. Found: {[d.id for d in deployed_indexes]}")
            elif len(deployed_indexes) == 1:
                 target_deployed_index = deployed_indexes[0]
                 self.logger.warning(f"DEPLOYED_INDEX_ID not specified in config, using the only deployed index found: {target_deployed_index.id}")
            else:
                 raise RuntimeError(f"Multiple deployed indexes found ({[di.id for di in deployed_indexes]}) but DEPLOYED_INDEX_ID not specified in config. Cannot determine which index to use.")

            self.deployed_index_id = target_deployed_index.id
            self.index_resource_name = target_deployed_index.index # Full resource name of the Index

            # Initialize index object
            # ... (keep index object initialization as is) ...
            self.logger.info(f"RAG Service: Initializing Index object for '{self.index_resource_name}'...")
            if not self.index_resource_name:
                 raise RuntimeError("Index resource name could not be determined during initialization.")
            self.index_object = aiplatform.MatchingEngineIndex(index_name=self.index_resource_name)
            self.logger.info(f"RAG Service: ME Endpoint client connected: {self.index_endpoint.resource_name}, Index object initialized: {self.index_object.resource_name}")

            # --- Translation Client Initialization Removed ---
            # self.logger.info("RAG Service: Initializing Google Translate Client...")
            # self.translate_client = translate.Client()
            # self.logger.info("RAG Service: Google Translate Client initialized.")

            self.clients_initialized = True
            self.logger.info(f"RAG Service: All clients initialized successfully ({time.time() - start_time:.2f}s).")
            return True

        except Exception as e:
            self.logger.error(f"RAG Service: FATAL - Client Initialization Failed: {e}", exc_info=True)
            self.initialization_error = str(e)
            self.clients_initialized = False
            raise e

    def generate_multiple_embeddings(self, queries: list):
        """Generates embeddings for a list of query texts."""
        # ... (keep this method exactly as is - it correctly handles batch_size=1) ...
        if not self._ensure_clients_initialized():
            return None, "Clients not initialized."

        self.logger.info(f"RAG Step 1b: Generate Embeddings for {len(queries)} queries")
        # Check for genai_client instead of embedding_model instance
        if not self.genai_client:
            self.logger.error("Google GenAI client not initialized.")
            return None, "Google GenAI client not initialized."
        if not queries:
            self.logger.warning("Query list is empty.")
            return [], None

        try:
            start_time = time.time()
            successful_embeddings = []
            num_requested = len(queries)
            num_generated = 0

            self.logger.info(f"Generating embeddings individually for {num_requested} queries...")

            for index, query in enumerate(queries):
                if not query or not query.strip():
                    self.logger.warning(f"Skipping empty query at index {index}.")
                    continue

                try:
                    single_query_list = [query]
                    # Use the genai_client and configured model name
                    current_embedding_model_name = current_app.config.get('EMBEDDING_MODEL_NAME')
                    task_type_for_embedding = "RETRIEVAL_QUERY" # Default for queries

                    api_response = self.genai_client.models.embed_content(
                        model=current_embedding_model_name,
                        contents=single_query_list,
                        config=EmbedContentConfig(
                            task_type=task_type_for_embedding,
                            # output_dimensionality=3072, # Optional: default for text-embedding-large-exp-03-07
                            # auto_truncate=False # Optional: default is True (silent truncation)
                        )
                    )

                    if api_response and api_response.embeddings:
                        embedding_values = api_response.embeddings[0].values
                        if embedding_values:
                            successful_embeddings.append(embedding_values)
                            num_generated += 1
                            self.logger.debug(f"Successfully generated embedding for query index {index}.")
                        else:
                            self.logger.warning(f"Received no/empty embedding values for query index {index}: '{query[:50]}...'")
                    else:
                        self.logger.error(f"Unexpected response structure from embed_content for query index {index}: '{query[:50]}...'. Response: {api_response}")

                except Exception as single_emb_error:
                    self.logger.error(f"Failed to generate embedding for query index {index}: '{query[:50]}...'. Error: {single_emb_error}", exc_info=True)

            duration = time.time() - start_time
            self.logger.info(f" -> Generated {num_generated} embeddings out of {num_requested} requested queries individually in {duration:.3f}s.")

            if not successful_embeddings:
                self.logger.error("Failed to generate any embeddings for the provided queries.")
                return [], "Failed to generate any embeddings."

            return successful_embeddings, None

        except GoogleAPICallError as e:
            self.logger.error(f" -> Error generating embeddings: {e}", exc_info=True)
            return None, f"Error generating embeddings: {e}"
        except Exception as e:
            self.logger.error(f" -> Unexpected error generating embeddings: {e}", exc_info=True)
            return None, f"Unexpected error generating embeddings: {e}"

    # --- generate_rephrased_queries METHOD REMOVED ---

    def generate_response(self, prompt: str, image_data: bytes = None, image_mime_type: str = None):
        """Generates a response using the LLM, potentially including image context."""
        # ... (keep this method exactly as is) ...
        if not self._ensure_clients_initialized():
            return None, "Clients not initialized.", None

        app_config = current_app.config
        self.logger.info(f"RAG Step 5: Generate Response (Image provided: {image_data is not None})")
        if not self.generation_model:
            self.logger.error("Generation model not initialized.")
            return None, "Generation model not initialized.", None

        generation_config = GenerationConfig(
            temperature=app_config.get('GENERATION_TEMPERATURE', 0.3),
            max_output_tokens=app_config.get('GENERATION_MAX_TOKENS', 512),
            top_p=0.95
        )

        content_parts = [prompt]
        if image_data and image_mime_type:
            self.logger.info(f" -> Including image ({image_mime_type}, {len(image_data)} bytes) in generation request.")
            image_part = Part.from_data(data=image_data, mime_type=image_mime_type)
            content_parts.append(image_part)

        try:
            start_time = time.time()
            response = self.generation_model.generate_content(
                contents=content_parts,
                generation_config=generation_config,
                safety_settings=SAFETY_SETTINGS,
                stream=False
            )
            duration = time.time() - start_time

            response_text = ""
            finish_reason = None
            token_counts = {}
            safety_ratings = []

            if response and response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    response_text = candidate.content.parts[0].text
                finish_reason = candidate.finish_reason
                if candidate.safety_ratings:
                    safety_ratings = [
                        {"category": rating.category.name, "probability": rating.probability.name}
                        for rating in candidate.safety_ratings
                    ]

                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    token_counts = {
                        'prompt_tokens': response.usage_metadata.prompt_token_count,
                        'candidates_token_count': response.usage_metadata.candidates_token_count,
                        'total_tokens': response.usage_metadata.total_token_count
                    }

            finish_reason_str = finish_reason.name if finish_reason else "UNKNOWN"
            self.logger.info(f" -> LLM generation finished. Reason: {finish_reason_str} (Time: {duration:.3f}s)")

            response_metadata = {
                "finish_reason": finish_reason_str,
                "token_counts": token_counts,
                "safety_ratings": safety_ratings
            }

            if finish_reason != FinishReason.STOP:
                self.logger.warning(f"LLM generation finished with non-STOP reason: {finish_reason_str}. Ratings: {safety_ratings}")
                error_message = f"Response generation issue: {finish_reason_str}."
                if finish_reason == FinishReason.SAFETY:
                    blocked_categories = [r['category'] for r in safety_ratings if r.get('probability') in ['HIGH', 'MEDIUM']]
                    error_message = f"Response blocked due to safety concerns ({', '.join(blocked_categories)})." if blocked_categories else "Response blocked due to safety concerns."
                    return None, error_message, response_metadata
                elif finish_reason == FinishReason.MAX_TOKENS:
                    error_message = "Response may be incomplete as the maximum length was reached."
                    return response_text.strip() + "...", error_message, response_metadata
                elif finish_reason == FinishReason.RECITATION:
                     error_message = "Response generation stopped to avoid potential recitation of copyrighted material."
                     return None, error_message, response_metadata
                else:
                    return None, error_message, response_metadata

            if not response_text and finish_reason == FinishReason.STOP:
                self.logger.warning("LLM generation finished with STOP but returned empty text.")
                return None, "The AI generated an empty response.", response_metadata

            return response_text.strip(), None, response_metadata

        except GoogleAPICallError as e:
            self.logger.error(f" -> Google API Error during generation: {e}", exc_info=True)
            return None, f"LLM API Error: {e}", None
        except Exception as e:
            self.logger.error(f" -> Unexpected error during LLM response generation: {e}", exc_info=True)
            return None, f"Unexpected error generating response: {e}", None

    # --- GCS Caching Helper ---
    # ... (keep this method exactly as is) ...
    @functools.lru_cache(maxsize=1024)
    def _download_chunk_from_gcs(self, blob_name: str):
        if not self._ensure_clients_initialized(): return None
        if not self.bucket: return None
        try:
            self.logger.debug(f" -> Cache MISS. Fetching from GCS: gs://{self.bucket.name}/{blob_name}")
            blob = self.bucket.blob(blob_name)
            return blob.download_as_text(encoding="utf-8")
        except GoogleNotFound:
            self.logger.warning(f" -> GCS blob NOT FOUND: gs://{self.bucket.name}/{blob_name}")
            return None
        except Exception as e:
            self.logger.error(f" -> GCS download error for '{blob_name}': {e}", exc_info=True)
            return None

    # --- _fetch_single_chunk_text ---
    # ... (keep this method exactly as is) ...
    # Updated to use chatbot_id and new vector ID/GCS path format
    def _fetch_single_chunk_text(self, vector_id: str, chatbot_id: int) -> tuple[bool, str | None]:
        self.logger.debug(f" -> Submitting fetch task for Vector ID: {vector_id}")
        try:
            # Expected format: chatbot_{chatbot_id}_source_{hash}_chunk_{index}
            parts = vector_id.split('_')
            # Need at least 6 parts: 'chatbot', id, 'source', hash, 'chunk', index
            if len(parts) < 6 or parts[0] != 'chatbot' or parts[2] != 'source' or parts[4] != 'chunk':
                 self.logger.warning(f" -> Invalid vector ID format, cannot parse GCS path: {vector_id}")
                 return False, None

            extracted_chatbot_id_str = parts[1]
            # Security check: Compare extracted chatbot_id with the one passed to the function
            if extracted_chatbot_id_str != str(chatbot_id):
                self.logger.error(f" -> SECURITY MISMATCH! Vector ID chatbot '{extracted_chatbot_id_str}' != Query chatbot '{chatbot_id}'. Vector ID: {vector_id}. Skipping!")
                return False, None

            hash_part = parts[3]
            chunk_index_part = parts[5]
            # Construct GCS path using the new format
            blob_name = f"chatbot_{extracted_chatbot_id_str}/source_{hash_part}/{chunk_index_part}.txt"
            chunk_text = self._download_chunk_from_gcs(blob_name)
            if chunk_text is not None:
                return True, chunk_text
            else:
                return False, None
        except Exception as e:
            self.logger.error(f" -> Unexpected error processing vector ID '{vector_id}' before GCS download: {e}", exc_info=True)
            return False, None

    # --- fetch_chunk_texts ---
    # ... (keep this method exactly as is) ...
    # Updated to accept chatbot_id instead of client_id
    def fetch_chunk_texts(self, vector_ids: list, chatbot_id: int):
        if not self._ensure_clients_initialized(): return [], "Clients not initialized."

        self.logger.info(f"RAG Step 3: Fetch Chunk Texts ({len(vector_ids)} IDs for Chatbot {chatbot_id})") # Log chatbot_id
        if not vector_ids: return [], None
        if not self.bucket: return [], "GCS bucket not initialized."

        retrieved_texts = []
        fetch_errors = 0
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Pass chatbot_id to _fetch_single_chunk_text
            future_to_vector_id = {executor.submit(self._fetch_single_chunk_text, vid, chatbot_id): vid for vid in vector_ids}
            for future in concurrent.futures.as_completed(future_to_vector_id):
                vector_id = future_to_vector_id[future]
                try:
                    success, text = future.result()
                    if success and text:
                        retrieved_texts.append(text)
                    elif not success:
                        fetch_errors += 1
                except Exception as exc:
                    self.logger.error(f" -> Exception retrieving result for vector ID '{vector_id}': {exc}", exc_info=True)
                    fetch_errors += 1

        duration = time.time() - start_time
        self.logger.info(f" -> Fetched text for {len(retrieved_texts)} chunks concurrently. Errors: {fetch_errors} (Time: {duration:.3f}s).")

        error_msg = None
        if fetch_errors > 0:
             error_msg = f"Failed to fetch text content for {fetch_errors} out of {len(vector_ids)} relevant document sections."
             if not retrieved_texts: # All failed
                  error_msg = f"Failed to fetch text content for any of the {len(vector_ids)} relevant document sections."

        return retrieved_texts, error_msg

    # --- retrieve_chunks_multi_query ---
    # ... (keep this method exactly as is) ...
    def retrieve_chunks_multi_query(self, query_embeddings: list, chatbot_id: int, client_id: str): # Added chatbot_id
        if not self._ensure_clients_initialized(): return [], "Clients not initialized."

        app_config = current_app.config
        rag_top_k = app_config.get('RAG_TOP_K', 5)
        # Updated log message to show chatbot_id
        self.logger.info(f"RAG Step 2: Retrieve Chunks (Multi-Query: {len(query_embeddings)}, Chatbot: {chatbot_id}, K per query: {rag_top_k})")
        if not self.index_endpoint or not self.deployed_index_id:
            self.logger.error("ME client/deployed ID not initialized.")
            return [], "ME client/deployed ID not initialized."
        if not query_embeddings:
            return [], None
        # Keep client_id check if needed elsewhere, but primary filter is chatbot_id
        if not chatbot_id:
             return [], "Chatbot ID missing."

        all_neighbors_with_distances = []
        total_api_time = 0
        retrieval_errors = 0
        futures = []

        # Filter by chatbot_id namespace
        chatbot_filter = [Namespace(name="chatbot_id", allow_tokens=[str(chatbot_id)])]
        self.logger.debug(f" -> Using Filter: namespace='chatbot_id', allow_tokens=['{chatbot_id}']")

        def _find_neighbors_task(embedding, query_index):
            self.logger.debug(f" -> Submitting query with embedding #{query_index+1}...")
            start_time = time.time()
            try:
                response = self.index_endpoint.find_neighbors(
                    deployed_index_id=self.deployed_index_id,
                    queries=[embedding],
                    num_neighbors=rag_top_k,
                    filter=chatbot_filter # Use the new filter
                )
                duration = time.time() - start_time
                if response and isinstance(response, list) and len(response) > 0 and isinstance(response[0], list):
                    neighbors = response[0]
                    self.logger.debug(f" -> Query #{query_index+1} retrieved {len(neighbors)} neighbors (Time: {duration:.3f}s).")
                    return neighbors, duration, True, query_index
                else:
                    self.logger.debug(f" -> Query #{query_index+1} found no neighbors (Time: {duration:.3f}s). Response: {response}")
                    return [], duration, True, query_index
            except (GoogleAPICallError, Exception) as e:
                duration = time.time() - start_time
                self.logger.error(f" -> Error retrieving chunks for query #{query_index+1}: {e}", exc_info=True)
                return [], duration, False, query_index

        max_workers = min(len(query_embeddings), 10)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i, embedding in enumerate(query_embeddings):
                futures.append(executor.submit(_find_neighbors_task, embedding, i))

            for future in concurrent.futures.as_completed(futures):
                try:
                    neighbors, duration, success, query_idx = future.result()
                    total_api_time += duration
                    if success:
                        for neighbor in neighbors:
                            if hasattr(neighbor, 'id') and hasattr(neighbor, 'distance'):
                                all_neighbors_with_distances.append((neighbor.id, neighbor.distance))
                            else:
                                self.logger.warning(f" -> Neighbor object missing id or distance: {neighbor}")
                    else:
                        retrieval_errors += 1
                except Exception as exc:
                    self.logger.error(f" -> Error processing result from concurrent task: {exc}", exc_info=True)
                    retrieval_errors += 1

        self.logger.info(f" -> Total neighbors found across all queries: {len(all_neighbors_with_distances)} (Errors: {retrieval_errors}, Total API time (summed): {total_api_time:.3f}s).")

        if retrieval_errors == len(query_embeddings) and query_embeddings:
            return [], "Failed to retrieve chunks for all query variations."

        if not all_neighbors_with_distances:
            self.logger.warning(" -> No neighbors found across any query variations.")
            return [], None

        min_distances = defaultdict(lambda: float('inf'))
        for chunk_id, distance in all_neighbors_with_distances:
            min_distances[chunk_id] = min(min_distances[chunk_id], distance)

        sorted_unique_chunk_ids = sorted(min_distances, key=min_distances.get)

        m_fused_chunks = app_config.get('M_FUSED_CHUNKS', 10)
        final_chunk_ids = sorted_unique_chunk_ids[:m_fused_chunks]

        self.logger.info(f" -> Selected top {len(final_chunk_ids)} unique chunks for context after fusion (Target M={m_fused_chunks}).")
        self.logger.debug(f" -> Final fused chunk IDs: {final_chunk_ids}")

        return final_chunk_ids, None

    # --- construct_prompt METHOD MODIFIED ---
    def construct_prompt(self, retrieved_texts: list, query: str, client_id: str, base_prompt: str = None, chat_history: list = None, knowledge_adherence_level: str = 'strict', is_image_only: bool = False):
        """Constructs the final prompt for the LLM."""
        # Removed query_language parameter as it's no longer used here for the prompt instruction
        if not self._ensure_clients_initialized(): return ""

        app_config = current_app.config
        # Removed language name mapping and calculation, as it's not used in the prompt template below
        self.logger.info(f"RAG Step 4: Construct Prompt (Adherence: {knowledge_adherence_level}, ImageOnly: {is_image_only})")

        history_str = ""
        if chat_history:
            temp_history = ""
            total_len = 0
            max_history_chars = app_config.get('MAX_HISTORY_CHARS', 1500)
            for msg in reversed(chat_history):
                role = "User" if msg.get('role') == 'user' else "Chatbot"
                content = msg.get('content', '')
                entry = f"{role}: {content}\n"
                entry_len = len(entry)
                if total_len + entry_len > max_history_chars:
                    remaining_space = max_history_chars - total_len
                    if remaining_space > len(role) + 5:
                         temp_history = f"{role}: {content[:remaining_space-len(role)-3]}...\n" + temp_history
                    break
                temp_history = entry + temp_history
                total_len += entry_len
            history_str = temp_history.strip()
            if history_str:
                 self.logger.debug(f" -> Included {len(history_str)} chars of history (limit: {max_history_chars}).")
                 history_str = f"Previous Conversation History:\n{history_str}\n\n"
            else:
                 self.logger.debug(" -> No chat history included.")

        system_instructions = ""
        if base_prompt:
             self.logger.info(" -> Using custom base prompt provided by chatbot settings.")
             # If base_prompt contains {query_language_name}, it will now cause an error unless you adapt it
             # For simplicity, assume base_prompt is adapted or doesn't need the language name
             try:
                 system_instructions = base_prompt.format(
                     client_id=client_id,
                     # query_language_name=query_language_name, # REMOVED
                     knowledge_adherence_level=knowledge_adherence_level
                 )
             except KeyError as ke:
                 self.logger.error(f"Error formatting custom base prompt. Missing key: {ke}. Falling back to default.")
                 base_prompt = None # Force fallback to default

        # Fallback to default if base_prompt is not set or formatting failed
        if not base_prompt:
             self.logger.info(" -> Using default system prompt template.")
             # --- MODIFIED PROMPT TEMPLATE ---
             DEFAULT_SYSTEM_PROMPT_TEMPLATE = f"""
You are a helpful AI assistant for the client identified as '{client_id}'.
Your primary goal is to answer the user's query using the provided context sections below as your main source of information.
Try to understand the user's intent, even if their wording isn't exact,PLEASE DONT BE LIKE A KEYWORD SEARCH , TRY TO UNDERSTAND WHAT THE USER WANTS WHATS HES ASKING , AND HOW YOU CAN HELP HIM THE CHUCK OF RETRIVED DATA U recived .
Use the given context to provide a relevant and accurate answer. While the context is your primary source, synthesize the information *from the context sections* to answer the user's question naturally.
If the context does not contain information relevant to the user's query intent, state that clearly. Do not invent answers if the information is not present.
Avoid phrases like "Based on the context provided..." unless it's necessary to clarify the source of information.
***IMPORTANT: translate your respones to much the same language as the User Query provided below, regardless of the context /retrived chuck  language.***

Knowledge Adherence Level: {knowledge_adherence_level}
- strict: Answer primarily from the context. If the answer isn't there or cannot be reasonably inferred, say so. Avoid external knowledge.
- moderate: Primarily use the context, but you may infer or combine context information logically. State if the answer is not directly in the context. Avoid external knowledge unless necessary for clarity.
- relaxed: Use the context as a primary source, but you can incorporate general knowledge if the context is insufficient or lacks detail. Clearly distinguish context-based info from external knowledge.
"""
             system_instructions = DEFAULT_SYSTEM_PROMPT_TEMPLATE
             # --- END MODIFIED PROMPT TEMPLATE ---

        prompt_parts = [system_instructions, history_str]

        if not is_image_only:
            max_context_chars = app_config.get('MAX_CONTEXT_CHARS', 8000) # Default to 8000 chars if not set
            context_str_parts = []
            current_chars = 0
            num_included = 0
            num_total = len(retrieved_texts)

            if retrieved_texts:
                for i, text in enumerate(retrieved_texts):
                    section_header = f"--- Context Section {i+1} ---\n"
                    section_content = text
                    section = section_header + section_content
                    # Estimate length including potential joining characters (\n\n)
                    section_len_estimate = len(section) + (2 if context_str_parts else 0)

                    if current_chars + section_len_estimate <= max_context_chars:
                        context_str_parts.append(section)
                        current_chars += section_len_estimate
                        num_included += 1
                    else:
                        self.logger.warning(f" -> Context truncated. Included {num_included}/{num_total} sections ({current_chars} chars) due to limit ({max_context_chars} chars).")
                        break # Stop adding more context

            if context_str_parts:
                 context_str = "\n\n".join(context_str_parts)
            else:
                 context_str = "No relevant context sections could be included within the size limit." if retrieved_texts else "No relevant context sections were found."

            prompt_parts.append(f"Context Sections:\n{context_str}\n")

        prompt_parts.append(f"User Query:\n{query}") # The LLM will see the query in its original language here

        final_prompt = "\n".join(prompt_parts).strip()
        self.logger.debug(f" -> Final Constructed Prompt (first 200 chars):\n{final_prompt[:200]}...")
        return final_prompt

    # --- execute_pipeline METHOD MODIFIED ---
    def execute_pipeline(self, query: str, chatbot_id: int, client_id: str, chat_history: list = None, query_language: str = None, image_data: bytes = None, image_mime_type: str = None):
        """
        Executes the full RAG pipeline.
        Returns a dictionary: {"answer": str, "sources": list, "metadata": dict, "error": str|None, "warnings": str|None}
        """
        request_id = str(uuid.uuid4())
        response_message_id = str(uuid.uuid4()) # Generate ID for the assistant's response
        self.logger.info(f"[ReqID: {request_id}] --- Starting RAG Pipeline ---")
        self.logger.info(f"[ReqID: {request_id}] Chatbot ID: {chatbot_id}, Client ID: {client_id}")
        self.logger.info(f"[ReqID: {request_id}] Original Text Query: '{query[:100] if query else 'N/A'}{'...' if query and len(query) > 100 else ''}'")
        self.logger.info(f"[ReqID: {request_id}] Image Provided: {'Yes (' + image_mime_type + ')' if image_data else 'No'}")
        self.logger.info(f"[ReqID: {request_id}] History Turns: {len(chat_history) if chat_history else 0}")
        self.logger.info(f"[ReqID: {request_id}] Requested Language Parameter: {query_language or 'Not Provided'}")

        pipeline_overall_start_time = time.time()
        final_result = { "answer": None, "sources": [], "metadata": {}, "error": None, "warnings": None }
        error_accumulator = [] 
        generation_metadata = {}

        # --- 0. Ensure Initialization & Validate Inputs ---
        step_start_time = time.time()
        if not self._ensure_clients_initialized():
            self.logger.error(f"[ReqID: {request_id}] Pipeline Failed: Service clients not initialized ({self.initialization_error}).")
            final_result["error"] = f"Service initialization failed: {self.initialization_error}"
            self.logger.info(f"[ReqID: {request_id}] PERF: Client Initialization Check took {time.time() - step_start_time:.4f} seconds (Failed).")
            self._log_usage(request_id, chatbot_id, client_id, query, None, [], time.time() - pipeline_overall_start_time, final_result["error"], 503, {})
            return final_result
        self.logger.info(f"[ReqID: {request_id}] PERF: Client Initialization Check took {time.time() - step_start_time:.4f} seconds.")

        if not query and not image_data:
             self.logger.warning(f"[ReqID: {request_id}] Pipeline Aborted: Both query text and image data are missing.")
             final_result["error"] = "Please provide a text question or upload an image."
             self._log_usage(request_id, chatbot_id, client_id, "[No Query]", None, [], time.time() - pipeline_overall_start_time, final_result["error"], 400, {})
             return final_result
        if not chatbot_id or not client_id:
             self.logger.error(f"[ReqID: {request_id}] Pipeline Aborted: Missing chatbot_id ({chatbot_id}) or client_id ({client_id}).")
             final_result["error"] = "Missing required chatbot identification."
             self._log_usage(request_id, chatbot_id, client_id, query, None, [], time.time() - pipeline_overall_start_time, final_result["error"], 400, {})
             return final_result

        # --- Get Chatbot Configuration ---
        step_start_time = time.time()
        try:
            chatbot = db.session.get(Chatbot, chatbot_id)
            if not chatbot:
                self.logger.error(f"[ReqID: {request_id}] Pipeline Error: Chatbot with ID {chatbot_id} not found.")
                final_result["error"] = "Chatbot configuration not found."
                self.logger.info(f"[ReqID: {request_id}] PERF: Chatbot Config Fetch took {time.time() - step_start_time:.4f} seconds (Failed).")
                self._log_usage(request_id, chatbot_id, client_id, query, None, [], time.time() - pipeline_overall_start_time, final_result["error"], 404, {})
                return final_result
            base_prompt_override = chatbot.base_prompt
            knowledge_adherence = chatbot.knowledge_adherence_level or 'strict'
            image_analysis_enabled = chatbot.image_analysis_enabled
            self.logger.info(f"[ReqID: {request_id}] Chatbot Config - Image Analysis: {image_analysis_enabled}, Rephrasing: Disabled")
            self.logger.info(f"[ReqID: {request_id}] PERF: Chatbot Config Fetch took {time.time() - step_start_time:.4f} seconds.")
        except SQLAlchemyError as db_err:
             self.logger.error(f"[ReqID: {request_id}] Pipeline Error: Database error fetching chatbot {chatbot_id}: {db_err}", exc_info=True)
             final_result["error"] = "Database error retrieving chatbot configuration."
             self.logger.info(f"[ReqID: {request_id}] PERF: Chatbot Config Fetch took {time.time() - step_start_time:.4f} seconds (DB Error).")
             self._log_usage(request_id, chatbot_id, client_id, query, None, [], time.time() - pipeline_overall_start_time, final_result["error"], 500, {})
             return final_result

        # --- Advanced RAG Check ---
        if chatbot.advanced_rag_enabled:
            step_start_time = time.time()
            self.logger.info(f"[ReqID: {request_id}] Chatbot {chatbot_id} has advanced RAG enabled. Routing to advanced service.")
            from . import advanced_rag_service 
            advanced_result = advanced_rag_service.process_advanced_query(
                query=query,
                chat_history=chat_history,
                chatbot_id=chatbot_id,
                session_id=client_id, 
                rag_service_instance=self 
            )
            self.logger.info(f"[ReqID: {request_id}] PERF: Advanced RAG Routing and Processing took {time.time() - step_start_time:.4f} seconds.")
            (adv_response_text, adv_sources, _, adv_error_message, adv_status_code, adv_metadata) = advanced_result
            final_result = {
                "answer": adv_response_text,
                "sources": adv_sources,
                "response_message_id": request_id, 
                "metadata": adv_metadata or {}
            }
            if adv_status_code != 200 and adv_error_message:
                final_result["error"] = adv_error_message
            # Advanced RAG service should handle its own _log_usage or return necessary info
            return final_result

        # --- 1. Image Text Extraction (if image provided and enabled) ---
        extracted_image_text = ""
        if image_data and image_analysis_enabled:
            step_start_time = time.time()
            self.logger.info(f"[ReqID: {request_id}] RAG Step 1: Extracting text/description from image...")
            try:
                image_part = Part.from_data(data=image_data, mime_type=image_mime_type)
                extraction_prompt = "Extract all text visible in this image. If no text is present, briefly describe the image's main subject and context."
                extraction_gen_config = GenerationConfig(max_output_tokens=500, temperature=0.1)
                extraction_response = self.generation_model.generate_content(
                    [extraction_prompt, image_part],
                    generation_config=extraction_gen_config,
                    safety_settings=SAFETY_SETTINGS
                )
                safety_blocked = False
                if hasattr(extraction_response, 'prompt_feedback') and extraction_response.prompt_feedback and extraction_response.prompt_feedback.safety_ratings:
                    safety_blocked = any(
                        rating.probability in [HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE, HarmBlockThreshold.BLOCK_ONLY_HIGH]
                        for rating in extraction_response.prompt_feedback.safety_ratings
                    )
                if safety_blocked:
                    self.logger.warning(f"[ReqID: {request_id}] Image extraction prompt blocked due to safety (Time: {time.time() - step_start_time:.3f}s)")
                    error_accumulator.append("Image processing prompt blocked due to safety concerns.")
                elif extraction_response.candidates:
                    img_candidate = extraction_response.candidates[0]
                    img_finish_reason = img_candidate.finish_reason
                    img_finish_reason_str = img_finish_reason.name if img_finish_reason else "UNKNOWN"
                    if img_finish_reason != FinishReason.STOP:
                        self.logger.warning(f"[ReqID: {request_id}] Image extraction finished with reason: {img_finish_reason_str} (Time: {time.time() - step_start_time:.3f}s)")
                        if img_finish_reason == FinishReason.SAFETY:
                            safety_ratings = [{"category": r.category.name, "probability": r.probability.name} for r in img_candidate.safety_ratings] if img_candidate.safety_ratings else []
                            self.logger.warning(f"[ReqID: {request_id}] -> Safety Ratings: {safety_ratings}")
                            error_accumulator.append("Image processing stopped due to safety settings.")
                    if img_finish_reason != FinishReason.SAFETY: # Check again, as it might be STOP but still have content
                        if img_candidate.content and img_candidate.content.parts:
                            extracted_image_text = img_candidate.content.parts[0].text.strip()
                            if extracted_image_text: self.logger.info(f"[ReqID: {request_id}] -> Extracted text/desc: '{extracted_image_text[:100]}...'")
                            else: self.logger.info(f"[ReqID: {request_id}] -> Image processed, but no text extracted (Reason: {img_finish_reason_str}).")
                        else: self.logger.warning(f"[ReqID: {request_id}] -> Could not extract text/desc (Reason: {img_finish_reason_str}, missing content parts).")
                else: self.logger.warning(f"[ReqID: {request_id}] -> Image extraction returned no candidates and prompt was not blocked.")
            except GoogleAPICallError as e:
                self.logger.error(f"[ReqID: {request_id}] -> API Error during image extraction: {e}", exc_info=True)
                error_accumulator.append(f"API error processing image: {e}")
            except Exception as e:
                self.logger.error(f"[ReqID: {request_id}] -> General Error during image extraction: {e}", exc_info=True)
                error_accumulator.append(f"Failed to process image: {e}")
            self.logger.info(f"[ReqID: {request_id}] PERF: Image Text Extraction took {time.time() - step_start_time:.4f} seconds.")
        elif image_data and not image_analysis_enabled:
             self.logger.warning(f"[ReqID: {request_id}] Image provided but image analysis is disabled for this chatbot. Ignoring image.")

        # --- 2. Determine RAG Query ---
        rag_query = extracted_image_text if extracted_image_text and not query else query # Prioritize user text query if both exist
        if not rag_query and extracted_image_text: # If only image text was available
            rag_query = extracted_image_text
            
        is_image_only_query = bool(image_data and not query and extracted_image_text) # True if query is based *solely* on image extraction

        if not rag_query: # Final check if we still don't have a query
             self.logger.error(f"[ReqID: {request_id}] No effective query available for RAG after considering image and text inputs.")
             final_result["error"] = "; ".join(error_accumulator) or "Could not determine a query from the provided input."
             self._log_usage(request_id, chatbot_id, client_id, "[No Effective Query]", None, [], time.time() - pipeline_overall_start_time, final_result["error"], 400, {})
             return final_result
        
        if extracted_image_text and query: self.logger.info(f"[ReqID: {request_id}] -> Using user text query ('{query[:50]}...') for RAG, extracted image text ('{extracted_image_text[:50]}...') might be used in prompt.")
        elif extracted_image_text: self.logger.info(f"[ReqID: {request_id}] -> Using extracted image text ('{rag_query[:50]}...') as the query for RAG.")
        else: self.logger.info(f"[ReqID: {request_id}] -> Using user-provided text query ('{rag_query[:50]}...') for RAG.")

        # --- 3. Language Detection & Translation (SKIPPED) ---
        self.logger.info(f"[ReqID: {request_id}] RAG Step 3: Skipping Language Detection & Translation.")
        query_for_embedding = rag_query 

        # --- 4. Generate Rephrased Queries (SKIPPED) ---
        self.logger.info(f"[ReqID: {request_id}] RAG Step 4a: Skipping Query Rephrasing.")
        all_queries_for_embedding = [query_for_embedding] 
        all_queries_for_embedding = [q for q in all_queries_for_embedding if q and q.strip()]


        # --- 5. Generate Embeddings ---
        query_embeddings = []
        if all_queries_for_embedding:
            step_start_time = time.time()
            self.logger.info(f"[ReqID: {request_id}] RAG Step 4b: Generate Embeddings for {len(all_queries_for_embedding)} queries")
            embeddings, emb_err = self.generate_multiple_embeddings(all_queries_for_embedding)
            self.logger.info(f"[ReqID: {request_id}] PERF: Embedding Generation took {time.time() - step_start_time:.4f} seconds.")
            if emb_err:
                self.logger.error(f"[ReqID: {request_id}] Pipeline Step Failed: Embedding Generation - {emb_err}")
                error_accumulator.append(f"Embedding generation failed: {emb_err}")
            elif embeddings:
                query_embeddings = embeddings
            else:
                 self.logger.error(f"[ReqID: {request_id}] Embedding generation returned no results and no error.")
                 error_accumulator.append("Embedding generation failed unexpectedly.")
        else:
             self.logger.warning(f"[ReqID: {request_id}] No valid queries available for embedding generation.")
             error_accumulator.append("No query available for embedding.")

        # --- 6. Retrieve Relevant Chunks ---
        retrieved_chunk_ids = []
        retrieved_texts = []
        sources = []
        if query_embeddings:
            step_start_time = time.time()
            self.logger.info(f"[ReqID: {request_id}] RAG Step 5: Retrieve Chunks")
            chunk_ids, ret_err = self.retrieve_chunks_multi_query(query_embeddings, chatbot_id, client_id)
            self.logger.info(f"[ReqID: {request_id}] PERF: Chunk Retrieval (Vector Search) took {time.time() - step_start_time:.4f} seconds.")
            if ret_err:
                self.logger.error(f"[ReqID: {request_id}] Pipeline Step Warning: Retrieval Failed - {ret_err}")
                error_accumulator.append(f"Chunk retrieval failed: {ret_err}")
            elif chunk_ids:
                retrieved_chunk_ids = chunk_ids
                # --- 7. Fetch Chunk Texts ---
                step_start_time_fetch = time.time()
                self.logger.info(f"[ReqID: {request_id}] RAG Step 6: Fetch Chunk Texts ({len(retrieved_chunk_ids)} IDs)")
                texts, fetch_err = self.fetch_chunk_texts(retrieved_chunk_ids, chatbot_id)
                self.logger.info(f"[ReqID: {request_id}] PERF: Fetch Chunk Texts (GCS) took {time.time() - step_start_time_fetch:.4f} seconds.")
                if fetch_err:
                    self.logger.warning(f"[ReqID: {request_id}] Pipeline Step Warning: Fetch Failed - {fetch_err}. Proceeding with partial/no context.")
                    error_accumulator.append(f"Chunk text fetching failed: {fetch_err}")
                    if texts: retrieved_texts = texts
                elif texts:
                    retrieved_texts = texts
                    # --- Map Vector IDs to Source Info ---
                    step_start_time_map = time.time()
                    try:
                         mappings = VectorIdMapping.query.filter(VectorIdMapping.vector_id.in_(retrieved_chunk_ids)).all()
                         source_map = {}
                         processed_sources = set()
                         for m in mappings:
                             if hasattr(m, 'source_identifier') and m.source_identifier:
                                 vector_id = m.vector_id
                                 identifier = m.source_identifier
                                 source_type = 'unknown'
                                 if identifier.startswith('file://'): source_type = 'file'
                                 elif identifier.startswith('http://') or identifier.startswith('https://'): source_type = 'web'
                                 source_info = {'type': source_type, 'identifier': identifier}
                                 source_map[vector_id] = source_info
                                 source_key = (source_type, identifier)
                                 if source_key not in processed_sources:
                                     sources.append(source_info)
                                     processed_sources.add(source_key)
                             else: self.logger.warning(f"[ReqID: {request_id}] Mapping object for vector_id {m.vector_id if hasattr(m, 'vector_id') else 'UNKNOWN'} missing source_identifier attribute or value.")
                         self.logger.info(f"[ReqID: {request_id}] -> Mapped {len(sources)} unique sources from {len(mappings)} vector ID mappings.")
                    except SQLAlchemyError as db_map_err:
                         self.logger.error(f"[ReqID: {request_id}] -> Database error mapping vector IDs to sources: {db_map_err}", exc_info=True)
                         error_accumulator.append("Failed to map retrieved chunks to sources.")
                    self.logger.info(f"[ReqID: {request_id}] PERF: Map Vector IDs to Sources took {time.time() - step_start_time_map:.4f} seconds.")
            else:
                 self.logger.info("[ReqID: {request_id}] -> No relevant chunks found after retrieval.")
        else:
             self.logger.warning("[ReqID: {request_id}] Skipping retrieval and fetch due to missing embeddings.")

        # --- 8. Construct Final Prompt ---
        step_start_time = time.time()
        # Use original query for the "User Query:" part of the prompt, and rag_query for context building if different
        prompt_user_query = query if query else extracted_image_text # The query as the user sees it or image content if no text query
        
        self.logger.info(f"[ReqID: {request_id}] DEBUG: rag_query for context: '{rag_query[:200]}{'...' if len(rag_query) > 200 else ''}'")
        self.logger.info(f"[ReqID: {request_id}] DEBUG: prompt_user_query for LLM: '{prompt_user_query[:200]}{'...' if len(prompt_user_query) > 200 else ''}'")
        self.logger.info(f"[ReqID: {request_id}] DEBUG: chat_history before construct_prompt: {chat_history}")
        self.logger.info(f"[ReqID: {request_id}] DEBUG: base_prompt_override before construct_prompt: {base_prompt_override}")
        self.logger.info(f"[ReqID: {request_id}] RAG Step 7: Construct Prompt")
        
        final_prompt = self.construct_prompt(
            retrieved_texts=retrieved_texts,
            query=prompt_user_query, # This is what the LLM sees as "User Query:"
            client_id=client_id,
            base_prompt=base_prompt_override,
            chat_history=chat_history,
            knowledge_adherence_level=knowledge_adherence,
            is_image_only=is_image_only_query # This flag indicates if context should be skipped
        )
        self.logger.info(f"[ReqID: {request_id}] PERF: Prompt Construction took {time.time() - step_start_time:.4f} seconds.")

        # --- 9. Generate Response ---
        step_start_time = time.time()
        self.logger.info(f"[ReqID: {request_id}] RAG Step 8: Generate Response")
        image_for_generation = None
        mime_for_generation = None
        if image_data and image_analysis_enabled:
            # Logic for when to resupply image to final LLM:
            # If RAG found no context (retrieved_texts is empty) AND the query was based on image extraction (is_image_only_query is True OR (extracted_image_text and not query))
            # OR if there was no text query at all and only an image was provided.
            should_resupply_image = False
            if not retrieved_texts:
                if is_image_only_query: # Query was purely from image, no context found
                    should_resupply_image = True
                elif extracted_image_text and not query: # Query was from image, no user text, no context found
                     should_resupply_image = True
            
            if image_data and not query and not extracted_image_text: # Only image, no text extracted, no RAG query
                should_resupply_image = True


            if should_resupply_image:
                self.logger.info(f"[ReqID: {request_id}] RAG found no context or query was image-based. Resupplying original image to LLM.")
                image_for_generation = image_data
                mime_for_generation = image_mime_type
            else:
                self.logger.info(f"[ReqID: {request_id}] RAG found context or query was text-based. NOT resupplying original image to LLM.")
        
        response_text, gen_err, generation_metadata = self.generate_response(
            prompt=final_prompt,
            image_data=image_for_generation,
            image_mime_type=mime_for_generation
        )
        self.logger.info(f"[ReqID: {request_id}] PERF: LLM Response Generation took {time.time() - step_start_time:.4f} seconds.")
        if gen_err:
            self.logger.error(f"[ReqID: {request_id}] Pipeline Step Failed: Generation - {gen_err}")
            final_result["error"] = "; ".join(error_accumulator + [f"Failed to generate response: {gen_err}"])
            self._log_usage(request_id, chatbot_id, client_id, query, None, sources, time.time() - pipeline_overall_start_time, final_result["error"], 500, generation_metadata or {})
            return final_result
        final_answer = response_text

        # --- 10. Final Translation (SKIPPED) ---
        self.logger.info(f"[ReqID: {request_id}] RAG Step 9: Skipping final translation step.")

        # --- 11. Final Result Assembly & Logging ---
        pipeline_duration = time.time() - pipeline_overall_start_time
        self.logger.info(f"[ReqID: {request_id}] --- RAG Pipeline END --- Total Duration: {pipeline_duration:.3f}s")
        final_result["answer"] = final_answer
        final_result["sources"] = sources
        final_result["metadata"] = generation_metadata or {}
        final_result["response_message_id"] = response_message_id 
        if error_accumulator:
             final_result["warnings"] = "; ".join(error_accumulator)
             self.logger.warning(f"[ReqID: {request_id}] Pipeline completed with warnings: {final_result['warnings']}")
        self._log_usage(request_id, chatbot_id, client_id, query, final_answer, sources, pipeline_duration, final_result.get("warnings"), 200, generation_metadata or {})
        return final_result


    def _log_usage(self, request_id: str, chatbot_id: int, client_id: str, query: str | None, response: str | None, sources: list, duration: float, error: str | None, status_code: int, metadata: dict):
         """Logs usage details to the database."""
         self.logger.debug(f"[ReqID: {request_id}] Logging usage to database...")
         try:
             user = User.query.filter_by(client_id=client_id).first()
             if not user:
                 self.logger.error(f"[ReqID: {request_id}] Could not find user with client_id {client_id} to log usage.")
                 return
             user_id = user.id
             truncated_query = (query[:4997] + '...') if query and len(query) > 5000 else query
             truncated_response = (response[:9997] + '...') if response and len(response) > 10000 else response
             log_entry = UsageLog(
                 user_id=user_id,
                 chatbot_id=chatbot_id,
                 action_type='query',
                 action_details=json.dumps({
                     'query': truncated_query,
                     'response': truncated_response,
                     'sources': sources,
                     'duration_ms': int(duration * 1000),
                     'status_code': status_code,
                     'error': error,
                     'metadata': metadata,
                     'request_id': request_id
                 }),
                 resource_usage=1,
             )
             db.session.add(log_entry)
             db.session.commit()
             self.logger.debug(f"[ReqID: {request_id}] Usage logged successfully.")
         except SQLAlchemyError as e:
             db.session.rollback()
             self.logger.error(f"[ReqID: {request_id}] Database error logging usage: {e}", exc_info=True)
         except Exception as e:
             db.session.rollback()
             self.logger.error(f"[ReqID: {request_id}] Unexpected error logging usage: {e}", exc_info=True)


    # --- Data Deletion Methods ---
    def cleanup_stale_data(self, vector_ids_to_delete: list, file_basenames_to_delete: list, chatbot_id: int):
        self.logger.warning("cleanup_stale_data not fully implemented in this example.")
        pass
    def delete_datapoints(self, datapoint_ids: list):
        if not self._ensure_clients_initialized(): raise VertexAIDeletionError(f"Client initialization failed: {self.initialization_error}")
        if not datapoint_ids: return
        self.logger.info(f"Attempting to delete {len(datapoint_ids)} datapoints from Vertex AI Index: {self.index_resource_name}...")
        if not self.index_object: raise VertexAIDeletionError("Vertex AI Index object not initialized.")
        try:
            self.index_object.remove_datapoints(datapoint_ids=datapoint_ids) 
            self.logger.info(f"Successfully submitted deletion request for {len(datapoint_ids)} datapoints.")
        except GoogleAPICallError as e:
            self.logger.error(f"Vertex AI API error deleting datapoints: {e}", exc_info=True)
            raise VertexAIDeletionError(f"Vertex AI API error: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error deleting datapoints from Vertex AI: {e}", exc_info=True)
            raise VertexAIDeletionError(f"Unexpected error: {e}")
    def delete_source_data(self, chatbot_id: int, source_identifier: str):
        """
        Deletes all data associated with a specific source identifier for a given chatbot.
        This includes vector embeddings in Vertex AI and corresponding database records.
        """
        self.logger.info(f"Attempting to delete data for source '{source_identifier}' from chatbot {chatbot_id}.")
        if not self._ensure_clients_initialized():
            return False, f"Client initialization failed: {self.initialization_error}"

        vector_ids_to_delete = []
        mappings_to_delete = []

        try:
            mappings_to_delete = VectorIdMapping.query.filter_by(
                chatbot_id=chatbot_id,
                source_identifier=source_identifier
            ).all()

            if not mappings_to_delete:
                self.logger.warning(f"No vector mappings found for chatbot {chatbot_id} and source '{source_identifier}'. Nothing to delete.")
                return True, f"No data found for source '{source_identifier}'."

            vector_ids_to_delete = [mapping.vector_id for mapping in mappings_to_delete]
            self.logger.info(f"Found {len(vector_ids_to_delete)} vector IDs to delete for source '{source_identifier}'.")

            db.session.begin_nested() 

            if vector_ids_to_delete:
                try:
                    self.delete_datapoints(vector_ids_to_delete)
                    self.logger.info(f"Successfully requested deletion of {len(vector_ids_to_delete)} vectors from Vertex AI.")
                except VertexAIDeletionError as e:
                    self.logger.error(f"Failed to delete vectors from Vertex AI for source '{source_identifier}': {e}")
                    db.session.rollback()
                    return False, f"Failed to delete vectors from Vertex AI: {e}"
                except Exception as e: 
                    self.logger.error(f"Unexpected error during Vertex AI deletion for source '{source_identifier}': {e}", exc_info=True)
                    db.session.rollback()
                    return False, f"Unexpected error during vector deletion: {e}"
            try:
                num_deleted = VectorIdMapping.query.filter(
                    VectorIdMapping.id.in_([m.id for m in mappings_to_delete])
                ).delete(synchronize_session=False) 

                if num_deleted != len(mappings_to_delete):
                     self.logger.warning(f"Expected to delete {len(mappings_to_delete)} mappings from DB, but deleted {num_deleted}.")

                self.logger.info(f"Successfully deleted {num_deleted} vector mapping records from database for source '{source_identifier}'.")
            except SQLAlchemyError as e:
                self.logger.error(f"Database error deleting vector mappings for source '{source_identifier}': {e}", exc_info=True)
                db.session.rollback()
                return False, f"Database error during cleanup: {e}"

            db.session.commit()
            self.logger.info(f"Successfully deleted all data for source '{source_identifier}' from chatbot {chatbot_id}.")
            return True, f"Source data for '{source_identifier}' deleted successfully."

        except SourceNotFoundError as e: 
             self.logger.error(f"Source not found error: {e}")
             return False, str(e)
        except SQLAlchemyError as e: 
            self.logger.error(f"Database error finding vector mappings for source '{source_identifier}': {e}", exc_info=True)
            db.session.rollback() 
            return False, f"Database error finding data: {e}"
        except Exception as e:
            self.logger.error(f"Unexpected error during source data deletion for '{source_identifier}': {e}", exc_info=True)
            db.session.rollback() 
            return False, f"An unexpected error occurred: {e}"


    def delete_chatbot_data(self, chatbot_id: int, user_id: int):
        """
        Deletes ALL data associated with a specific chatbot, including vectors,
        database mappings, chat history, feedback, and usage logs.
        Does NOT delete the Chatbot record itself.
        """
        self.logger.info(f"Attempting to delete ALL data for chatbot {chatbot_id} (owned by user {user_id}).")
        if not self._ensure_clients_initialized():
            return False, f"Client initialization failed: {self.initialization_error}"

        vector_ids_to_delete = []

        try:
            chatbot = db.session.get(Chatbot, chatbot_id)
            if not chatbot:
                 self.logger.warning(f"Chatbot {chatbot_id} not found. Assuming already deleted or invalid ID.")
                 return True, "Chatbot not found, no data to delete."
            if chatbot.user_id != user_id:
                 self.logger.error(f"User {user_id} does not own chatbot {chatbot_id}. Deletion forbidden.")
                 return False, "Permission denied: You do not own this chatbot."

            mappings = VectorIdMapping.query.filter_by(chatbot_id=chatbot_id).all()
            vector_ids_to_delete = [mapping.vector_id for mapping in mappings]
            self.logger.info(f"Found {len(vector_ids_to_delete)} vector IDs to delete for chatbot {chatbot_id}.")

            db.session.begin_nested() 

            if vector_ids_to_delete:
                try:
                    self.delete_datapoints(vector_ids_to_delete)
                    self.logger.info(f"Successfully requested deletion of {len(vector_ids_to_delete)} vectors from Vertex AI for chatbot {chatbot_id}.")
                except VertexAIDeletionError as e:
                    self.logger.error(f"Failed to delete vectors from Vertex AI for chatbot {chatbot_id}: {e}")
                    db.session.rollback()
                    return False, f"Failed to delete vectors from Vertex AI: {e}"
                except Exception as e: 
                    self.logger.error(f"Unexpected error during Vertex AI deletion for chatbot {chatbot_id}: {e}", exc_info=True)
                    db.session.rollback()
                    return False, f"Unexpected error during vector deletion: {e}"
            try:
                feedback_id_select = select(DetailedFeedback.id)\
                    .join(ChatMessage, DetailedFeedback.message_id == ChatMessage.id)\
                    .where(ChatMessage.chatbot_id == chatbot_id)
                num_feedback_deleted = db.session.query(DetailedFeedback)\
                    .filter(DetailedFeedback.id.in_(feedback_id_select))\
                    .delete(synchronize_session=False)
                self.logger.info(f"Deleted {num_feedback_deleted} detailed feedback records for chatbot {chatbot_id}.")

                num_messages_deleted = ChatMessage.query.filter_by(chatbot_id=chatbot_id).delete(synchronize_session=False)
                self.logger.info(f"Deleted {num_messages_deleted} chat message records for chatbot {chatbot_id}.")

                num_logs_deleted = UsageLog.query.filter_by(chatbot_id=chatbot_id).delete(synchronize_session=False)
                self.logger.info(f"Deleted {num_logs_deleted} usage log records for chatbot {chatbot_id}.")

                num_mappings_deleted = VectorIdMapping.query.filter_by(chatbot_id=chatbot_id).delete(synchronize_session=False)
                self.logger.info(f"Deleted {num_mappings_deleted} vector mapping records for chatbot {chatbot_id}.")

                chatbot.total_chunks_indexed = 0
                chatbot.last_index_update = None 
                chatbot.status = 'Empty' 
                db.session.add(chatbot) 
                self.logger.info(f"Reset index stats and status for chatbot {chatbot_id}.")

            except SQLAlchemyError as e:
                self.logger.error(f"Database error during cleanup for chatbot {chatbot_id}: {e}", exc_info=True)
                db.session.rollback()
                return False, f"Database error during cleanup: {e}"

            db.session.commit()
            self.logger.info(f"Successfully deleted all associated data for chatbot {chatbot_id}.")
            return True, "All chatbot data deleted successfully."

        except SQLAlchemyError as e: 
            self.logger.error(f"Database error during initial phase of chatbot data deletion for {chatbot_id}: {e}", exc_info=True)
            db.session.rollback() 
            return False, f"Database error preparing deletion: {e}"
        except Exception as e:
            self.logger.error(f"Unexpected error during chatbot data deletion for {chatbot_id}: {e}", exc_info=True)
            db.session.rollback() 
            return False, f"An unexpected error occurred: {e}"
