# chatbot-backend/app/services/advanced_rag_service.py
import logging
import re
import time # Ensure time is imported
import json
from typing import List, Tuple, Any, Dict, TYPE_CHECKING
from flask import current_app
from collections import defaultdict # Add this import
from google.api_core.exceptions import GoogleAPICallError, RetryError, DeadlineExceeded # Added specific API errors
from sqlalchemy.exc import SQLAlchemyError # Added for DB error handling
from cachetools import LFUCache, TTLCache # Added for BM25 caching

# --- Service Dependencies ---
if TYPE_CHECKING:
    from .rag_service import RAGService
# TYPE_CHECKING block for RAGService removed; local import used instead within process_advanced_query.
# Updated import to include VectorIdMapping
from app.models import Chatbot, db, VectorIdMapping
from sqlalchemy.orm import Session # Added Session for type hinting if needed
# Assuming TextEmbeddingModel is accessible via rag_service instance or needs specific import if used directly
# from vertexai.language_models import TextEmbeddingModel # Example if needed directly
from rank_bm25 import BM25Okapi # Added for Hybrid Search

from sentence_transformers import CrossEncoder # Added for re-ranking

# --- LLM Interaction ---
# [MEMORIZE]
# Library: google-cloud-aiplatform (specifically vertexai.generative_models)
# Version: 1.91.0 (Verified on 2025-05-01)
# Function: GenerativeModel.generate_content
# Key Parameters: contents, generation_config (takes GenerationConfig object for temp, max tokens etc.), safety_settings (takes dict/list defining harm thresholds).
# Source: https://cloud.google.com/python/docs/reference/aiplatform/latest/vertexai.generative_models.GenerativeModel#vertexai_generative_models_GenerativeModel_generate_content
# Corrected imports based on ValueError - using vertexai's internal types
# [MEMORIZE] The vertexai.generative_models.GenerativeModel requires HarmCategory/HarmBlockThreshold from google.cloud.aiplatform_v1.types, not google.genai.types. Source: ValueError traceback analysis. (Verified 2025-05-01).
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, Part # Keep these
# from google.genai.types import SafetySetting # Not used directly here
# Import safety types compatible with vertexai.generative_models
from google.cloud.aiplatform_v1.types import HarmCategory, SafetySetting
# HarmBlockThreshold is accessed via SafetySetting.HarmBlockThreshold

# Configure logging
logger = logging.getLogger(__name__)

# --- Default Safety Settings (used if not found in config) ---
# Define default safety settings dictionaries here to avoid repeating them in .get() calls
DEFAULT_QUERY_REPHRASING_SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

DEFAULT_RELAXED_JSON_SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

# --- Global Cache for BM25 ---
# Cache up to 100 BM25 indexes, expire after 1 hour (3600 seconds)
# [MEMORIZE] Using cachetools.LFUCache(maxsize=100, ttl=3600) for BM25 index caching. Key: f"bm25_index_{chatbot_id}". Stores (bm25_object, corpus_chunk_ids).
# TODO: Make maxsize and ttl configurable via Flask config, similar to CrossEncoder loading issues.
bm25_cache = TTLCache(maxsize=100, ttl=3600) # Using TTLCache to support both maxsize and ttl


# --- Advanced RAG Processor Class ---
class AdvancedRagProcessor:
    """
    Encapsulates the logic for advanced RAG processing, including LLM interactions,
    query decomposition, re-ranking, and context management. Initializes LLM models once.
    """
    def __init__(self):
        """Initializes the processor and loads necessary LLM and CrossEncoder models."""
        logger.info("Initializing AdvancedRagProcessor and loading models...")
        start_time = time.time()
        self.rephrasing_llm = None
        self.final_llm = None
        self.cross_encoder = None # Initialize cross_encoder attribute
        self.cross_encoder_model_name = None # Store name for logging

        try:
            # Initialize LLMs used for different tasks
            # [MEMORIZE] Initializing GenerativeModel instances once in AdvancedRagProcessor.__init__ for reuse.
            # Load model names from Flask config
            rephrasing_model_name = current_app.config.get('QUERY_REPHRASING_MODEL_NAME', "gemini-2.5-flash-preview-04-17") # Updated default
            final_response_model_name = current_app.config.get('FINAL_RESPONSE_MODEL_NAME', "gemini-2.5-flash-preview-04-17") # Updated default

            self.rephrasing_llm = GenerativeModel(rephrasing_model_name)
            self.final_llm = GenerativeModel(final_response_model_name)
            logger.info(f"Using Rephrasing LLM: {rephrasing_model_name}")
            logger.info(f"Using Final Response LLM: {final_response_model_name}")

            # Initialize CrossEncoder model
            self.cross_encoder_model_name = current_app.config.get('CROSS_ENCODER_MODEL_NAME', 'cross-encoder/ms-marco-TinyBERT-L-2-v2') # Corrected default name
            cross_encoder_max_length = current_app.config.get('CROSS_ENCODER_MAX_LENGTH', 512)

            # [MEMORIZE] Using sentence-transformers CrossEncoder for re-ranking. Initialized in AdvancedRagProcessor.__init__. Model fetched from config ('CROSS_ENCODER_MODEL_NAME', default 'cross-encoder/ms-marco-TinyBERT-L-2-v2'), max_length from config ('CROSS_ENCODER_MAX_LENGTH', default 512). Source: Hugging Face Model Hub / Flask Config. Verified 2025-05-01.
            self.cross_encoder = CrossEncoder(self.cross_encoder_model_name, max_length=cross_encoder_max_length)
            logger.info(f"Successfully loaded CrossEncoder model: {self.cross_encoder_model_name} with max_length={cross_encoder_max_length}")

            logger.info(f"All models initialized in {time.time() - start_time:.2f}s.")

        except Exception as e:
            logger.error(f"FATAL: Failed to initialize models in AdvancedRagProcessor: {e}", exc_info=True)
            # Ensure models are None if initialization fails partially or fully
            if not self.rephrasing_llm: logger.error("Rephrasing LLM failed to initialize.")
            if not self.final_llm: logger.error("Final LLM failed to initialize.")
            if not self.cross_encoder: logger.error(f"CrossEncoder model '{self.cross_encoder_model_name or 'N/A'}' failed to initialize.")
            # Raise a more specific error or handle based on which model failed if needed
            raise RuntimeError("Failed to initialize one or more core models for Advanced RAG.") from e

    def _estimate_token_count(self, text: str) -> int:
        """
        Estimates the token count of a given text using the model's tokenizer.
        Falls back to character approximation if the LLM call fails.
        """
        # [MEMORIZE] Token estimation now uses the Vertex AI GenerativeModel.count_tokens method for accuracy. Fallback is len(text)//4.
        if not self.rephrasing_llm:
            logger.warning("Rephrasing LLM not available for token counting. Falling back to approximation.")
            return len(text) // 4

        try:
            # Use the count_tokens method from one of the initialized models
            response = self.rephrasing_llm.count_tokens(contents=[text])
            # Ensure response and total_tokens exist
            if response and hasattr(response, 'total_tokens'):
                 logger.debug(f"Counted tokens for text (first 50 chars: '{text[:50]}...'): {response.total_tokens}")
                 return response.total_tokens
            else:
                 logger.warning(f"count_tokens response invalid: {response}. Falling back to approximation.")
                 return len(text) // 4
        except Exception as e:
            logger.error(f"Error calling count_tokens: {e}. Falling back to approximation.", exc_info=True)
            return len(text) // 4

    def _compress_context(self, context_string: str, target_token_limit: int, original_query: str) -> str:
        """
        Compresses the context string using an LLM if it exceeds the target token limit,
        focusing on relevance to the original query. Uses the pre-initialized final_llm.
        """
        estimated_tokens = self._estimate_token_count(context_string)
        logger.info(f"Estimated tokens in context before compression: {estimated_tokens}")

        # Get target token limit from config
        config_target_token_limit = current_app.config.get('CONTEXT_COMPRESSION_TARGET_TOKENS', 4000)

        if estimated_tokens <= config_target_token_limit:
            logger.info("Context is within token limit, no compression needed.")
            return context_string

        if not self.final_llm:
            logger.error("Compression LLM (final_llm) not initialized. Falling back to truncation.")
            # Fallback truncation length calculation
            fallback_chars = int(config_target_token_limit * current_app.config.get('CONTEXT_TRUNCATION_CHAR_MULTIPLIER', 5))
            return context_string[:fallback_chars]

        logger.info(f"Context exceeds limit ({estimated_tokens} > {config_target_token_limit}). Attempting compression...")
        start_time = time.time()

        prompt = f"""The following context has been retrieved to answer the query: "{original_query}"

Context:
---
{context_string}
---

The context is too long ({estimated_tokens} estimated tokens) and needs to be summarized to fit within approximately {config_target_token_limit} tokens.
Please summarize the context concisely, focusing *only* on the information most relevant to answering the original query: "{original_query}".
Preserve key details and source references (like "[Source X: ...]") if possible within the summary.
Output *only* the summarized context.
"""

        # Add specific error handling for the LLM call
        try:
            # Use the pre-initialized final response model
            # [MEMORIZE] Context compression uses LLM model defined in config ('FINAL_RESPONSE_MODEL_NAME').
            compression_temp = current_app.config.get('CONTEXT_COMPRESSION_TEMPERATURE', 0.3)
            # Calculate max_output_tokens based on config target + buffer
            compression_token_buffer = current_app.config.get('CONTEXT_COMPRESSION_TOKEN_BUFFER', 500)
            compression_max_tokens = config_target_token_limit + compression_token_buffer
            compression_safety_settings = current_app.config.get('FINAL_RESPONSE_SAFETY_SETTINGS', DEFAULT_QUERY_REPHRASING_SAFETY_SETTINGS) # Reuse default if specific not set

            generation_config = GenerationConfig(
                temperature=compression_temp,
                max_output_tokens=compression_max_tokens,
            )

            response = self.final_llm.generate_content(
                contents=[prompt],
                generation_config=generation_config,
                safety_settings=compression_safety_settings,
                stream=False,
            )

            if response and response.candidates and response.candidates[0].content.parts:
                summarized_context = response.candidates[0].content.parts[0].text
                summarized_tokens = self._estimate_token_count(summarized_context)
                logger.info(f"Context compressed successfully in {time.time() - start_time:.2f}s. New estimated tokens: {summarized_tokens}")
                # [MEMORIZE] Context compression triggers when estimated tokens exceed config CONTEXT_COMPRESSION_TARGET_TOKENS. Fallback is truncation using config CONTEXT_TRUNCATION_CHAR_MULTIPLIER.
                return summarized_context
            else:
                logger.warning(f"LLM response for context compression was empty or invalid. Falling back to truncation. Response: {response}")
                # Fallback: Truncate the original context (be generous with length)
                fallback_chars = int(config_target_token_limit * current_app.config.get('CONTEXT_TRUNCATION_CHAR_MULTIPLIER', 5))
                return context_string[:fallback_chars]

        except (GoogleAPICallError, RetryError, DeadlineExceeded) as e:
             logger.error(f"API Error during context compression LLM call: {e}", exc_info=True)
             # Fallback: Truncate the original context
             fallback_chars = int(config_target_token_limit * current_app.config.get('CONTEXT_TRUNCATION_CHAR_MULTIPLIER', 5))
             return context_string[:fallback_chars]
        except Exception as e: # Catch other unexpected errors
            logger.error(f"Unexpected error during context compression LLM call: {e}", exc_info=True)
            # Fallback: Truncate the original context
            fallback_chars = int(config_target_token_limit * current_app.config.get('CONTEXT_TRUNCATION_CHAR_MULTIPLIER', 5))
            return context_string[:fallback_chars]


    def _format_context(self, chunks: List[Dict]) -> str:
        """
        Formats the selected context chunks into a single string for the LLM prompt.
        """
        formatted_context = ""
        if not chunks:
            return "No context available."

        for i, chunk in enumerate(chunks):
            text = chunk.get('text', 'Missing text')
            metadata = chunk.get('metadata', {})
            source = metadata.get('source', 'Unknown source') # Assuming 'source' key in metadata
            # Ensure source is treated as a string
            source_str = str(source) if source is not None else 'Unknown source'
            formatted_context += f"[Source {i+1}: {source_str}]\n{text}\n---\n"

        return formatted_context.strip()


    def _generate_query_variations(self, original_query: str, chat_history: list) -> list[str]:
        """
        Generates diverse rephrasings of the original query using the pre-initialized rephrasing_llm,
        considering chat history for context.
        """
        start_time = time.time()
        logger.info(f"Generating query variations for: '{original_query[:100]}...'")

        if not self.rephrasing_llm:
            logger.error("Rephrasing LLM not initialized. Falling back to original query.")
            return [original_query]

        # Format chat history for the prompt
        formatted_history = "\n".join([f"{turn['role']}: {turn['content']}" for turn in chat_history])

        prompt = f"""Given the following chat history and the latest user query, generate 3-5 diverse rephrasings or expansions of the original query. Focus on capturing different facets or underlying intents of the query, considering the conversation context. Output *only* the rephrased queries, each on a new line, without any preamble or numbering.
Include the original query itself in the output list.

Chat History:
---
{formatted_history}
---

Original Query: "{original_query}"

Rephrased Queries (including original):"""

        # Add specific error handling for the LLM call
        try:
            # Load settings from config
            rephrasing_temp = current_app.config.get('QUERY_REPHRASING_TEMPERATURE', 0.7)
            rephrasing_max_tokens = current_app.config.get('QUERY_REPHRASING_MAX_TOKENS', 150)
            rephrasing_safety_settings = current_app.config.get('QUERY_REPHRASING_SAFETY_SETTINGS', DEFAULT_QUERY_REPHRASING_SAFETY_SETTINGS)

            generation_config = GenerationConfig(
                temperature=rephrasing_temp,
                max_output_tokens=rephrasing_max_tokens,
            )

            response = self.rephrasing_llm.generate_content(
                contents=[prompt], # Pass prompt as contents
                generation_config=generation_config,
                safety_settings=rephrasing_safety_settings,
                stream=False,
            )

            # Extract text and parse variations
            if response and response.candidates and response.candidates[0].content.parts:
                generated_text = response.candidates[0].content.parts[0].text
                # Split by newline and filter out empty strings
                variations = [q.strip() for q in generated_text.split('\n') if q.strip()]
                # Ensure original query is included, even if LLM forgets
                if original_query not in variations:
                    variations.insert(0, original_query)
                logger.info(f"Generated {len(variations)} query variations in {time.time() - start_time:.2f}s.")
                return variations
            else:
                logger.warning(f"LLM response for query rephrasing was empty or invalid. Response: {response}")
                return [original_query]

        except (GoogleAPICallError, RetryError, DeadlineExceeded) as e:
             logger.error(f"API Error generating query variations: {e}", exc_info=True)
             return [original_query] # Fallback to original query on error
        except Exception as e: # Catch other unexpected errors
            logger.error(f"Unexpected error generating query variations: {e}", exc_info=True)
            return [original_query] # Fallback to original query on error


    def _rerank_chunks(self, original_query: str, chunks: List[Dict]) -> List[Dict]:
        """
        Re-ranks retrieved chunks based on relevance to the original query using a CrossEncoder model.
        Uses the instance's cross_encoder model.
        """
        logger.info(f"Starting re-ranking for {len(chunks)} chunks...")
        start_time = time.time()

        if not chunks:
            logger.info("No chunks to re-rank.")
            return []

        # Use the instance's cross_encoder
        if self.cross_encoder is None:
            # Use the instance's stored name for the warning message
            model_name_for_warning = self.cross_encoder_model_name or 'default (config access failed or init failed)'
            logger.warning(f"CrossEncoder model '{model_name_for_warning}' not loaded/initialized. Skipping re-ranking.")
            return chunks

        try:
            # Prepare input pairs: (query, chunk_text)
            # Ensure chunk text is retrieved safely, defaulting to empty string if 'text' key is missing
            model_input = [(original_query, chunk.get('text', '')) for chunk in chunks]

            # Predict scores using the instance's model
            scores = self.cross_encoder.predict(model_input, show_progress_bar=False) # Disable progress bar for cleaner logs

            # Combine scores with original chunks
            # Use list() to ensure it's a modifiable list if needed later
            chunks_with_scores = list(zip(scores, chunks))

            # Sort chunks based on scores (descending)
            sorted_chunks_with_scores = sorted(chunks_with_scores, key=lambda x: x[0], reverse=True)

            # Extract sorted chunks
            reranked_chunks = [chunk for score, chunk in sorted_chunks_with_scores]

            duration = time.time() - start_time
            logger.info(f"CrossEncoder re-ranking finished in {duration:.2f}s. Resulting chunks: {len(reranked_chunks)}")
            return reranked_chunks

        except Exception as e:
            logger.error(f"Error during CrossEncoder prediction/re-ranking: {e}", exc_info=True)
            return chunks # Fallback to original order on error


    def _clean_json_string(self, raw_string: str) -> str:
        """Removes markdown fences and strips whitespace from a string potentially containing JSON."""
        if not isinstance(raw_string, str):
            return ""
        # Remove markdown fences (optional 'json' language tag)
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_string, re.DOTALL | re.IGNORECASE)
        if match:
            cleaned = match.group(1)
        else:
            cleaned = raw_string
        # Strip leading/trailing whitespace
        return cleaned.strip()

    def _call_llm_with_retry_and_parse_json(
        self,
        model: GenerativeModel, # Pass the specific model instance to use
        prompt: str,
        generation_config: GenerationConfig,
        safety_settings: dict,
        expected_keys: list = None, # Optional: List of keys expected in the root level
        expected_types: dict = None, # Optional: Dict of key -> expected type
        max_retries: int = None, # Default to None, will be fetched from config
        retry_delay: int = None, # Default to None, will be fetched from config
        fallback_value: Any = None
    ) -> Any:
        # Fetch retry parameters from config with defaults
        config_max_retries = current_app.config.get('LLM_JSON_MAX_RETRIES', 2) if max_retries is None else max_retries
        config_retry_delay = current_app.config.get('LLM_JSON_RETRY_DELAY', 1) if retry_delay is None else retry_delay
        """
        Calls the LLM, cleans the response, parses JSON, validates structure, and retries on specific errors.
        """
        if not model:
             logger.error(f"LLM model provided to _call_llm_with_retry_and_parse_json is None. Cannot proceed.")
             return fallback_value

        attempts = 0
        last_exception = None
        while attempts <= config_max_retries:
            attempts += 1
            try:
                logger.debug(f"LLM JSON Call Attempt {attempts}/{config_max_retries + 1}")
                response = model.generate_content(
                    contents=[prompt],
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    stream=False,
                )

                if not (response and response.candidates and response.candidates[0].content.parts):
                    logger.warning(f"LLM response was empty or invalid structure on attempt {attempts}. Response: {response}")
                    last_exception = ValueError("LLM response empty or invalid structure")
                    if attempts <= config_max_retries: time.sleep(config_retry_delay)
                    continue # Retry

                raw_text = response.candidates[0].content.parts[0].text
                cleaned_text = self._clean_json_string(raw_text)

                if not cleaned_text:
                     logger.warning(f"LLM response was empty after cleaning on attempt {attempts}. Raw: '{raw_text}'")
                     last_exception = ValueError("LLM response empty after cleaning")
                     if attempts <= config_max_retries: time.sleep(config_retry_delay)
                     continue # Retry

                try:
                    parsed_json = json.loads(cleaned_text)

                    # --- Structure Validation ---
                    if expected_keys:
                        if not all(key in parsed_json for key in expected_keys):
                            missing_keys = [k for k in expected_keys if k not in parsed_json]
                            logger.warning(f"Parsed JSON missing expected keys: {missing_keys} on attempt {attempts}. JSON: {parsed_json}")
                            last_exception = ValueError(f"Parsed JSON missing expected keys: {missing_keys}")
                            if attempts <= config_max_retries: time.sleep(config_retry_delay)
                            continue # Retry

                    if expected_types:
                        type_errors = []
                        for key, expected_type in expected_types.items():
                            if key in parsed_json and not isinstance(parsed_json[key], expected_type):
                                type_errors.append(f"Key '{key}' expected type {expected_type}, got {type(parsed_json[key])}")
                        if type_errors:
                             logger.warning(f"Parsed JSON type validation failed: {type_errors} on attempt {attempts}. JSON: {parsed_json}")
                             last_exception = TypeError(f"Parsed JSON type validation failed: {'; '.join(type_errors)}")
                             if attempts <= config_max_retries: time.sleep(config_retry_delay)
                             continue # Retry

                    # --- Success ---
                    logger.debug(f"LLM JSON call successful on attempt {attempts}.")
                    return parsed_json

                except json.JSONDecodeError as e:
                    logger.warning(f"JSONDecodeError on attempt {attempts}: {e}. Cleaned text: '{cleaned_text}'")
                    last_exception = e
                    if attempts <= config_max_retries: time.sleep(config_retry_delay)
                    # Continue to retry loop

            # Catch specific, potentially retryable API errors
            except (GoogleAPICallError, RetryError, DeadlineExceeded) as e:
                logger.warning(f"API Error on LLM JSON call attempt {attempts}: {type(e).__name__} - {e}")
                last_exception = e
                if attempts <= config_max_retries:
                    logger.info(f"Retrying in {config_retry_delay}s...")
                    time.sleep(config_retry_delay)
                # Continue to retry loop

            except Exception as e:
                logger.error(f"Unexpected error during LLM JSON call attempt {attempts}: {e}", exc_info=True)
                last_exception = e
                # Break on unexpected errors rather than retrying indefinitely
                break

        logger.error(f"LLM JSON call failed after {attempts} attempts. Last exception: {last_exception}")
        return fallback_value


    def _decompose_query(self, original_query: str, chat_history: list) -> list[str]:
        """
        Decomposes a potentially complex query into simpler, self-contained sub-questions
        using the pre-initialized rephrasing_llm, considering chat history. Retries on failure.
        """
        start_time = time.time()
        logger.info("Decomposing query...")

        if not self.rephrasing_llm:
            logger.error("Decomposition LLM (rephrasing_llm) not initialized. Falling back to original query.")
            return [original_query]

        formatted_history = "\n".join([f"{turn['role']}: {turn['content']}" for turn in chat_history])
        prompt = f"""Analyze the 'Original Query' in the context of the 'Chat History'.
Break it down into one or more simpler, self-contained sub-questions that can be answered independently to fully address the original query.
If the original query is already simple and self-contained, just return the original query as a single item in the list.

Output the results as a JSON list of strings. Example: ["sub-question 1", "sub-question 2"] or ["original query"]

Chat History:
---
{formatted_history}
---

Original Query: "{original_query}"

JSON Output:"""

        # Load settings from config
        decomp_temp = current_app.config.get('QUERY_DECOMPOSITION_TEMPERATURE', 0.7) # Reuse rephrasing temp default
        decomp_max_tokens = current_app.config.get('QUERY_DECOMPOSITION_MAX_TOKENS', 150) # Reuse rephrasing max tokens default
        decomp_safety_settings = current_app.config.get('RELAXED_JSON_SAFETY_SETTINGS', DEFAULT_RELAXED_JSON_SAFETY_SETTINGS) # Use relaxed default

        generation_config = GenerationConfig(
            temperature=decomp_temp,
            max_output_tokens=decomp_max_tokens,
            response_mime_type="application/json",
        )

        # Define expected structure for validation
        # We expect a list, but the helper validates dicts. We'll validate the list type after the call.
        # For the helper, we don't specify expected_keys or expected_types for the root list itself.
        parsed_result = self._call_llm_with_retry_and_parse_json(
            model=self.rephrasing_llm, # Use the instance variable
            prompt=prompt,
            generation_config=generation_config,
            safety_settings=decomp_safety_settings,
            fallback_value=None # Fallback handled below
        )

        # Post-call validation for list type
        if isinstance(parsed_result, list) and all(isinstance(q, str) for q in parsed_result) and parsed_result:
            logger.info(f"Decomposed query into {len(parsed_result)} sub-questions in {time.time() - start_time:.2f}s.")
            return parsed_result
        else:
            logger.warning(f"Decomposition failed or returned invalid structure after retries. Result: {parsed_result}. Falling back to original query.")
            return [original_query]


    def _recognize_intent_and_slots(self, query: str, chat_history: list) -> dict:
        """
        Recognizes the primary user intent and extracts key slots/entities from the query
        using the pre-initialized rephrasing_llm, considering chat history. Retries on failure.
        """
        start_time = time.time()
        logger.info("Recognizing intent and slots...")

        if not self.rephrasing_llm:
            logger.error("Intent/Slot LLM (rephrasing_llm) not initialized. Returning empty dict.")
            return {"intent": "error", "slots": {}}

        formatted_history = "\n".join([f"{turn['role']}: {turn['content']}" for turn in chat_history])
        prompt = f"""Analyze the 'Original Query' in the context of the 'Chat History'.
Identify the primary user intent (e.g., 'information_seeking', 'comparison', 'greeting', 'request_action', 'clarification', 'other').
Extract key named entities or slots relevant to the query (e.g., product names, features, locations, dates).

Output the results as a single JSON object with two keys: "intent" (string) and "slots" (object).
Example: {{"intent": "comparison", "slots": {{"product_a": "XYZ", "product_b": "ABC", "feature": "battery life"}}}}
If no specific slots are identified, return an empty object for "slots": {{"intent": "greeting", "slots": {{}}}}

Chat History:
---
{formatted_history}
---

Original Query: "{query}"

JSON Output:"""

        # Load settings from config
        intent_temp = current_app.config.get('INTENT_SLOT_TEMPERATURE', 0.2)
        intent_max_tokens = current_app.config.get('INTENT_SLOT_MAX_TOKENS', 200)
        intent_safety_settings = current_app.config.get('RELAXED_JSON_SAFETY_SETTINGS', DEFAULT_RELAXED_JSON_SAFETY_SETTINGS) # Use relaxed default

        generation_config = GenerationConfig(
            temperature=intent_temp,
            max_output_tokens=intent_max_tokens,
            response_mime_type="application/json",
        )

        # Define expected structure for validation
        expected_keys = ["intent", "slots"]
        expected_types = {"intent": str, "slots": dict}
        fallback = {"intent": "unknown", "slots": {}}

        parsed_result = self._call_llm_with_retry_and_parse_json(
            model=self.rephrasing_llm, # Use the instance variable
            prompt=prompt,
            generation_config=generation_config,
            safety_settings=intent_safety_settings,
            expected_keys=expected_keys,
            expected_types=expected_types,
            fallback_value=fallback
        )

        logger.info(f"Intent/Slot recognition finished in {time.time() - start_time:.2f}s. Result: {parsed_result}")
        return parsed_result


    def _analyze_retrieval_and_generate_followups(self, original_query: str, sub_questions: list, retrieved_chunks: List[Dict]) -> dict:
        """
        Analyzes the retrieved context against the query/sub-questions to determine sufficiency
        and generates potential follow-up questions using the pre-initialized rephrasing_llm.
        """
        start_time = time.time()
        logger.info("Analyzing retrieval sufficiency and generating follow-ups...")

        if not self.rephrasing_llm:
            logger.error("Analysis/Followup LLM (rephrasing_llm) not initialized. Returning default insufficient.")
            return {"sufficient": False, "follow_ups": []}

        context_preview = "\n---\n".join([chunk.get('text', '')[:200] + "..." for chunk in retrieved_chunks[:3]]) # Preview first few chunks
        sub_questions_str = "\n".join([f"- {q}" for q in sub_questions])

        prompt = f"""Given the original user query, the sub-questions derived from it, and a preview of the retrieved context, analyze if the context likely contains enough information to fully answer *all* the sub-questions.
Then, generate 1-3 potential follow-up questions the user might ask next, based on the original query and the context provided.

Original Query: "{original_query}"

Sub-questions derived:
{sub_questions_str}

Retrieved Context Preview:
---
{context_preview}
---

Analysis Task:
1. Sufficiency: Based *only* on the preview, is it likely the full retrieved context can answer *all* the sub-questions? Answer true or false.
2. Follow-up Questions: Generate 1-3 concise follow-up questions a user might ask next, related to the original query or the provided context. If unsure, provide an empty list.

Output the results as a single JSON object with two keys: "sufficient" (boolean) and "follow_ups" (list of strings).
Example: {{"sufficient": true, "follow_ups": ["What are the side effects?", "How does it compare to product Y?"]}}
"""

        # Load settings from config
        followup_temp = current_app.config.get('FOLLOWUP_TEMPERATURE', 0.6)
        followup_max_tokens = current_app.config.get('FOLLOWUP_MAX_TOKENS', 300)
        followup_safety_settings = current_app.config.get('RELAXED_JSON_SAFETY_SETTINGS', DEFAULT_RELAXED_JSON_SAFETY_SETTINGS) # Use relaxed default

        generation_config = GenerationConfig(
            temperature=followup_temp,
            max_output_tokens=followup_max_tokens,
            response_mime_type="application/json",
        )

        # Define expected structure for validation
        expected_keys = ["sufficient", "follow_ups"]
        expected_types = {"sufficient": bool, "follow_ups": list}
        fallback = {"sufficient": False, "follow_ups": []} # Default to insufficient if analysis fails

        parsed_result = self._call_llm_with_retry_and_parse_json(
            model=self.rephrasing_llm, # Use the instance variable
            prompt=prompt,
            generation_config=generation_config,
            safety_settings=followup_safety_settings,
            expected_keys=expected_keys,
            expected_types=expected_types,
            fallback_value=fallback
        )

        # Further validation for the list items in follow_ups
        if not isinstance(parsed_result.get("follow_ups"), list) or not all(isinstance(q, str) for q in parsed_result.get("follow_ups", [])):
             logger.warning(f"Follow-up questions list is not valid: {parsed_result.get('follow_ups')}. Resetting to empty list.")
             parsed_result["follow_ups"] = []


        logger.info(f"Retrieval analysis/follow-up generation finished in {time.time() - start_time:.2f}s. Result: {parsed_result}")
        return parsed_result


# --- Main Processing Function ---

def process_advanced_query(query: str, chat_history: list, chatbot_id: int, session_id: str, rag_service_instance: 'RAGService'):
    """
    Processes a user query using an advanced RAG pipeline involving multiple steps:
    1. Query Understanding (Intent/Slots, Decomposition)
    2. Multi-Step Retrieval (Query Variations, Vector Search, BM25, Re-ranking)
    3. Context Analysis & Follow-up Generation
    4. Context Compression
    5. Final Answer Synthesis

    Args:
        query: The user's query string.
        chat_history: List of previous conversation turns.
        chatbot_id: The ID of the chatbot being interacted with.
        session_id: The current chat session ID.
        rag_service_instance: An instance of the base RAGService for vector search.

    Returns:
        A dictionary containing the final answer, source chunks, and potentially follow-up questions.
        Example: {
            "answer": "The final synthesized answer.",
            "sources": [{"text": "...", "metadata": {...}}, ...],
            "follow_ups": ["Follow-up question 1?", ...]
        }
        Returns an error structure on failure.
    """
    start_pipeline_time = time.time()
    logger.info(f"Starting advanced RAG pipeline for chatbot {chatbot_id}, session {session_id}")
    current_app.logger.info(f"Received query for advanced RAG: '{query[:100]}...'") # Use current_app logger for Flask context

    # --- Initialization ---
    try:
        # Instantiate the processor which initializes LLMs
        processor = AdvancedRagProcessor()
    except Exception as e:
        logger.error(f"Failed to instantiate AdvancedRagProcessor: {e}", exc_info=True)
        # Return 6-tuple for error
        return ("Sorry, I encountered an internal error and cannot process your request.", [], None, "Failed to initialize RAG processor.", 500, {})

    # Check if LLMs were initialized successfully within the processor
    if not processor.rephrasing_llm or not processor.final_llm:
         logger.error("LLM models were not initialized correctly within AdvancedRagProcessor.")
         # Return 6-tuple for error
         return ("Sorry, I encountered an internal error and cannot process your request.", [], None, "Failed to initialize necessary LLM models.", 500, {})


    chatbot = Chatbot.query.get(chatbot_id)
    if not chatbot:
        logger.error(f"Chatbot with ID {chatbot_id} not found.")
         # Return 6-tuple for error (404 status code)
        return ("Error: Chatbot configuration not found.", [], None, "Chatbot not found.", 404, {})

    # --- 1. Query Understanding ---
    logger.info("--- Step 1: Query Understanding ---")
    intent_slots = processor._recognize_intent_and_slots(query, chat_history)
    logger.info(f"Recognized Intent: {intent_slots.get('intent', 'N/A')}, Slots: {intent_slots.get('slots', {})}")

    # Decompose query into sub-questions
    sub_questions = processor._decompose_query(query, chat_history)
    logger.info(f"Decomposed into Sub-questions: {sub_questions}")

    # --- 2. Multi-Step Retrieval ---
    logger.info("--- Step 2: Multi-Step Retrieval ---")
    all_retrieved_chunks_map = {} # Use dict to store unique chunks by ID/hash
    processed_count = 0 # Track number of variations processed
    fetched_texts_dict = {} # Store fetched texts for BM25 and final details

    # --- 2a. Fetch All Texts and Prepare BM25 (with Caching) ---
    bm25 = None
    corpus_chunk_ids = []
    cache_key = f"bm25_index_{chatbot_id}"

    # Check cache first
    if cache_key in bm25_cache:
        try:
            bm25, corpus_chunk_ids = bm25_cache[cache_key]
            logger.info(f"BM25 index found in cache for chatbot {chatbot_id}. Using cached index with {len(corpus_chunk_ids)} documents.")
            # Need to ensure fetched_texts_dict is populated for later use if cache hit
            # Re-fetch texts based on cached IDs if needed for consistency elsewhere
            if corpus_chunk_ids:
                 cached_texts, fetch_err = rag_service_instance.fetch_chunk_texts(corpus_chunk_ids, chatbot_id)
                 if fetch_err:
                     logger.warning(f"Error fetching texts for cached BM25 IDs: {fetch_err}")
                     # Potentially invalidate cache or proceed cautiously
                 elif cached_texts and len(cached_texts) == len(corpus_chunk_ids):
                     fetched_texts_dict = dict(zip(corpus_chunk_ids, cached_texts))
                 else:
                     logger.warning(f"Mismatch fetching texts for cached BM25 IDs. Fetched {len(cached_texts) if cached_texts else 0}, expected {len(corpus_chunk_ids)}. Proceeding without full text dict.")
            else:
                 logger.info("Cached BM25 index has no associated chunk IDs.")

        except Exception as e:
            logger.error(f"Error retrieving BM25 from cache or re-fetching texts: {e}. Will attempt rebuild.", exc_info=True)
            bm25 = None # Force rebuild on error
            corpus_chunk_ids = []
            if cache_key in bm25_cache: # Remove potentially corrupted entry
                 del bm25_cache[cache_key]

    if bm25 is None: # If not in cache or cache retrieval failed
        logger.info(f"BM25 index not in cache for chatbot {chatbot_id}. Attempting to build...")
        try:
            mappings_result = db.session.query(VectorIdMapping.vector_id).filter_by(chatbot_id=chatbot_id).all()
            if mappings_result:
                vector_ids_to_fetch = [m.vector_id for m in mappings_result if m.vector_id]
                logger.info(f"Found {len(vector_ids_to_fetch)} vector IDs for chatbot {chatbot_id} for potential BM25.")
                if vector_ids_to_fetch:
                    fetched_texts, fetch_err = rag_service_instance.fetch_chunk_texts(vector_ids_to_fetch, chatbot_id)
                    if fetch_err:
                        logger.warning(f"Error fetching some texts for BM25 build: {fetch_err}")
                    # Initialize containers for safe building
                    fetched_texts_dict = {}
                    corpus_texts = []
                    current_corpus_chunk_ids = []

                    # Check if fetched_texts is a list and lengths match *before* iterating
                    if isinstance(fetched_texts, list) and len(fetched_texts) == len(vector_ids_to_fetch):
                        logger.info(f"Successfully fetched {len(fetched_texts)} texts matching the {len(vector_ids_to_fetch)} requested IDs.")
                        # Build the dictionary and lists safely, ensuring alignment
                        for vec_id, text in zip(vector_ids_to_fetch, fetched_texts): # Zip is safe here due to prior length check
                            if text: # Ensure text is not None or empty
                                fetched_texts_dict[vec_id] = text
                                corpus_texts.append(text)
                                current_corpus_chunk_ids.append(vec_id)
                            else:
                                logger.debug(f"Skipping vector ID {vec_id} for BM25 due to empty text.")

                        if corpus_texts:
                            logger.info(f"Building BM25 index with {len(corpus_texts)} documents.")
                            tokenized_corpus = [doc.split(" ") for doc in corpus_texts]
                            bm25 = BM25Okapi(tokenized_corpus)
                            corpus_chunk_ids = current_corpus_chunk_ids # Use the safely built list
                            logger.info(f"BM25 index built successfully. Caching...")
                            # Store the built index and associated IDs in cache
                            bm25_cache[cache_key] = (bm25, corpus_chunk_ids)
                        else:
                            logger.warning("No valid text found after filtering fetched texts for BM25 index build.")

                    elif fetched_texts is None:
                         logger.warning(f"Fetching texts for BM25 build failed (returned None). Skipping BM25.")
                    else:
                        # This case handles fetch_err or length mismatch detected after fetch
                        logger.warning(f"Mismatch or error fetching texts for BM25 build. Fetched {len(fetched_texts) if isinstance(fetched_texts, list) else 'non-list'}, expected {len(vector_ids_to_fetch)}. Skipping BM25.")
            else:
                 logger.warning(f"No VectorIdMappings found for chatbot {chatbot_id}. Skipping BM25 build.")
        except Exception as e:
             logger.error(f"Error preparing BM25 index: {e}", exc_info=True)


    # --- 2b. Multi-Step Retrieval Loop ---
    all_retrieved_chunk_ids = set()
    all_retrieved_chunks = [] # Holds detailed chunk dicts {'id': str, 'text': str, 'metadata': dict}
    analysis_result = {"sufficient": False, "follow_ups": []} # Initialize

    # Get max retrieval steps from config
    max_retrieval_steps = current_app.config.get('MAX_RETRIEVAL_STEPS', 3)
    for current_step in range(max_retrieval_steps):
        logger.info(f"--- Starting Retrieval Step {current_step + 1}/{max_retrieval_steps} ---")

        # Determine queries for this step
        if current_step == 0:
            queries_for_step = sub_questions
            logger.info(f"Step {current_step + 1}: Using initial sub-questions: {queries_for_step}")
        else:
            queries_for_step = analysis_result.get("follow_ups", [])
            if not queries_for_step:
                logger.info(f"Step {current_step + 1}: No follow-up questions generated. Breaking loop.")
                break
            logger.info(f"Step {current_step + 1}: Using follow-up questions: {queries_for_step}")

        # Generate variations
        current_step_variations = []
        for q in queries_for_step:
            current_step_variations.extend(processor._generate_query_variations(q, chat_history))
        current_step_variations = list(dict.fromkeys(current_step_variations)) # Deduplicate
        logger.info(f"Step {current_step + 1}: Generated {len(current_step_variations)} unique variations.")

        # Retrieve for variations
        current_step_chunk_ids = set()
        for i, current_variation in enumerate(current_step_variations):
            # Get processing limit from config
            variation_processing_limit = current_app.config.get('VARIATION_PROCESSING_LIMIT', 10)
            if processed_count >= variation_processing_limit: # Overall limit check
                 logger.warning(f"Reached processing limit ({processed_count}). Skipping remaining variations.")
                 break

            logger.debug(f"Step {current_step + 1}, Variation {i+1}/{len(current_step_variations)}: Processing '{current_variation[:100]}...'")
            # Initialize results for this variation
            vector_results = []
            bm25_scores = {}
            vector_ranks = {}
            bm25_ranks = {}

            try:
                # 1. Generate embedding for the current variation
                query_embeddings, emb_error = rag_service_instance.generate_multiple_embeddings([current_variation])
                if emb_error or not query_embeddings:
                    logger.error(f"Failed to generate embedding for variation '{current_variation[:50]}...': {emb_error}")
                    # Decide how to handle: skip variation, use default, etc. Here, we skip.
                    processed_count += 1 # Count as processed even if skipped due to embedding error
                    continue # Skip to the next variation

                # 2. Vector Search using the correct method and embedding
                # Note: retrieve_chunks_multi_query uses RAG_TOP_K from config internally.
                # It requires client_id, which is not directly available here. Using session_id instead.
                # It returns (list_of_tuples, error_message)
                neighbors_list, retrieval_error = rag_service_instance.retrieve_chunks_multi_query(
                    query_embeddings=query_embeddings,
                    chatbot_id=chatbot_id,
                    client_id=session_id # Use session_id as client_id for the RAG service call
                )

                if retrieval_error:
                     logger.warning(f"Vector search failed for variation '{current_variation[:50]}...': {retrieval_error}")
                     vector_results_raw = [] # Ensure it's an empty list on error
                else:
                    # Convert list of (id, distance) tuples to list of dicts expected by subsequent code
                    # Convert list of chunk IDs to list of dicts (distance is not returned by retrieve_chunks_multi_query)
                    vector_results_raw = [{'id': neighbor_id} for neighbor_id in neighbors_list]

                # 3. Process results (sorting and ranking remain the same)
                vector_results = sorted(vector_results_raw, key=lambda x: x.get('distance', float('inf'))) # Sort by distance ascending
                vector_ranks = {res['id']: r + 1 for r, res in enumerate(vector_results) if 'id' in res}

                # BM25 Search (starts from here)
                if bm25 is not None and corpus_chunk_ids:
                    tokenized_variation = current_variation.split(" ")
                    if tokenized_variation:
                        doc_scores_all = bm25.get_scores(tokenized_variation)
                        # Map scores to chunk IDs and filter positive scores
                        scored_chunk_ids = [(corpus_chunk_ids[idx], doc_scores_all[idx])
                                            for idx in range(len(doc_scores_all))
                                            if doc_scores_all[idx] > 0]
                        # Sort by score descending to get ranks
                        bm25_results_ranked = sorted(scored_chunk_ids, key=lambda item: item[1], reverse=True)
                        bm25_ranks = {item[0]: r + 1 for r, item in enumerate(bm25_results_ranked)}
                        # Store scores for potential future use if needed, though ranks are primary for RRF
                        bm25_scores = dict(bm25_results_ranked)


                # --- Reciprocal Rank Fusion (RRF) ---
                rrf_scores = defaultdict(float)
                all_ids_variation = set(vector_ranks.keys()) | set(bm25_ranks.keys())

                for chunk_id in all_ids_variation:
                    score = 0.0
                    if chunk_id in vector_ranks:
                        # Get RRF K from config
                        rrf_k_val = current_app.config.get('RRF_K', 60)
                        score += 1.0 / (rrf_k_val + vector_ranks[chunk_id])
                    if chunk_id in bm25_ranks:
                        rrf_k_val = current_app.config.get('RRF_K', 60) # Get again in case loop modifies it (unlikely but safe)
                        score += 1.0 / (rrf_k_val + bm25_ranks[chunk_id])
                    rrf_scores[chunk_id] = score

                # Sort by RRF score and select top N for this variation
                rrf_sorted_chunk_ids_variation = sorted(all_ids_variation, key=lambda cid: rrf_scores[cid], reverse=True)
                # Get num hybrid chunks from config
                num_hybrid_chunks = current_app.config.get('NUM_HYBRID_CHUNKS_PER_VARIATION', 5)
                top_rrf_chunk_ids_variation = rrf_sorted_chunk_ids_variation[:num_hybrid_chunks]

                # Use RRF selected chunks for accumulation
                variation_chunk_ids = set(top_rrf_chunk_ids_variation)
                current_step_chunk_ids.update(variation_chunk_ids)
                processed_count += 1

            except Exception as e:
                logger.error(f"Error during retrieval or RRF for variation '{current_variation[:50]}...': {e}", exc_info=True)

        # Check limit again after processing variations for the step
        if processed_count >= variation_processing_limit: break # Exit outer loop if limit reached

        # Identify and fetch details for new chunks
        new_chunk_ids_to_fetch = current_step_chunk_ids - all_retrieved_chunk_ids
        logger.info(f"Step {current_step + 1}: Identified {len(new_chunk_ids_to_fetch)} new chunk IDs.")

        if new_chunk_ids_to_fetch:
            newly_fetched_chunks = []
            mappings = db.session.query(VectorIdMapping).filter(
                VectorIdMapping.vector_id.in_(list(new_chunk_ids_to_fetch)),
                VectorIdMapping.chatbot_id == chatbot_id
            ).all()
            mapping_dict = {m.vector_id: m for m in mappings}

            for chunk_id in new_chunk_ids_to_fetch:
                mapping = mapping_dict.get(chunk_id)
                chunk_text = fetched_texts_dict.get(chunk_id) # Get pre-fetched text
                if mapping and chunk_text:
                    newly_fetched_chunks.append({
                        "id": chunk_id,
                        "text": chunk_text,
                        "metadata": {"source": mapping.source_identifier or 'Unknown source'}
                    })
                else:
                     logger.warning(f"Could not find mapping or text for new chunk_id: {chunk_id}")

            if newly_fetched_chunks:
                all_retrieved_chunks.extend(newly_fetched_chunks)
                all_retrieved_chunk_ids.update(new_chunk_ids_to_fetch)
                logger.info(f"Step {current_step + 1}: Accumulated {len(newly_fetched_chunks)} new chunks. Total unique: {len(all_retrieved_chunks)}")

        # Analyze context sufficiency
        if not all_retrieved_chunks:
            logger.warning(f"Step {current_step + 1}: No chunks accumulated, skipping analysis.")
            if current_step == 0: break # Stop if no chunks after first step
        else:
            analysis_result = processor._analyze_retrieval_and_generate_followups(query, sub_questions, all_retrieved_chunks)
            logger.info(f"Step {current_step + 1}: Analysis Result: {analysis_result}")
            if analysis_result.get("sufficient", False):
                logger.info(f"Step {current_step + 1}: Sufficient context found. Breaking loop.")
                break

    logger.info(f"Multi-step retrieval finished. Total unique chunks: {len(all_retrieved_chunks)}")

    # --- 3. Post-Loop Processing ---
    # Re-rank all accumulated chunks based on the original query
    if all_retrieved_chunks:
        reranked_chunks = processor._rerank_chunks(query, all_retrieved_chunks)
    else:
        reranked_chunks = []
    logger.info(f"Chunks after final re-ranking: {len(reranked_chunks)}")

    # Select top N, format, and compress
    # Get num final chunks from config
    num_final_chunks = current_app.config.get('NUM_FINAL_CONTEXT_CHUNKS', 5)
    final_context_chunks = reranked_chunks[:num_final_chunks]
    logger.info(f"Selected top {len(final_context_chunks)} chunks for final context.")
    formatted_context = processor._format_context(final_context_chunks)
    # Pass config value for target tokens
    config_target_token_limit = current_app.config.get('CONTEXT_COMPRESSION_TARGET_TOKENS', 4000)
    compressed_context = processor._compress_context(formatted_context, config_target_token_limit, query)

    # --- 4. Final Answer Synthesis ---
    logger.info("--- Step 5: Final Answer Synthesis ---")
    final_answer = "Sorry, I couldn't generate a response based on the available information." # Default answer

    # Construct final prompt (using processor's method might be cleaner in future)
    formatted_history = "\n".join([f"{turn['role']}: {turn['content']}" for turn in chat_history])
    base_prompt = chatbot.base_prompt or "You are a helpful assistant."
    final_prompt = f"""{base_prompt}

Chat History:
---
{formatted_history}
---

Context (potentially summarized for relevance and length):
---
{compressed_context}
---

User Query: "{query}"

Instructions: Based *only* on the provided context and chat history, answer the user's query. If the context does not contain the answer, state that clearly. Cite the source number (e.g., [Source 1], [Source 2]) where the information was found, if possible. Do not make up information.

Answer:"""

    try:
        # Load final response settings from config
        final_temp = current_app.config.get('FINAL_RESPONSE_TEMPERATURE', 0.5)
        final_max_tokens = current_app.config.get('FINAL_RESPONSE_MAX_TOKENS', 1500)
        final_safety_settings = current_app.config.get('FINAL_RESPONSE_SAFETY_SETTINGS', DEFAULT_QUERY_REPHRASING_SAFETY_SETTINGS) # Reuse default if specific not set

        generation_config = GenerationConfig(
            temperature=final_temp,
            max_output_tokens=final_max_tokens,
        )

        # Use the pre-initialized final response model from the processor
        llm_response = processor.final_llm.generate_content(
            contents=[final_prompt],
            generation_config=generation_config,
            safety_settings=final_safety_settings,
            stream=False,
        )

        # Extract the answer
        if llm_response and llm_response.candidates and llm_response.candidates[0].content.parts:
            response_text = llm_response.candidates[0].content.parts[0].text.strip()
            if response_text:
                final_answer = response_text
                logger.info("Successfully generated final answer.")
            else:
                 logger.warning("Final LLM response text was empty.")
                 # Keep default answer
        else:
            logger.warning(f"Final LLM response was empty or invalid. Response: {llm_response}")
            try: # Check for blocked content
                if llm_response.prompt_feedback.block_reason:
                    final_answer = f"Sorry, I couldn't generate a response due to safety filters ({llm_response.prompt_feedback.block_reason.name})."
                    logger.warning(f"Final response blocked. Reason: {llm_response.prompt_feedback.block_reason.name}")
            except Exception:
                 pass # Ignore if block_reason is not accessible

    except Exception as e:
        logger.error(f"Error during final answer synthesis: {e}", exc_info=True)
        final_answer = "Sorry, I encountered an error while generating the final response."


    # --- 5. Result Packaging ---
    end_pipeline_time = time.time()
    duration = end_pipeline_time - start_pipeline_time
    logger.info(f"Advanced RAG pipeline finished in {duration:.2f}s.")

    # Map source numbers back to original metadata if needed for frontend display
    result = {
        "answer": final_answer,
        "sources": final_context_chunks, # Chunks used for the final prompt
        "follow_ups": analysis_result.get("follow_ups", []) # Use 'follow_ups' key
    }

    # Return 6-tuple matching expected structure in rag_service.py
    # (response_text, sources, history, error_message, status_code, metadata)
    return (
        result.get("answer", "Error: No answer generated."), # response_text
        result.get("sources", []),                         # sources
        None,                                              # history (not applicable here)
        None,                                              # error_message (success path)
        200,                                               # status_code (success)
        {"follow_ups": result.get("follow_ups", [])}       # metadata
    )