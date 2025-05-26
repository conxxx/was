# app/services/summarization_service.py
import logging
import requests
from bs4 import BeautifulSoup
import validators
from urllib.parse import urlparse
import os
import time # For timing
from flask import current_app
from google.api_core import exceptions as google_exceptions
import vertexai
from vertexai.generative_models import GenerativeModel, Part, HarmCategory, HarmBlockThreshold, FinishReason # Added Vertex AI imports
from google.cloud import aiplatform # Though not directly used in init, good practice if extending later

from app.models import Chatbot, db # Assuming db is initialized in app

# --- Configuration (Mirrors rag_service for consistency) ---
PROJECT_ID = os.environ.get('PROJECT_ID', "roo-code-459017") # Use the same default as RAG
REGION = os.environ.get('REGION', "us-central1") # Use the same default as RAG
GEMINI_MODEL_NAME = 'gemini-2.5-flash-preview-04-17' # Model for summarization/translation tasks

# --- Safety Settings (Mirrors rag_service) ---
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}
class SummarizationService:
    def __init__(self, logger):
        # Store the passed logger instance
        self.logger = logger
        self.gemini_model = None

        # Initialize Vertex AI SDK and Generative Model
        self.initialization_error = None
        try:
            self.logger.info(f"Summarization Service: Initializing Vertex AI SDK (Project: {PROJECT_ID}, Region: {REGION})...")
            start_time = time.time()
            # Check if already initialized (optional, but can prevent warnings if called multiple times)
            try:
                aiplatform.Client() # Check if a client context exists
                self.logger.info("Vertex AI SDK already initialized.")
            except Exception:
                vertexai.init(project=PROJECT_ID, location=REGION)
                self.logger.info("Vertex AI SDK initialized.")

            # Initialize the specific model we intend to use
            self.logger.info(f"Summarization Service: Initializing Generative Model '{GEMINI_MODEL_NAME}'...")
            self.gemini_model = GenerativeModel(GEMINI_MODEL_NAME)
            self.logger.info(f"Summarization Service: Generative Model '{GEMINI_MODEL_NAME}' OK. Initialization complete ({time.time() - start_time:.2f}s).")

        except google_exceptions.GoogleAPIError as e:
            self.logger.error(f"Summarization Service: FATAL - Vertex AI SDK/Model Initialization Failed (API Error): {e}", exc_info=True)
            self.initialization_error = f"Vertex AI API Error: {e}"
            self.gemini_model = None
        except Exception as e:
            self.logger.error(f"Summarization Service: FATAL - Vertex AI SDK/Model Initialization Failed (Other Error): {e}", exc_info=True)
            self.initialization_error = f"Initialization Error: {e}"
            self.gemini_model = None

    def _normalize_domain(self, domain_str: str) -> str:
        """Normalizes a domain string to its base form (lowercase, no www., no port)."""
        if not domain_str:
            return ""
        # Lowercase and remove leading/trailing whitespace
        normalized = domain_str.strip().lower()
        # Remove www. prefix if it exists
        if normalized.startswith('www.'):
            normalized = normalized[4:]
        # Remove potential port number
        normalized = normalized.split(':')[0]
        return normalized

    def _validate_url_domain(self, url: str, allowed_domains: list) -> bool:
        """
        Validates the URL format (scheme, netloc) and checks if its domain/prefix
        matches any of the allowed domains, explicitly handling localhost.
        """
        try:
            parsed_url = urlparse(url)

            # 1. Basic Format Validation using urlparse
            if not parsed_url.scheme or parsed_url.scheme not in ['http', 'https']:
                self.logger.warning(f"URL scheme '{parsed_url.scheme}' is not allowed or missing: {url}")
                raise ValueError("URL scheme must be HTTP or HTTPS.")

            if not parsed_url.netloc:
                self.logger.warning(f"Could not extract network location (domain/host) from URL: {url}")
                raise ValueError("Invalid URL format: Missing domain/host.")

            # 2. Domain Allowlist Check
            if not allowed_domains:
                self.logger.warning(f"No allowed domains configured for this chatbot. Denying URL: {url}")
                raise PermissionError("Summarization/Scraping is not configured for any domains for this chatbot.")

            # Iterate through allowed domains and check if the URL starts with any of them correctly
            for allowed_base_url in allowed_domains:
                allowed_base_url = allowed_base_url.strip()
                if not allowed_base_url:
                    continue

                # Basic validation of the allowed domain format itself
                try:
                    parsed_allowed = urlparse(allowed_base_url)
                    # Ensure allowed domains also have valid schemes and netlocs
                    if parsed_allowed.scheme not in ['http', 'https'] or not parsed_allowed.netloc:
                        self.logger.warning(f"Skipping invalid or non-HTTP/S allowed domain in config: '{allowed_base_url}'")
                        continue
                except Exception: # Handle potential errors from urlparse on malformed strings
                    self.logger.warning(f"Skipping malformed allowed domain in config: '{allowed_base_url}'")
                    continue

                # Normalize by removing trailing slash for comparison
                normalized_allowed = allowed_base_url.rstrip('/')

                # Check if the input URL starts with the normalized allowed URL
                if url.startswith(normalized_allowed):
                    # Ensure it's either an exact match or a valid sub-path
                    len_allowed = len(normalized_allowed)
                    if len(url) == len_allowed or (len(url) > len_allowed and url[len_allowed] == '/'):
                        self.logger.debug(f"URL '{url}' allowed based on configured domain '{allowed_base_url}'")
                        return True # Found a valid match

            # If loop completes without finding a match
            self.logger.warning(f"URL '{url}' denied. It does not match or start as a sub-path of any allowed domains: {allowed_domains}")
            # Return False instead of raising an error here, let the caller handle denial.
            return False # Explicitly return False if no match found

        except ValueError as ve: # Catch specific validation errors raised above
             self.logger.error(f"URL validation failed for {url}: {ve}")
             raise ve # Re-raise the specific validation error
        except PermissionError as pe: # Catch permission error if no domains configured
             self.logger.error(f"URL validation failed for {url}: {pe}")
             raise pe # Re-raise permission error
        except Exception as e:
            self.logger.error(f"Unexpected error during URL domain validation for {url}: {e}", exc_info=True)
            # Avoid leaking potentially sensitive details from arbitrary exceptions
            raise ValueError(f"An unexpected error occurred during URL domain validation.") # Generic error for unexpected issues


    def _scrape_url(self, url: str) -> str:
        """
        Performs manual scraping of a given URL.
        Assumes domain validation happened before calling this.
        Returns the extracted text content or raises an exception.
        """
        self.logger.info(f"Starting scrape for URL: {url}")
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; ChatbotSummarizer/1.0)'} # Identify your bot

        try:
            response = requests.get(url, headers=headers, timeout=20, allow_redirects=True) # Increased timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            content_type = response.headers.get('content-type', '').lower()
            if 'html' not in content_type:
                 self.logger.warning(f"Content type for {url} is not HTML ({content_type}). Skipping scrape.")
                 raise ValueError(f"Cannot scrape non-HTML content ({content_type}).")

            # Use html5lib parser for robustness
            soup = BeautifulSoup(response.content, 'html5lib')

            # Attempt to find main content areas (common tags/attributes)
            # This is heuristic and might need refinement based on target sites
            main_content_tags = soup.find_all(['main', 'article', 'div'], {'role': 'main'})
            if not main_content_tags:
                 main_content_tags = soup.find_all('body') # Fallback to body

            text_parts = []
            for tag in main_content_tags:
                 # Remove script, style, nav, header, footer elements before extracting text
                 for unwanted_tag in tag(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form']):
                     unwanted_tag.decompose()
                 text = tag.get_text(separator=' ', strip=True)
                 if text:
                     text_parts.append(text)

            full_text = ' '.join(text_parts)

            if not full_text:
                 # Fallback if specific tags yielded nothing
                 self.logger.warning(f"Could not find specific main content tags for {url}, falling back to full body text.")
                 body = soup.find('body')
                 if body:
                     for unwanted_tag in body(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form']):
                         unwanted_tag.decompose()
                     full_text = body.get_text(separator=' ', strip=True)

            if not full_text:
                 self.logger.warning(f"No text content extracted after scraping {url}.")
                 # Return empty string, let the caller decide if it's an error
                 return ""

            self.logger.info(f"Successfully scraped URL: {url}. Extracted ~{len(full_text)} characters.")
            return full_text

        except requests.exceptions.Timeout:
            self.logger.error(f"Timeout occurred while scraping URL: {url}")
            raise TimeoutError(f"Scraping timed out for URL: {url}")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network/HTTP error scraping URL {url}: {e}", exc_info=True)
            raise ConnectionError(f"Failed to scrape URL {url}: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error during scraping of {url}: {e}", exc_info=True)
            raise RuntimeError(f"An unexpected error occurred during scraping: {e}")

    def _call_vertex_gemini_api(self, prompt: str, purpose: str = "generic") -> str: # Added purpose for logging
        """ Helper function to call the Vertex AI Gemini API and handle common errors. """
        if not self.gemini_model:
             self.logger.error(f"Vertex AI Gemini model is not initialized. Cannot make API call for {purpose}.")
             # Use the stored initialization error if available
             error_msg = self.initialization_error or "Summarization service (Vertex AI Gemini) is not available."
             raise RuntimeError(error_msg)

        api_call_start_time = time.time()
        try:
            self.logger.debug(f"Sending prompt for {purpose} to Vertex AI Gemini ({GEMINI_MODEL_NAME}): {prompt[:100]}...") # Log truncated prompt
            # Use the SAFETY_SETTINGS defined at the module level
            response = self.gemini_model.generate_content(
                contents=prompt, # Vertex AI uses 'contents' parameter
                generation_config={"temperature": 0.3, "max_output_tokens": 8192}, 
                safety_settings=SAFETY_SETTINGS,
                # stream=False # Default is False
            )
            self.logger.info(f"PERF: Vertex AI Gemini API call for {purpose} took {time.time() - api_call_start_time:.4f} seconds.")
            self.logger.debug(f"Raw Vertex AI Gemini response received for {purpose}.")

            # --- Vertex AI Response Parsing ---
            # Check finish reason first
            if response.candidates and response.candidates[0].finish_reason != FinishReason.STOP:
                finish_reason_str = response.candidates[0].finish_reason.name
                # Check for safety blocks specifically
                if response.candidates[0].finish_reason == FinishReason.SAFETY:
                     safety_ratings_str = ", ".join([f"{rating.category.name}: {rating.probability.name}" for rating in response.candidates[0].safety_ratings])
                     self.logger.error(f"Vertex AI Gemini API call for {purpose} blocked due to safety settings. Finish Reason: {finish_reason_str}. Ratings: [{safety_ratings_str}]")
                     raise RuntimeError(f"Content blocked by safety filters ({finish_reason_str}).")
                else:
                     # Other non-STOP reasons (MAX_TOKENS, RECITATION, etc.)
                     self.logger.error(f"Vertex AI Gemini API call for {purpose} finished with non-STOP reason: {finish_reason_str}")
                     raise RuntimeError(f"Gemini API call failed (Reason: {finish_reason_str}).")

            # Check for content parts
            if response.candidates and response.candidates[0].content.parts:
                result_text = response.candidates[0].content.parts[0].text
                self.logger.debug(f"Vertex AI Gemini API call for {purpose} successful, text extracted.")
                return result_text.strip()
            else:
                # This case should be less common if finish_reason was STOP, but handle defensively
                self.logger.error(f"Vertex AI Gemini API response format unexpected or missing text parts for {purpose}, despite FinishReason=STOP. Response: {response}")
                raise RuntimeError("Gemini API call failed (unexpected response format or missing text).")

        except google_exceptions.GoogleAPIError as e:
            self.logger.error(f"Google API error calling Vertex AI Gemini for {purpose}: {e}", exc_info=True)
            self.logger.info(f"PERF: Vertex AI Gemini API call for {purpose} took {time.time() - api_call_start_time:.4f} seconds (API Error).")
            # Check for specific API errors like Quota, Permission Denied, etc. if needed
            if isinstance(e, google_exceptions.ResourceExhausted):
                raise ConnectionError(f"Vertex AI Gemini API quota exceeded: {e}")
            elif isinstance(e, google_exceptions.PermissionDenied):
                 raise ConnectionError(f"Vertex AI Gemini API permission denied: {e}")
            else:
                 raise ConnectionError(f"Vertex AI Gemini API communication error: {e}")
        except Exception as e:
            # Catch potential errors during response parsing or other unexpected issues
            self.logger.error(f"Unexpected error calling Vertex AI Gemini API for {purpose} or processing response: {e}", exc_info=True)
            self.logger.info(f"PERF: Vertex AI Gemini API call for {purpose} took {time.time() - api_call_start_time:.4f} seconds (Exception).")
            raise RuntimeError(f"An unexpected error occurred during the Vertex AI Gemini API call: {e}")


    def summarize(self, chatbot_id: int, content_type: str, content: str, target_language: str, source_language: str = None) -> dict:
        """
        Summarizes content using Google Gemini API. Handles language detection and translation via Gemini.
        """
        overall_start_time = time.time()
        self.logger.info(f"Received summarization request for chatbot {chatbot_id} via Gemini. Type: {content_type}, Target Lang: {target_language}, Source Lang: {source_language or 'auto'}")

        if not self.gemini_model:
             error_msg = self.initialization_error or "Summarization service (Vertex AI Gemini) is not available."
             self.logger.error(f"Summarization service is not initialized: {error_msg}")
             self.logger.info(f"PERF: Summarize method (chatbot {chatbot_id}) took {time.time() - overall_start_time:.4f} seconds (Init Error).")
             raise RuntimeError("Summarization service dependencies are not available.")

        step_start_time = time.time()
        chatbot = Chatbot.query.get(chatbot_id)
        self.logger.info(f"PERF: Chatbot query for ID {chatbot_id} took {time.time() - step_start_time:.4f} seconds.")
        if not chatbot:
             self.logger.error(f"Summarization request failed: Chatbot with ID {chatbot_id} not found.")
             self.logger.info(f"PERF: Summarize method (chatbot {chatbot_id}) took {time.time() - overall_start_time:.4f} seconds (Chatbot Not Found).")
             raise ValueError(f"Chatbot with ID {chatbot_id} not found.")

        if not chatbot.summarization_enabled:
             self.logger.warning(f"Summarization attempt denied for disabled chatbot {chatbot_id}.")
             self.logger.info(f"PERF: Summarize method (chatbot {chatbot_id}) took {time.time() - overall_start_time:.4f} seconds (Summarization Disabled).")
             raise PermissionError("Summarization feature is not enabled for this chatbot.")

        text_to_process = ""
        detected_source_language = source_language

        # 1. Get Content (Scrape or use Paste)
        if content_type == 'url':
            step_start_time = time.time()
            allowed_domains_str = chatbot.allowed_scraping_domains or ""
            allowed_domains = [d.strip() for d in allowed_domains_str.replace(',', '\n').splitlines() if d.strip()]
            self.logger.debug(f"Allowed domains for chatbot {chatbot_id}: {allowed_domains}")

            try:
                validate_start_time = time.time()
                is_valid_domain = self._validate_url_domain(content, allowed_domains)
                self.logger.info(f"PERF: URL Domain Validation for '{content}' took {time.time() - validate_start_time:.4f} seconds.")
                if not is_valid_domain:
                    self.logger.warning(f"Domain validation failed for URL: {content}")
                    self.logger.info(f"PERF: Summarize method (chatbot {chatbot_id}) took {time.time() - overall_start_time:.4f} seconds (Invalid Domain).")
                    raise PermissionError(f"Domain for URL '{content}' is not allowed for summarization.")
                
                scrape_start_time = time.time()
                text_to_process = self._scrape_url(content)
                self.logger.info(f"PERF: URL Scraping for '{content}' took {time.time() - scrape_start_time:.4f} seconds.")
            except (ValueError, PermissionError, ConnectionError, TimeoutError, RuntimeError) as e:
                 self.logger.error(f"Failed to get content from URL {content}: {e}")
                 self.logger.info(f"PERF: Summarize method (chatbot {chatbot_id}) URL processing took {time.time() - step_start_time:.4f} seconds (Error).")
                 raise e # Re-raise specific errors
            except Exception as e:
                 self.logger.error(f"Unexpected error during URL processing for {content}: {e}", exc_info=True)
                 self.logger.info(f"PERF: Summarize method (chatbot {chatbot_id}) URL processing took {time.time() - step_start_time:.4f} seconds (Unexpected Error).")
                 raise RuntimeError(f"An unexpected error occurred while processing the URL: {e}")
            self.logger.info(f"PERF: URL Content Processing (Validate + Scrape) took {time.time() - step_start_time:.4f} seconds.")

        elif content_type == 'paste' or content_type == 'text': # Accept 'text' like 'paste'
            text_to_process = content
            self.logger.info(f"Processing pasted/text content (~{len(text_to_process)} chars). Type: {content_type}")
        else:
            self.logger.error(f"Invalid content_type provided: {content_type}")
            self.logger.info(f"PERF: Summarize method (chatbot {chatbot_id}) took {time.time() - overall_start_time:.4f} seconds (Invalid Content Type).")
            raise ValueError("Invalid content_type. Must be 'url', 'paste', or 'text'.")

        if not text_to_process or not text_to_process.strip():
             self.logger.warning("No content available to summarize after processing input.")
             self.logger.info(f"PERF: Summarize method (chatbot {chatbot_id}) took {time.time() - overall_start_time:.4f} seconds (No Content).")
             return {"error": "No content found to summarize."}

        # Truncate very long text before sending to API to avoid excessive cost/limits
        # Adjust this limit as needed based on typical usage and Gemini constraints
        MAX_INPUT_CHARS = 15000 # Example limit
        if len(text_to_process) > MAX_INPUT_CHARS:
            self.logger.warning(f"Input text length ({len(text_to_process)}) exceeds limit ({MAX_INPUT_CHARS}). Truncating.")
            text_to_process = text_to_process[:MAX_INPUT_CHARS]

        # 2. Language Detection (if not provided, using Gemini)
        if not detected_source_language:
            step_start_time = time.time()
            self.logger.info("Attempting language detection using Gemini...")
            try:
                # Use a sample for potentially long text to speed up detection
                sample = text_to_process[:1000]
                prompt = f"Detect the primary language of the following text. Respond with only the two-letter ISO 639-1 language code (e.g., 'en', 'es', 'fr'). Text: \"\"\"{sample}\"\"\""
                detected_code = self._call_vertex_gemini_api(prompt, purpose="language_detection").lower()
                # Basic validation of the returned code format
                if len(detected_code) == 2 and detected_code.isalpha():
                    detected_source_language = detected_code
                    self.logger.info(f"Detected source language via Gemini: {detected_source_language}")
                else:
                    self.logger.warning(f"Gemini language detection returned unexpected format: '{detected_code}'. Defaulting to 'en'.")
                    detected_source_language = 'en'
            except (RuntimeError, ConnectionError) as e:
                 self.logger.warning(f"Gemini language detection failed: {e}. Defaulting to 'en'.")
                 detected_source_language = 'en'
            except Exception as e:
                 self.logger.error(f"Unexpected error during Gemini language detection: {e}", exc_info=True)
                 detected_source_language = 'en' # Safer fallback
            self.logger.info(f"PERF: Language Detection took {time.time() - step_start_time:.4f} seconds.")

        # 3. Summarization (using Gemini)
        summary_text = ""
        step_start_time = time.time()
        try:
            self.logger.info(f"Performing summarization using Gemini ({GEMINI_MODEL_NAME})...")
            # Construct a more explicit prompt for summarization
            instruction = "Summarize the text concisely."
            # If source language is known and different from target, include it in the prompt
            if detected_source_language and detected_source_language != target_language:
                instruction = f"The following text is in {detected_source_language}. {instruction}"
            
            # Moved the IMPORTANT instruction to the end of the prompt
            prompt = f"{instruction}\n\nText to summarize:\n\"\"\"{text_to_process}\"\"\"\n\nTask: Provide a concise summary of the text above.\nIMPORTANT: The summary MUST be written in {target_language}."
            
            self.logger.debug(f"Summarization prompt for Gemini: {prompt[:200]}...") # Log a part of the prompt

            summary_text = self._call_vertex_gemini_api(prompt, purpose="summarization")
            self.logger.info(f"Summarization successful via Gemini. Generated summary length: {len(summary_text)} in target language {target_language}")

        except (RuntimeError, ConnectionError) as e:
            self.logger.error(f"Gemini summarization failed: {e}")
            self.logger.info(f"PERF: Summarization (LLM call) took {time.time() - step_start_time:.4f} seconds (Error).")
            self.logger.info(f"PERF: Summarize method (chatbot {chatbot_id}) overall took {time.time() - overall_start_time:.4f} seconds (Summarization Error).")
            raise RuntimeError(f"Failed to generate summary using Gemini: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error during Gemini summarization: {e}", exc_info=True)
            self.logger.info(f"PERF: Summarization (LLM call) took {time.time() - step_start_time:.4f} seconds (Unexpected Error).")
            self.logger.info(f"PERF: Summarize method (chatbot {chatbot_id}) overall took {time.time() - overall_start_time:.4f} seconds (Unexpected Summarization Error).")
            raise RuntimeError(f"An unexpected error occurred during summarization: {e}")
        self.logger.info(f"PERF: Summarization (LLM call) took {time.time() - step_start_time:.4f} seconds.")

        # 4. Return Result (Translation step removed)
        self.logger.info(f"PERF: Summarize method (chatbot {chatbot_id}) overall took {time.time() - overall_start_time:.4f} seconds (Success).")
        return {
            "summary": summary_text, # Use the directly generated summary
            "original_language": detected_source_language,
            "target_language": target_language # Return the requested target language
        }

# Optional: Helper function if needed elsewhere, but likely used internally by the app factory
# def get_summarization_service(logger):
#     return SummarizationService(logger)
