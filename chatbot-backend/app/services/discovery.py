    # app/services/discovery.py

import requests
import threading # Keep for lock if used elsewhere, but not in task logic
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET
from flask import current_app # Keep for potential use outside task? Better to use celery's logger
import time
import traceback # Import traceback for error logging
# GCP imports not used directly in this file based on current logic
# from google.cloud import aiplatform
# from google.cloud import storage
# --- Import Celery app instance and logger ---
from celery_worker import celery_app # Import from celery_worker.py
from celery import current_app as celery_current_app # Import celery's current_app for logger
from celery.utils.log import get_task_logger # Import the recommended logger utility

# --- Import specific exceptions for retry logic ---
import requests # Ensure requests is imported
from requests.exceptions import Timeout, ConnectionError, HTTPError, RequestException

# --- Constants ---
MAX_DEPTH = 2
MAX_PAGES = 50
REQUEST_TIMEOUT = 15
USER_AGENT = "ChatbotDataDiscoveryBot/1.0 (requests)"

# Define specific retryable exceptions for discovery (Network focus) based on doc.md
DISCOVERY_RETRYABLE_EXCEPTIONS = (
    Timeout,
    ConnectionError,
    # HTTPError needs checking status code 5xx inside the task
)

# Custom exception for non-retryable HTTP errors (e.g., 4xx)
class NonRetryableHTTPError(RequestException):
    pass

# --- Sync Helper Function (Uses current_app.logger) ---
def fetch_url_sync(url, task_instance=None): # Removed attempt parameter
    """
    Fetches content from a URL synchronously using requests.
    Allows retryable exceptions (Timeout, ConnectionError, HTTPError) to propagate.

    Args:
        url: The URL to fetch
        task_instance: The Celery task instance (optional, for logging context)
    """
    logger = get_task_logger(__name__) # Use standard Celery task logger
    headers = {'User-Agent': USER_AGENT}
    task_id_log = f"Task {task_instance.request.id}: " if task_instance and hasattr(task_instance, 'request') else ""

    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers, allow_redirects=True)
        # Raise HTTPError for bad responses (4xx or 5xx) - will be caught by caller
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '').lower()
        if 'html' in content_type:
            return response.text, 'html'
        elif 'xml' in content_type:
            # Use response.content for XML to handle encoding correctly
            return response.content, 'xml'
        else:
            logger.debug(f"{task_id_log}Skipping non-html/xml content type '{content_type}' for URL: {url}")
            return None, None

    # Let Timeout, ConnectionError, HTTPError propagate up naturally
    # Remove the except RETRYABLE_EXCEPTIONS block that called self.retry()

    except RequestException as e: # Catch other request exceptions if needed
        # Log non-retryable request errors specifically if desired
        logger.error(f"{task_id_log}Non-retryable request error fetching URL {url}: {e}", exc_info=True)
        raise # Re-raise to fail the task

    except Exception as e: # Catch other unexpected errors
        logger.error(f"{task_id_log}Unexpected error fetching URL {url}: {e}", exc_info=True)
        raise # Re-raise to fail the task

# --- Sync Discovery Logic (Returns result or raises error) ---
def discover_links_from_url_sync(start_url, task_id, task_instance=None):
    """
    Synchronous discovery using requests. Returns list of URLs or raises Exception.
    Exceptions from fetch_url_sync will propagate up.

    Args:
        start_url: The URL to start crawling from
        task_id: Identifier for the task (used for logging)
        task_instance: The Celery task instance (optional, for logging context)
    """
    logger = get_task_logger(__name__) # Use standard Celery task logger
    task_id_log = f"Task {task_instance.request.id}: " if task_instance and hasattr(task_instance, 'request') else f"Task {task_id}: "
    discovered_urls = set()
    queue = [(start_url, 0)]
    visited = {start_url}
    page_count = 0

    # Remove the outer try/except Exception block, let errors propagate to the task runner
    parsed_start_url = urlparse(start_url)
    base_domain = parsed_start_url.netloc
    if not base_domain:
        raise ValueError("Invalid start URL provided.")

    discovered_urls.add(start_url)

    while queue and page_count < MAX_PAGES:
        current_url, depth = queue.pop(0)

        if depth > MAX_DEPTH:
            continue

        logger.debug(f"{task_id_log}Crawling (sync) {current_url} (Depth: {depth}, Count: {page_count})")
        try:
            # Call the refactored fetch_url_sync, passing task_instance for logging context
            content, content_type = fetch_url_sync(current_url, task_instance=task_instance)

            if content and content_type == 'html':
                page_count += 1
                try:
                    soup = BeautifulSoup(content, 'html.parser')
                    for link in soup.find_all('a', href=True):
                        abs_url = urljoin(current_url, link['href']).split('#')[0]
                        # Use helper function for validation
                        if is_valid_url(abs_url, base_domain) and abs_url not in visited:
                            visited.add(abs_url)
                            discovered_urls.add(abs_url)
                            if depth + 1 <= MAX_DEPTH:
                                queue.append((abs_url, depth + 1))
                except Exception as parse_e:
                    # Keep specific exception handling for parsing errors
                    logger.warning(f"{task_id_log}Error parsing HTML from {current_url}: {parse_e}")

        except requests.exceptions.HTTPError as http_err:
            # Log HTTP errors (like 404) but continue crawling other links
            logger.warning(f"{task_id_log}HTTP error fetching {current_url}: {http_err}. Skipping this URL.")
            continue # Continue to the next URL in the queue
        except requests.exceptions.RequestException as req_err:
             # Log other request errors (like connection errors) but continue
             logger.warning(f"{task_id_log}Request error fetching {current_url}: {req_err}. Skipping this URL.")
             continue # Continue to the next URL in the queue

    result_list = sorted(list(discovered_urls))
    logger.info(f"{task_id_log}Sync Discovery completed. Found {len(result_list)} links.")
    return result_list # Return the result

def discover_links_from_sitemap_sync(sitemap_url, task_id, task_instance=None):
    """
    Synchronous sitemap discovery using requests. Returns list of URLs or raises Exception.
    Exceptions from fetch_url_sync will propagate up.

    Args:
        sitemap_url: The URL of the sitemap to process
        task_id: Identifier for the task (used for logging)
        task_instance: The Celery task instance (optional, for logging context)
    """
    logger = celery_current_app.logger # Use Celery's logger
    task_id_log = f"Task {task_instance.request.id}: " if task_instance and hasattr(task_instance, 'request') else f"Task {task_id}: "
    discovered_urls = set()

    # Remove the outer try/except Exception block, let errors propagate
    parsed_start_url = urlparse(sitemap_url)
    base_domain = parsed_start_url.netloc
    if not base_domain:
         raise ValueError("Invalid sitemap URL provided.")

    logger.debug(f"{task_id_log}Fetching sitemap (sync) {sitemap_url}")
    # Call refactored fetch_url_sync, passing task_instance for logging context
    # Exceptions will propagate from here
    content, content_type = fetch_url_sync(sitemap_url, task_instance=task_instance)

    if content and content_type == 'xml':
        try:
            root = ET.fromstring(content)
            namespaces = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            for url_element in root.findall('.//sm:url', namespaces):
                loc_element = url_element.find('sm:loc', namespaces)
                if loc_element is not None and loc_element.text:
                    url = loc_element.text.strip()
                    # Use helper function for validation
                    if is_valid_url(url, base_domain):
                         discovered_urls.add(url)
        except ET.ParseError as xml_e:
             # Keep specific error handling for XML parsing
             logger.error(f"{task_id_log}Failed to parse XML sitemap from {sitemap_url}: {xml_e}")
             raise ValueError(f"Failed to parse XML sitemap from {sitemap_url}") from xml_e # Chain exception
    elif content is None:
         # fetch_url_sync returned None, likely due to a handled error during fetch
         logger.error(f"{task_id_log}Failed to fetch sitemap content from {sitemap_url}")
         raise ValueError(f"Failed to fetch sitemap content from {sitemap_url}")
    else:
        # Content fetched but wasn't XML
        logger.error(f"{task_id_log}Content fetched from {sitemap_url} is not XML (type: {content_type})")
        raise ValueError(f"Content fetched from {sitemap_url} is not XML")

    result_list = sorted(list(discovered_urls))
    logger.info(f"{task_id_log}Sync Sitemap discovery completed. Found {len(result_list)} links.")
    return result_list # Return the result


# --- Celery Task Runner ---
@celery_app.task(
    bind=True,
    autoretry_for=DISCOVERY_RETRYABLE_EXCEPTIONS, # Only auto-retry specific network errors
    max_retries=4,
    retry_backoff=True,
    retry_backoff_max=300, # Max 5 minutes delay
    retry_jitter=True
)
def run_discovery_task(self, task_id, source_url, source_type):
    """
    Runs the appropriate synchronous discovery function as a Celery task.
    Uses autoretry_for for common network errors and handles HTTPError 5xx manually.
    """
    # No need for manual app context creation if Celery worker is configured correctly
    logger = get_task_logger(__name__) # Use Celery's recommended task logger
    logger.info(f"Task {self.request.id}: Starting discovery for {source_type} '{source_url}', Attempt {self.request.retries + 1}/{self.max_retries + 1}")

    # Update Celery state
    self.update_state(state='PROGRESS', meta={'status': 'running', 'source_url': source_url, 'attempt': self.request.retries + 1})

    try:
        # Call refactored helper functions, passing self for logging context
        if source_type == 'url':
            result = discover_links_from_url_sync(source_url, task_id, task_instance=self)
        elif source_type == 'sitemap':
            result = discover_links_from_sitemap_sync(source_url, task_id, task_instance=self)
        else:
            raise ValueError(f"Unsupported source_type '{source_type}' for discovery.")

        logger.info(f"Task {self.request.id}: Discovery completed successfully for {source_type} '{source_url}'. Found {len(result)} links.")
        # Success state handled by Celery implicitly upon return
        return {'status': 'completed', 'result': result, 'task_id': task_id}

    except HTTPError as http_err:
        # Explicitly handle HTTPError to check status code
        status_code = http_err.response.status_code if http_err.response else 500 # Default to 500 if no response
        if 500 <= status_code <= 599:
            # Retry on 5xx errors
            logger.warning(f"Task {self.request.id}: Retryable HTTP error {status_code} for {source_url} (Attempt {self.request.retries + 1}): {http_err}")
            # Update Celery state before retry
            self.update_state(state='RETRY', meta={'status': 'retrying_http_5xx', 'error': str(http_err), 'attempt': self.request.retries + 1})
            # Manually trigger retry for this specific case
            # Calculate countdown based on Celery's default backoff logic if possible, or use a simple exponential
            # Note: Celery's internal backoff calculation isn't directly exposed easily.
            # Using the task's default_retry_delay might not exist if not set, fallback needed.
            base_delay = getattr(self, 'default_retry_delay', 30) # Fallback to 30s
            countdown = base_delay * (2 ** self.request.retries)
            # Ensure countdown doesn't exceed retry_backoff_max if defined on task
            max_backoff = getattr(self, 'retry_backoff_max', 300)
            countdown = min(countdown, max_backoff)
            raise self.retry(exc=http_err, countdown=countdown) # Use self.retry for manual cases
        else:
            # Fail immediately on non-5xx errors (e.g., 401, 403, 404)
            logger.error(f"Task {self.request.id}: Non-retryable HTTP error {status_code} for {source_url}: {http_err}")
            self.update_state(state='FAILURE', meta={'status': 'failed_http_non_5xx', 'error': str(http_err)})
            # Raise the custom exception to prevent autoretry
            raise NonRetryableHTTPError(f"HTTP Error {status_code}") from http_err

    except DISCOVERY_RETRYABLE_EXCEPTIONS as e:
         # Caught by autoretry_for (Timeout, ConnectionError)
         logger.warning(f"Task {self.request.id}: Retryable network error for {source_url} (Attempt {self.request.retries + 1}): {e}", exc_info=True)
         # Update Celery state before automatic retry
         self.update_state(state='RETRY', meta={'status': 'retrying_network', 'error': str(e), 'attempt': self.request.retries + 1})
         raise e # Re-raise for autoretry

    except Exception as e:
        # Final failure (non-retryable like ValueError, NonRetryableHTTPError, or max retries exceeded from autoretry/manual retry)
        logger.error(f"Task {self.request.id}: FINAL FAILURE for discovery {source_type} '{source_url}' after {self.request.retries} retries: {e}", exc_info=True)
        self.update_state(state='FAILURE', meta={'status': 'failed', 'error': str(e), 'traceback': traceback.format_exc()})
        # Removed raise: Let update_state handle marking the task as failed.
        # Re-raising the raw exception can cause serialization issues with the result backend.


# --- is_valid_url function (Uses urlparse) ---
def is_valid_url(url, base_domain):
    """Checks if a URL is valid, absolute, and belongs to the base domain."""
    try:
        parsed = urlparse(url)
        # Check scheme, netloc, and if it matches the base domain
        return (parsed.scheme in ['http', 'https'] and
                parsed.netloc == base_domain)
    except ValueError:
        # Handle potential errors from urlparse on malformed URLs
        return False

# --- Removed initialize_clients and find_similar_chunks as they belong to RAG/Ingestion ---
