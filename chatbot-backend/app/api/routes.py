# app/api/routes.py
import uuid
import threading
import secrets # Import secrets module for key generation
import os
import json
import logging # Add logging
import uuid # Add uuid for unique filenames
# import asyncio # No longer needed here for discovery
import time
import mimetypes # For image validation
from io import BytesIO # For image validation
from flask import request, jsonify, current_app, Response, stream_with_context, g, url_for # Import g and url_for
from werkzeug.utils import secure_filename
from app import db
from app.models import User, Chatbot, VectorIdMapping, ChatMessage, DetailedFeedback # Import VectorIdMapping and ChatMessage, DetailedFeedback
from datetime import datetime # Import datetime
from PIL import Image as PILImage # Use alias to avoid conflict if Image model exists
from PIL import UnidentifiedImageError
from app.api import bp
# from flask_jwt_extended import jwt_required, get_jwt_identity # No longer needed here
# Import Celery tasks
from app.services.discovery import run_discovery_task # This is now a Celery task
from app.services.ingestion import run_ingestion_task # This is now a Celery task
# Deletion tasks imported within functions to avoid circular import
import queue # Keep for queue.Empty exception
# Import the RAG Service
from app.services.rag_service import RagService
from app.services.rag_service import RagService, SourceNotFoundError, VertexAIDeletionError, DatabaseCleanupError # Import RAG exceptions
# Import celery app instance for result checking
from celery_worker import celery_app # Import from celery_worker.py
from celery.result import AsyncResult
from app import limiter # Import the limiter instance from __init__
from config import Config # Import Config to access constants
import redis # Import redis for pubsub
from functools import wraps # For API key decorator
from werkzeug.security import generate_password_hash, check_password_hash # For API key hashing
from app.services.summarization_service import SummarizationService # Import the new service


# Language detection/translation functions removed from language_service
# from app.services.language_service import detect_language, translate_text # Import the functions
# --- Import shared SSE utilities ---
# *** Assuming sse_utils.py exists in the app folder ***
# We still import push_status_update, but sse_message_queue is no longer used here.
try:
    from app.sse_utils import push_status_update, SSE_CHANNEL, redis_client as sse_redis_client # Import channel and client
except ImportError:
     print("ERROR: Cannot import from app.sse_utils. SSE will not work.")
     # Define dummies to prevent further crashes
     def push_status_update(chatbot_id, status, client_id): pass
     SSE_CHANNEL = "chatbot-status-updates" # Define fallback channel name
     sse_redis_client = None # Define fallback client
# -----------------------------------

# --- API Key Authentication Decorator ---
def require_api_key(f):
    """Decorator to protect routes requiring a valid chatbot API key."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Allow OPTIONS requests to pass through for CORS preflight
        if request.method == 'OPTIONS':
            return f(*args, **kwargs) # Or potentially just `None` if Flask-CORS handles it

        api_key = None
        chatbot_id = kwargs.get('chatbot_id') # Get chatbot_id from URL rule
        if not chatbot_id:
            # This should not happen if the decorator is used on routes with <int:chatbot_id>
            current_app.logger.error("API key decorator used on route without chatbot_id argument.")
            return jsonify({"error": "Internal server configuration error"}), 500

        # Check Authorization header first
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            api_key = auth_header.split(' ')[1]
        # Optional: Check for X-Api-Key header as fallback
        # if not api_key:
        #     api_key = request.headers.get('X-Api-Key')

        if not api_key:
            current_app.logger.warning(f"API access attempt missing API key for chatbot {chatbot_id}.")
            return jsonify({"error": "Missing API key"}), 401

        # Find chatbot by ID first
        chatbot = Chatbot.query.filter_by(id=chatbot_id).first()
        if not chatbot:
            # Log that the chatbot ID itself wasn't found
            current_app.logger.warning(f"API access attempt for non-existent chatbot ID: {chatbot_id}.")
            # Return 404 Not Found if the chatbot ID doesn't exist
            return jsonify({"error": "Chatbot not found"}), 404

        # Now verify the provided API key against the stored hash
        stored_hash = chatbot.api_key
        # Use check_password_hash to compare the received plaintext key against the stored hash
        if not stored_hash or not check_password_hash(stored_hash, api_key):
            current_app.logger.warning(f"API access attempt with invalid key for chatbot {chatbot_id}.")
            return jsonify({"error": "Invalid or unauthorized API key"}), 403 # Forbidden

        # Inject chatbot object into request context if needed by the route
        g.chatbot = chatbot # Store the validated chatbot object in Flask's 'g' context

        return f(*args, **kwargs)
    return decorated_function
# ------------------------------------


# --- In-memory storage for discovery results (Simple approach) ---
discovery_tasks_results = {} # No longer used with Celery backend
discovery_tasks_lock = threading.Lock() # No longer used with Celery backend
# ----------------------------------------------------------------


# --- Constants and File Helpers ---
UPLOAD_FOLDER = 'uploads' # For general file uploads
LOGO_UPLOAD_FOLDER = os.path.join('app', 'static', 'logos') # Relative to chatbot-backend
AVATAR_UPLOAD_FOLDER_BASE = os.path.join('app', 'static', 'avatars') # Base directory for avatars
LAUNCHER_ICON_UPLOAD_FOLDER_BASE = os.path.join('app', 'static', 'launcher_icons') # Base directory for launcher icons
ALLOWED_LOGO_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'svg'}
ALLOWED_AVATAR_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'} # Define allowed avatar types
ALLOWED_LAUNCHER_ICON_EXTENSIONS = {'png', 'svg', 'ico', 'jpg', 'jpeg'} # Define allowed launcher icon types
MAX_LOGO_SIZE_MB = 2
MAX_AVATAR_SIZE_MB = 1 # Define max avatar size (e.g., 1MB)
MAX_LAUNCHER_ICON_SIZE_MB = 0.5 # Define max launcher icon size (e.g., 0.5MB)

# Ensure directories exist on app startup (or at least before first request)

# --- Function to initialize/get RAG service instance ---
# Stores the instance on the app object to ensure it's created only once per process.
def get_rag_service():
    """Gets or creates the singleton RagService instance for the application."""
    # Use app.extensions for storing shared instances
    if not hasattr(current_app, 'extensions') or 'rag_service' not in current_app.extensions:
        current_app.logger.info("Initializing RAG Service instance for the first time for this app process...")
        try:
            # Ensure extensions dict exists
            if not hasattr(current_app, 'extensions'):
                current_app.extensions = {}
            # Create and store the instance
            current_app.extensions['rag_service'] = RagService(current_app.logger)
            current_app.logger.info("RAG Service initialized successfully for this app process.")
        except Exception as e:
             current_app.logger.error(f"CRITICAL: Failed to initialize RagService for this app process: {e}", exc_info=True)
             # Store None to indicate failure for this process, prevent repeated attempts
             if not hasattr(current_app, 'extensions'):
                 current_app.extensions = {}
             current_app.extensions['rag_service'] = None
             raise RuntimeError(f"Failed to initialize RagService: {e}") # Re-raise

    # Check if initialization failed previously in this process
    rag_instance = current_app.extensions.get('rag_service')
    if rag_instance is None:
        # This checks if the stored value is None due to a previous init failure
        raise RuntimeError("RAG Service failed previous initialization attempt for this app process.")

    return rag_instance
# --------------------------------------------------------

def allowed_file(filename, allowed_extensions):
    """Checks if a filename has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

# --- Login Endpoint ---
@bp.route('/login', methods=['POST'])
@limiter.limit("10 per minute") # Stricter limit for login attempts
def login():
    data = request.get_json()
    if not data or not data.get('email'):
        return jsonify({"error": "Email is required"}), 400

    email = data['email']
    user = User.query.filter_by(email=email).first()

    if not user:
        try:
            user = User(email=email)
            db.session.add(user)
            db.session.commit()
            current_app.logger.info(f"New user created: {email}, client_id: {user.client_id}")
        except Exception as e:
            db.session.rollback()
            # Catch potential specific DB errors if needed, e.g., IntegrityError
            current_app.logger.error(f"Failed to create user {email}: {e}", exc_info=True)
            return jsonify({"error": "Database error creating user account"}), 500
    else:
        current_app.logger.info(f"User logged in: {email}, client_id: {user.client_id}")

    return jsonify({"client_id": user.client_id}), 200

# --- Create Chatbot Endpoint ---
@bp.route('/chatbots', methods=['POST'])
@limiter.limit("30 per hour") # Limit creation rate
# @jwt_required() # Removed JWT requirement
def create_chatbot():
    # --- Get Data (Handle JSON or Form data for non-file fields) ---
    client_id = None
    name = None
    use_url = False
    url_value = ''
    use_sitemap = False
    sitemap_value = ''
    use_files = False
    selected_urls_input = '[]' # Default to empty list string

    widget_primary_color = None
    widget_text_color = None
    widget_welcome_message = None

    if request.is_json:
        data = request.get_json()
        if not data:
             return jsonify({"error": "Request body is missing or not valid JSON"}), 400
        client_id = data.get('client_id')
        name = data.get('name')
        use_url = str(data.get('useUrlSource', 'false')).lower() == 'true'
        url_value = data.get('sourceValueUrl', '') if use_url else ''
        use_sitemap = str(data.get('useSitemapSource', 'false')).lower() == 'true'
        sitemap_value = data.get('sourceValueSitemap', '') if use_sitemap else ''
        use_files = str(data.get('useFiles', 'false')).lower() == 'true'
        selected_urls_input = data.get('selected_urls', []) # Get list directly if JSON
        # Get widget customization fields from JSON
        widget_primary_color = data.get('widget_primary_color')
        widget_text_color = data.get('widget_text_color')
        widget_welcome_message = data.get('widget_welcome_message')
        # Get advanced RAG setting from JSON
        advanced_rag_enabled = data.get('advanced_rag_enabled', False) # Default to False
        current_app.logger.debug("Processing create_chatbot request with JSON data.")
    else:
        # Assume form data if not JSON
        client_id = request.form.get('client_id')
        name = request.form.get('name')
        use_url = request.form.get('useUrlSource', 'false').lower() == 'true'
        url_value = request.form.get('sourceValueUrl', '') if use_url else ''
        use_sitemap = request.form.get('useSitemapSource', 'false').lower() == 'true'
        sitemap_value = request.form.get('sourceValueSitemap', '') if use_sitemap else ''
        use_files = request.form.get('useFiles', 'false').lower() == 'true'
        selected_urls_input = request.form.get('selected_urls', '[]') # Get string if form data
        # Get widget customization fields from Form
        widget_primary_color = request.form.get('widget_primary_color')
        widget_text_color = request.form.get('widget_text_color')
        widget_welcome_message = request.form.get('widget_welcome_message')
        # Get advanced RAG setting from Form
        advanced_rag_enabled = request.form.get('advanced_rag_enabled', 'false').lower() == 'true' # Default to False
        current_app.logger.debug("Processing create_chatbot request with Form data.")

    # --- Validate required fields ---
    if not client_id: return jsonify({"error": "client_id is required"}), 400
    if not name: return jsonify({"error": "name is required"}), 400
 
    # --- Add logging to check the received client_id ---
    current_app.logger.debug(f"Attempting to find user with client_id: '{client_id}' (Type: {type(client_id)})")
    # ----------------------------------------------------

    # Find user by client_id
    # Replace User with your actual User model
    user = User.query.filter_by(client_id=client_id).first()
    if not user:
        current_app.logger.warning(f"Chatbot creation attempt with invalid client_id: {client_id}")
        return jsonify({"error": "Invalid client_id"}), 404

    current_app.logger.info(f"Chatbot creation request for client_id: {client_id} (User: {user.email})")

    # Check chatbot limit using client_id
    # Replace Chatbot with your actual Chatbot model
    chatbot_count = Chatbot.query.filter_by(client_id=client_id).count()
    # TODO: Replace '3' with a limit from the user's subscription plan if applicable
    if chatbot_count >= 3:
        current_app.logger.warning(f"Chatbot limit reached for client_id: {client_id}")
        return jsonify({"error": "Maximum chatbots (3) reached."}), 400

    # --- Parse selected_urls (could be list from JSON or string from form) ---
    selected_urls = []
    try:
        if isinstance(selected_urls_input, str):
            parsed_urls = json.loads(selected_urls_input)
        elif isinstance(selected_urls_input, list):
             parsed_urls = selected_urls_input # Already a list
        else:
             parsed_urls = [] # Default for other types

        if isinstance(parsed_urls, list):
            selected_urls = [str(url) for url in parsed_urls if isinstance(url, str)]
        else:
             current_app.logger.warning(f"Parsed selected_urls is not a list: {type(parsed_urls)}")

    except json.JSONDecodeError:
        current_app.logger.warning(f"Failed to parse selected_urls input string: {selected_urls_input}")
        selected_urls = []


    # --- File Handling (request.files is independent of JSON/form body) ---
    saved_file_basenames = [] # Store basenames of successfully saved files
    uploaded_file_paths_to_cleanup = [] # Store full paths for potential cleanup
    # --- ADD RAW FORM/FILES DEBUGGING ---
    current_app.logger.debug(f"Raw request.form: {request.form}")
    current_app.logger.debug(f"Raw request.files: {request.files}")
    # --- END RAW DEBUGGING ---
    if use_files:
        current_app.logger.debug(f"File upload requested. Checking request.files...")
        # Make sure the key 'files' matches the name attribute in the frontend <input type="file" name="files">
        if 'files' not in request.files:
            current_app.logger.error("File upload error: 'files' key missing in request.files")
            return jsonify({"error": "No file part named 'files' found"}), 400

        files = request.files.getlist('files')
        current_app.logger.debug(f"Received files via request.files.getlist('files'): {[f.filename for f in files]}")

        if not files or all(f.filename == '' for f in files): # Check if all filenames are empty
            current_app.logger.error("File upload error: No files selected or file list is empty.")
            return jsonify({"error": "No files selected"}), 400

        # Ensure upload folder exists right before saving
        if not os.path.exists(UPLOAD_FOLDER):
             try:
                 os.makedirs(UPLOAD_FOLDER)
                 current_app.logger.info(f"Created upload folder: {UPLOAD_FOLDER}")
             except OSError as e:
                 current_app.logger.error(f"Failed to create upload directory '{UPLOAD_FOLDER}': {e}")
                 return jsonify({"error": f"Could not create upload dir: {e}"}), 500

        for file in files:
            if file and file.filename != '' and allowed_file(file.filename, {'txt', 'pdf', 'docx'}): # Use specific extensions for data
                filename = secure_filename(file.filename)
                # Use UUID in the basename for uniqueness on the filesystem
                unique_basename = f"{uuid.uuid4()}_{filename}"
                file_path = os.path.join(UPLOAD_FOLDER, unique_basename)
                # <<< Log 3: Before file.save >>>
                current_app.logger.debug(f"Attempting to save file: {filename} to path: {file_path}")
                try:
                    file.save(file_path)
                    # <<< Log 4: After file.save >>>
                    current_app.logger.debug(f"Successfully saved file: {file_path}")
                    saved_file_basenames.append(unique_basename) # Store the unique basename
                    uploaded_file_paths_to_cleanup.append(file_path) # Store full path for cleanup on error
                    # <<< Log 5: After appending basename >>>
                    current_app.logger.debug(f"Added unique basename to list. Current list: {saved_file_basenames}")
                except Exception as e:
                    current_app.logger.error(f"Failed to save file {filename}: {e}", exc_info=True)
                    # Clean up any files saved *before* this error in the current request
                    current_app.logger.warning(f"Cleaning up partially saved files due to error saving {filename}")
                    for path in uploaded_file_paths_to_cleanup:
                         if os.path.exists(path):
                             try: os.remove(path)
                             except Exception as cleanup_e: current_app.logger.error(f"Error cleaning up {path}: {cleanup_e}")
                    return jsonify({"error": f"Could not save file {filename}"}), 500
            elif file and file.filename != '': # This case handles if allowed_file() returned False or filename was empty but file existed
                 current_app.logger.warning(f"File type not allowed or invalid filename, skipping: {file.filename}")
                 # No need to clean up here, as disallowed files aren't saved, but we should reject the request.
                 # Clean up any previously SAVED files from this SAME request
                 current_app.logger.warning(f"Cleaning up any previously saved files due to disallowed file type {file.filename}")
                 for path in uploaded_file_paths_to_cleanup:
                     if os.path.exists(path):
                         try: os.remove(path)
                         except Exception as cleanup_e: current_app.logger.error(f"Error cleaning up {path}: {cleanup_e}")
                 return jsonify({"error": f"File type not allowed: {file.filename}"}), 400
            # If file is None or filename is '', it's skipped silently

        # <<< Log 6: After file loop >>>
        current_app.logger.debug(f"Finished file processing loop. Final saved_file_basenames: {saved_file_basenames}")

    # --- Source Details Consolidation ---
    # This uses the final state of saved_file_basenames
    source_details = {
        'original_url': url_value, 'original_sitemap': sitemap_value,
        'selected_urls': selected_urls if (use_url or use_sitemap) else [],
        'files_uploaded': saved_file_basenames # Use the list of saved basenames
    }
    source_types = []
    # Determine type based on what will be processed
    if selected_urls:
        source_types.append('Web_Filtered') # Keep it simple
    elif url_value or sitemap_value: # Direct processing (if filtering somehow skipped)
        source_types.append('Web_Direct')
    # Correctly check if files were uploaded for source type
    if saved_file_basenames: # Check the list of successfully saved files
        source_types.append('Files')
    combined_source_type = '+'.join(source_types) if source_types else 'None'

    chatbot_id_for_response = None
    try:
        # --- Create Chatbot Record ---
        # Generate plaintext API key
        plaintext_api_key = secrets.token_urlsafe(32)
        # Hash the API key for storage
        hashed_api_key = generate_password_hash(plaintext_api_key)

        new_chatbot = Chatbot( # Replace Chatbot with your actual model
            name=name,
            user_id=user.id,
            client_id=client_id,
            status='Queued',
            source_type=combined_source_type,
            source_details=json.dumps(source_details), # Store details with only basenames
            api_key=hashed_api_key, # Store the hashed key
            # Add widget customization fields
            widget_primary_color=widget_primary_color,
            widget_text_color=widget_text_color,
            widget_welcome_message=widget_welcome_message
        )
        db.session.add(new_chatbot) # Replace db with your actual db instance
        db.session.commit()
        chatbot_id = new_chatbot.id
        chatbot_id_for_response = chatbot_id
        current_app.logger.info(f"Created Chatbot record ID: {chatbot_id} for client: {client_id}. Stored hashed API key.")

        # --- Push initial status using imported function ---
        push_status_update(chatbot_id, 'Queued', client_id) # Assuming push_status_update exists
        # --------------------------------------------------

        # --- Queue Celery Ingestion Task ---
        # Pass the source_details dictionary containing the unique basenames
        # The Celery task will reconstruct the full path using UPLOAD_FOLDER and the basenames
        # <<< Log 7: Before queuing task >>>
        current_app.logger.debug(f"Queuing ingestion task for chatbot {chatbot_id} with source_details: {source_details}")
        task = run_ingestion_task.delay(
            chatbot_id,
            client_id,
            source_details # Pass the dict containing the basenames
        )
        current_app.logger.info(f"Queued ingestion task (ID: {task.id}) for chatbot ID: {chatbot_id}")
        # Optionally store task.id in Chatbot model if you need to track it

    except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Error during chatbot creation (DB save or Celery queueing) for client {client_id}: {e}", exc_info=True)
            # Attempt cleanup even on DB error - use the cleanup list
            current_app.logger.warning(f"Cleaning up files due to error during DB save or Celery queueing.")
            for path in uploaded_file_paths_to_cleanup:
                if path and os.path.exists(path):
                    try: os.remove(path)
                    except Exception as cleanup_e: current_app.logger.error(f"Error cleaning up {path}: {cleanup_e}")
            # Return 500 error
            return jsonify({"error": "Failed configuration save or processing start"}), 500

    # --- Return Success Response ---
    # Return the PLAINTEXT API key ONCE upon creation as requested
    return jsonify({
        "message": "Chatbot creation initiated.",
        "chatbot_id": chatbot_id_for_response,
        "status": "Queued",
        "api_key": plaintext_api_key # Return the plaintext key for the user to copy
    }), 202

# --- SSE Endpoint ---
@bp.route('/chatbots/status-stream')
# Rate limiting SSE is tricky, might limit connections instead if needed. Skipping direct limit for now.
def chatbot_status_stream():
    request_client_id = request.args.get('clientId')
    if not request_client_id:
        return Response("clientId query parameter is required", status=400, mimetype='text/plain')

    current_app.logger.info(f"SSE Client connected: {request.remote_addr}, ClientID: {request_client_id}")

    def event_stream():
        if not sse_redis_client:
            current_app.logger.error(f"SSE Stream: Cannot connect for {request_client_id}, Redis client unavailable.")
            # Optionally yield an error message to the client?
            return # Stop the generator

        pubsub = None
        try:
            # Create a new pubsub object for this connection
            pubsub = sse_redis_client.pubsub(ignore_subscribe_messages=True)
            pubsub.subscribe(SSE_CHANNEL)
            current_app.logger.info(f"SSE Stream: Subscribed client {request_client_id} to Redis channel '{SSE_CHANNEL}'")

            # Send a keep-alive initially
            yield ": keep-alive\n\n"

            for message in pubsub.listen():
                # message format: {'type': 'message', 'pattern': None, 'channel': 'channel_name', 'data': 'your_json_string'}
                if message['type'] == 'message':
                    # Decode data if it's bytes (common with Redis)
                    if isinstance(message['data'], bytes):
                        message_str = message['data'].decode('utf-8')
                    else:
                        message_str = message['data']

                    try:
                        message_data = json.loads(message_str)
                        sse_client_id = message_data.get('client_id')
                        # Filter messages for the specific client connected to this stream
                        if sse_client_id == request_client_id:
                            yield f"data: {message_str}\n\n"
                            current_app.logger.debug(f"SSE Sent to {request_client_id}: {message_str[:100]}...")
                    except json.JSONDecodeError:
                        current_app.logger.error(f"SSE Stream: Failed to parse message from Redis: {message_str}")
                    except Exception as e:
                        current_app.logger.error(f"SSE Stream: Error processing message for {request_client_id}: {e}", exc_info=True)
                # Add periodic keep-alive if needed, though pubsub.listen might block efficiently
                # Consider using pubsub.get_message(timeout=...) in a loop if more control is needed

        except redis.exceptions.ConnectionError as redis_err:
             current_app.logger.error(f"SSE Stream: Redis connection error for {request_client_id}: {redis_err}", exc_info=True)
        except GeneratorExit:
            # This is expected when the client disconnects
            current_app.logger.info(f"SSE Stream: Client disconnected normally: {request.remote_addr}, ClientID: {request_client_id}")
        except Exception as e:
            current_app.logger.error(f"SSE Stream: Unexpected error in event_stream for {request_client_id}: {e}", exc_info=True)
        finally:
            if pubsub:
                try:
                    pubsub.unsubscribe(SSE_CHANNEL)
                    pubsub.close() # Close the pubsub connection
                    current_app.logger.info(f"SSE Stream: Unsubscribed and closed pubsub for {request_client_id}")
                except Exception as close_e:
                     current_app.logger.error(f"SSE Stream: Error closing pubsub for {request_client_id}: {close_e}")
            current_app.logger.info(f"SSE Stream: Connection closing routine finished for {request_client_id}")

    headers = {
        'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', 'Connection': 'keep-alive',
        # Add CORS headers if frontend is on a different origin
        'Access-Control-Allow-Origin': '*', # Adjust as needed for security
    }
    return Response(stream_with_context(event_stream()), headers=headers)

# --- Discovery Endpoint (Starts Background Task) ---
@bp.route('/discover-links', methods=['POST'])
@limiter.limit("20 per hour") # Limit how often discovery can be started
def discover_links_start():
    data = request.get_json()
    if not data: return jsonify({"error": "Request body must be JSON"}), 400
    source_url = data.get('source_url')
    source_type = data.get('source_type')
    if not source_url or not source_type or source_type not in ['url', 'sitemap']:
        return jsonify({"error": "source_url and valid type ('url' or 'sitemap') required"}), 400

    # Use Celery to run discovery in the background
    try:
        # Create a unique task ID for tracking (optional but good practice)
        celery_task_id = f"discover_{uuid.uuid4()}" # Create a trackable ID
        task = run_discovery_task.apply_async(
            args=[celery_task_id, source_url, source_type], # Add task_id here
            task_id=celery_task_id
        )
        current_app.logger.info(f"Discovery task queued with ID: {task.id} (Trackable ID: {celery_task_id}) for URL: {source_url}")
        return jsonify({"message": "Link discovery started.", "task_id": task.id}), 202
    except Exception as e:
        current_app.logger.error(f"Failed to queue discovery task for {source_url}: {e}", exc_info=True)
        return jsonify({"error": "Failed to start discovery process"}), 500

# --- Discovery Results Endpoint (Checks Task Status) ---
@bp.route('/discover-links/<task_id>', methods=['GET'])
def get_discovery_results(task_id):
    """Checks the status and retrieves results of a discovery task."""
    try:
        task_result = AsyncResult(task_id, app=celery_app)
        state = task_result.state # Get the state safely

        # Initialize response data with the safe state
        response_data = {
            "task_id": task_id,
            "status": state, # Use the safe state string
            "result": None
        }

        if state == 'SUCCESS':
            # Task completed successfully
            response_data["result"] = task_result.get() # Get the actual result
            current_app.logger.debug(f"Discovery task {task_id} completed successfully.")
            return jsonify(response_data), 200
        elif state == 'FAILURE':
            # Task failed - retrieve error info safely
            error_info = "Task failed, but error details could not be retrieved."
            tb = None
            try:
                # Try accessing .info (might contain exception) and .traceback
                if task_result.info:
                    if isinstance(task_result.info, Exception):
                         error_info = str(task_result.info)
                    else:
                         # Use repr for non-exception info to get a string representation
                         error_info = repr(task_result.info)
                tb = task_result.traceback # Get traceback string if available
                # Log the failure details we could retrieve
                current_app.logger.error(f"Discovery task {task_id} failed. State: {state}, Info: {error_info}, Traceback available: {'Yes' if tb else 'No'}")
            except Exception as info_exc:
                # Catch errors trying to access .info or .traceback
                current_app.logger.error(f"Error retrieving failure details for task {task_id}: {info_exc}", exc_info=True)

            response_data["result"] = {"error": error_info, "traceback": tb}
            # Return 200 OK: The API request to check status succeeded,
            # even though the underlying task failed. The 'status' field indicates the failure.
            # Note: Changed from 500 in original code to 200 to align with frontend expectations
            # that the status check itself didn't fail, only the underlying task.
            return jsonify(response_data), 200
        else:
            # Task is PENDING, STARTED, RETRY, etc.
            current_app.logger.debug(f"Discovery task {task_id} status: {state}")
            # Return 202 Accepted: The task is not yet successfully completed.
            return jsonify(response_data), 202

    except Exception as e:
        # This outer exception catches errors in *this function* itself (e.g., AsyncResult init failed)
        current_app.logger.error(f"Error checking discovery task status for {task_id}: {e}", exc_info=True)
        # Return 500 Internal Server Error as the status check itself failed
        return jsonify({"error": "Failed to get task status", "details": str(e)}), 500


# --- Get Chatbots Endpoint ---
@bp.route('/chatbots', methods=['GET'])
@limiter.limit("120 per minute") # Allow frequent listing
# @jwt_required() # Removed JWT requirement
def get_chatbots():
    client_id = request.args.get('client_id')
    if not client_id:
        return jsonify({"error": "client_id query parameter is required"}), 400

    # Find user by client_id
    user = User.query.filter_by(client_id=client_id).first()
    if not user:
        return jsonify({"error": "Invalid client_id"}), 404

    # Get chatbots associated with the user's client_id
    chatbots = Chatbot.query.filter_by(client_id=client_id).order_by(Chatbot.created_at.desc()).all()

    chatbot_list = [
        {
            "id": bot.id, "name": bot.name, "status": bot.status,
            "created_at": bot.created_at.isoformat() if bot.created_at else None,
            "source_type": bot.source_type
        }
        for bot in chatbots
    ]
    return jsonify(chatbot_list), 200

# --- Get Chatbot Details Endpoint ---
@bp.route('/chatbots/<int:chatbot_id>', methods=['GET'])
@limiter.limit("120 per minute")
def get_chatbot_details(chatbot_id):
    """Gets details for a specific chatbot."""
    # Get client_id from query parameter for verification
    client_id = request.args.get('client_id')
    if not client_id:
        current_app.logger.warning(f"get_chatbot_details request missing client_id for chatbot {chatbot_id}")
        return jsonify({"error": "client_id query parameter is required"}), 400

    chatbot = db.session.get(Chatbot, chatbot_id)
    if not chatbot:
        return jsonify({"error": "Chatbot not found"}), 404

    # Ownership check:
    user = User.query.filter_by(client_id=client_id).first()
    if not user or chatbot.client_id != user.client_id:
        current_app.logger.warning(f"Unauthorized attempt to access details for chatbot {chatbot_id} by client_id {client_id}")
        return jsonify({"error": "Forbidden"}), 403

    # --- Log the image_analysis_enabled value ---
    current_app.logger.info(f"Chatbot {chatbot_id} - image_analysis_enabled from DB: {chatbot.image_analysis_enabled}, Type: {type(chatbot.image_analysis_enabled)}") # Use current_app.logger
    # -------------------------------------------

    # Prepare data for response (exclude sensitive info like API key by default)
    # Consider adding a to_dict() method to your Chatbot model for cleaner serialization
    chatbot_data = {
        "id": chatbot.id,
        "name": chatbot.name,
        "status": chatbot.status,
        "source_type": chatbot.source_type,
        "source_details": json.loads(chatbot.source_details or '{}'), # Parse JSON details
        "created_at": chatbot.created_at.isoformat() if chatbot.created_at else None,
        "updated_at": chatbot.updated_at.isoformat() if chatbot.updated_at else None,
        "widget_primary_color": chatbot.widget_primary_color,
        "widget_text_color": chatbot.widget_text_color,
        "widget_welcome_message": chatbot.widget_welcome_message,
        "logo_path": chatbot.logo_path, # Relative path
        "logo_url": url_for('static', filename=chatbot.logo_path, _external=False) if chatbot.logo_path else None, # Full URL
        "avatar_path": chatbot.avatar_path, # Relative path for avatar
        "avatar_url": url_for('static', filename=chatbot.avatar_path, _external=False) if chatbot.avatar_path else None, # Full URL for avatar
        'source_document_language': chatbot.source_document_language, # Added for multilingual support

        "launcher_text": chatbot.launcher_text,
        "widget_background_color": chatbot.widget_background_color,
        "user_message_color": chatbot.user_message_color,
        "bot_message_color": chatbot.bot_message_color,
        "input_background_color": chatbot.input_background_color,
        # Include feature toggles and other settings
        "text_chat_enabled": chatbot.text_chat_enabled,
        "text_language": chatbot.text_language,
        "file_uploads_enabled": chatbot.file_uploads_enabled,
        "allowed_file_types": chatbot.allowed_file_types,
        "max_file_size_mb": chatbot.max_file_size_mb,
        "save_history_enabled": chatbot.save_history_enabled,
        "history_retention_days": chatbot.history_retention_days,
        "allow_user_history_clearing": chatbot.allow_user_history_clearing,
        "feedback_thumbs_enabled": chatbot.feedback_thumbs_enabled,
        "detailed_feedback_enabled": chatbot.detailed_feedback_enabled,
        "base_prompt": chatbot.base_prompt,
        "knowledge_adherence_level": chatbot.knowledge_adherence_level,
        "consent_message": chatbot.consent_message,
        "consent_required": chatbot.consent_required,
        # Add index operation status if available
        "index_operation_status": chatbot.get_index_operation_status(),
        # Add voice settings
        "voice_enabled": chatbot.voice_enabled,
        "image_analysis_enabled": chatbot.image_analysis_enabled,
        "voice_activity_detection_enabled": bool(chatbot.vad_enabled), # Convert DB int (0/1) to boolean
        'summarization_enabled': chatbot.summarization_enabled, # Added
        'allowed_scraping_domains': chatbot.allowed_scraping_domains, # Added
        'advanced_rag_enabled': chatbot.advanced_rag_enabled # Add the new flag
    }
    return jsonify(chatbot_data), 200

# --- Get Widget Config Endpoint ---
@bp.route('/chatbots/<int:chatbot_id>/widget-config', methods=['GET'])
@limiter.limit("300 per minute") # Allow more frequent config fetches for widgets
@require_api_key # Use the decorator to authenticate via API key
def get_widget_config(chatbot_id):
    """
    Provides the necessary configuration for the chatbot widget.
    Requires API Key authentication.
    The chatbot object is available via g.chatbot thanks to the decorator.
    """
    chatbot = g.chatbot # Get the chatbot object validated by the decorator

    # Prepare configuration data for the widget
    config_data = {
        "chatbot_id": chatbot.id,
        "name": chatbot.name,
        "primary_color": chatbot.widget_primary_color,
        "text_color": chatbot.widget_text_color,
        "welcome_message": chatbot.widget_welcome_message,
        "logo_url": url_for('static', filename=chatbot.logo_path, _external=False) if chatbot.logo_path else None,
        "avatar_url": url_for('static', filename=chatbot.avatar_path, _external=False) if chatbot.avatar_path else None, # Add avatar URL
        "launcher_icon_url": url_for('static', filename=chatbot.launcher_icon_path, _external=False) if chatbot.launcher_icon_path else None, # Add launcher icon URL
        "launcher_text": chatbot.launcher_text,
        "widget_position": chatbot.widget_position, # Added widget position
        "background_color": chatbot.widget_background_color,
        "user_message_color": chatbot.user_message_color,
        "bot_message_color": chatbot.bot_message_color,
        "input_background_color": chatbot.input_background_color,
        "voice_enabled": chatbot.voice_enabled,
        "vad_enabled": chatbot.vad_enabled,
        "text_chat_enabled": chatbot.text_chat_enabled,
        "text_language": chatbot.text_language,
        "file_uploads_enabled": chatbot.file_uploads_enabled,
        "allowed_file_types": chatbot.allowed_file_types,
        "max_file_size_mb": chatbot.max_file_size_mb,
        "save_history_enabled": chatbot.save_history_enabled,
        "history_retention_days": chatbot.history_retention_days,
        "allow_user_history_clearing": chatbot.allow_user_history_clearing,
        "feedback_thumbs_enabled": chatbot.feedback_thumbs_enabled,
        "detailed_feedback_enabled": chatbot.detailed_feedback_enabled,
        # Add new UI toggle fields
        'show_widget_header': chatbot.show_widget_header,
        'show_message_timestamps': chatbot.show_message_timestamps,
        'start_open': chatbot.start_open, # Add the new field here
        "show_typing_indicator": chatbot.show_typing_indicator,
        "default_error_message": chatbot.default_error_message, # Add the new error message field
        "response_delay_ms": chatbot.response_delay_ms, # Add response delay
        'enable_sound_notifications': chatbot.enable_sound_notifications, # Include sound notification setting
        "consent_message": chatbot.consent_message,
        "consent_required": chatbot.consent_required,
        'image_analysis_enabled': chatbot.image_analysis_enabled, # Added image analysis flag
        'summarization_enabled': chatbot.summarization_enabled, # Added
        # allowed_scraping_domains is not needed by the widget config
        # Add other relevant config fields here
    }
    current_app.logger.debug(f"Widget config including image_analysis_enabled: {chatbot.image_analysis_enabled} for chatbot {chatbot_id}")
    return jsonify(config_data), 200

# --- Message Feedback Endpoint ---
@bp.route('/messages/<int:message_id>/feedback', methods=['POST'])
@require_api_key # Requires API key associated with the chatbot the message belongs to
def post_message_feedback(message_id):
    """Adds simple positive/negative feedback to a specific message (V2)."""
    chatbot = g.chatbot # Get chatbot from decorator context

    data = request.get_json()
    if not data or 'feedback_type' not in data:
        return jsonify({"error": "Missing 'feedback_type' in request body (expecting 'positive' or 'negative')"}), 400

    feedback_type_value = data['feedback_type'].lower()
    if feedback_type_value not in ['positive', 'negative']:
        return jsonify({"error": "Invalid feedback_type value. Must be 'positive' or 'negative'."}), 400

    message = ChatMessage.query.get(message_id)
    if not message:
        return jsonify({"error": "Message not found"}), 404

    # Authorization check: Ensure the message belongs to the chatbot identified by the API key
    if message.chatbot_id != chatbot.id:
        current_app.logger.warning(f"Unauthorized feedback attempt on message {message_id} by chatbot {chatbot.id}")
        return jsonify({"error": "Forbidden: Message does not belong to this chatbot"}), 403

    # Check if feedback is enabled for this chatbot
    if not chatbot.feedback_thumbs_enabled:
         return jsonify({"error": "Feedback is disabled for this chatbot"}), 403

    try:
        # Map V2 feedback_type to V1 thumb_feedback values for storage
        thumb_feedback_value = 'up' if feedback_type_value == 'positive' else 'down'

        message.thumb_feedback = thumb_feedback_value # Store 'up' or 'down'
        db.session.commit()
        current_app.logger.info(f"Feedback '{feedback_type_value}' (stored as '{thumb_feedback_value}') recorded for message {message_id}")
        return jsonify({"message": "Feedback recorded successfully"}), 200
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Database error saving feedback for message {message_id}: {e}", exc_info=True)
        return jsonify({"error": "Failed to save feedback"}), 500

# --- Detailed Feedback Endpoint ---
@bp.route('/api/feedback/detailed', methods=['POST'])
@limiter.limit("10 per hour") # Lower limit for detailed feedback
@require_api_key # Requires API key associated with the chatbot the message belongs to
def post_detailed_feedback():
    """Saves detailed feedback for a specific message."""
    from app.models import DetailedFeedback # Local import
    chatbot = g.chatbot # Get chatbot from decorator context

    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON request body"}), 400

    message_id = data.get('message_id')
    feedback_text = data.get('feedback_text')
    session_id = data.get('session_id') # Get session_id from request

    if not message_id or not feedback_text or not session_id:
        return jsonify({"error": "Missing required fields: 'message_id', 'feedback_text', 'session_id'"}), 400

    message = ChatMessage.query.get(message_id)
    if not message:
        return jsonify({"error": "Message not found"}), 404

    # Authorization check: Ensure the message belongs to the chatbot identified by the API key
    if message.chatbot_id != chatbot.id:
        current_app.logger.warning(f"Unauthorized detailed feedback attempt on message {message_id} by chatbot {chatbot.id}")
        return jsonify({"error": "Forbidden: Message does not belong to this chatbot"}), 403

    # Check if detailed feedback is enabled for this chatbot
    if not chatbot.detailed_feedback_enabled:
         return jsonify({"error": "Detailed feedback is disabled for this chatbot"}), 403

    # Check if feedback already exists for this message (optional, depends on requirements)
    existing_feedback = DetailedFeedback.query.filter_by(message_id=message_id).first()
    if existing_feedback:
        # Decide whether to update or reject (here we reject)
        return jsonify({"error": "Detailed feedback already submitted for this message"}), 409 # Conflict

    try:
        detailed_feedback = DetailedFeedback(
            message_id=message_id,
            session_id=session_id, # Store the session ID
            feedback_text=feedback_text
        )
        db.session.add(detailed_feedback)
        db.session.commit()
        current_app.logger.info(f"Detailed feedback saved for message {message_id} in session {session_id}")
        return jsonify({"message": "Detailed feedback saved successfully"}), 201 # Created
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Database error saving detailed feedback for message {message_id}: {e}", exc_info=True)
        return jsonify({"error": "Failed to save feedback due to server error"}), 500


# --- UPDATE Chatbot Endpoint ---
@bp.route('/chatbots/<int:chatbot_id>', methods=['PUT'])
@limiter.limit("30 per hour") # Limit update rate
def update_chatbot(chatbot_id): # noqa: C901 - Ignore complexity for now
    """
    Updates an existing chatbot's configuration, including name, sources, logo, avatar, widget settings,
    and feature toggles. Accepts multipart/form-data.
    """
    # Get client_id from form data for verification
    client_id = request.form.get('client_id')
    if not client_id:
        current_app.logger.warning(f"update_chatbot request missing client_id for chatbot {chatbot_id}")
        return jsonify({"error": "client_id is required in form data"}), 400

    chatbot = db.session.get(Chatbot, chatbot_id)
    if not chatbot:
        return jsonify({"error": "Chatbot not found"}), 404

    # Ownership check:
    user = User.query.filter_by(client_id=client_id).first()
    if not user or chatbot.client_id != user.client_id:
        current_app.logger.warning(f"Unauthorized attempt to update chatbot {chatbot_id} by client_id {client_id}")
        return jsonify({"error": "Forbidden"}), 403

    # --- Data extraction and initial setup ---
    newly_uploaded_basenames_for_ingestion = [] # Store unique basenames of *new* data files
    files_to_remove_basenames = []
    incoming_selected_urls = []
    files_changed = False
    urls_changed = False
    data_cleanup_paths = [] # Paths of newly uploaded DATA SOURCE files in case of error
    logo_file_saved = False # Flag to track if a new logo was saved
    new_logo_full_path = None # Full path for potential cleanup
    new_logo_relative_path = None # Relative path for DB update
    old_logo_path = chatbot.logo_path # Store old path before potential update
    delete_old_logo = False # Flag to delete old logo file after commit

    avatar_file_saved = False # Flag for avatar
    new_avatar_full_path = None # Full path for avatar cleanup
    new_avatar_relative_path = None # Relative path for avatar DB update
    old_avatar_path = chatbot.avatar_path # Store old avatar path
    delete_old_avatar = False # Flag to delete old avatar file

    try:
        # --- Get Basic Fields ---
        name = request.form.get('name', chatbot.name)
        launcher_text = request.form.get('launcher_text', chatbot.launcher_text)
        widget_primary_color = request.form.get('widget_primary_color', chatbot.widget_primary_color)
        widget_text_color = request.form.get('widget_text_color', chatbot.widget_text_color)
        widget_background_color = request.form.get('widget_background_color', chatbot.widget_background_color)
        user_message_color = request.form.get('user_message_color', chatbot.user_message_color)
        bot_message_color = request.form.get('bot_message_color', chatbot.bot_message_color)
        input_background_color = request.form.get('input_background_color', chatbot.input_background_color)
        widget_welcome_message = request.form.get('widget_welcome_message', chatbot.widget_welcome_message)
        default_error_message = request.form.get('default_error_message', chatbot.default_error_message) # Get the new error message field
        fallback_message = request.form.get('fallback_message', chatbot.fallback_message) # Get the new fallback message field
        # --- Handle Response Delay ---
        response_delay_ms_str = request.form.get('response_delay_ms')
        if response_delay_ms_str is not None:
            try:
                response_delay_ms_int = int(response_delay_ms_str)
                if response_delay_ms_int < 0:
                    return jsonify({"error": "response_delay_ms must be a non-negative integer"}), 400
                chatbot.response_delay_ms = response_delay_ms_int
            except ValueError:
                return jsonify({"error": "response_delay_ms must be a valid integer"}), 400
        # --- End Handle Response Delay ---


        # --- Update Widget Position ---
        source_doc_lang = request.form.get('source_document_language')
        if source_doc_lang is not None:
            # TODO: Add validation for language code format if needed
            chatbot.source_document_language = source_doc_lang
            current_app.logger.debug(f"Updating source_document_language to: {source_doc_lang}")

        allowed_positions = ['bottom-right', 'bottom-left', 'top-right', 'top-left']
        new_widget_position = request.form.get('widget_position')
        if new_widget_position:
            if new_widget_position in allowed_positions:
                chatbot.widget_position = new_widget_position
            else:
                # Log invalid value but don't necessarily fail the whole update
                current_app.logger.warning(f"Invalid widget_position '{new_widget_position}' provided for chatbot {chatbot_id}. Using existing value '{chatbot.widget_position}'.")
        else:
             # If not provided in the form, keep the existing value (or default if it was null)
             pass # No change needed if not provided

        # --- Feature Toggles Helper ---
        def form_to_bool_or_existing(form_key, existing_value):
            val = request.form.get(form_key)
            if val is None:
                return existing_value
            # Handle 'true', '1', 'on' as True (case-insensitive)
            val_lower = str(val).lower()
            return val_lower == 'true' or val_lower == '1' or val_lower == 'on'

        def form_checkbox_to_bool(form_key):
            """Converts form checkbox ('on' or 'true') to True, False otherwise."""
            # Handles standard HTML checkbox behavior (sends 'on' if checked, missing if not)
            # Also handles explicit 'true'/'false' for flexibility.
            value = request.form.get(form_key)
            if value is None: # Missing key means unchecked
                return False
            # Handle 'on' (standard checkbox), 'true' (explicit), case-insensitive
            value_lower = value.lower()
            return value_lower == 'true' or value_lower == '1' # Accept 'true' or '1'
        # --- Apply Feature Toggles ---
        # Process voice_enabled toggle
        voice_enabled_bool = form_to_bool_or_existing('voice_enabled', chatbot.voice_enabled)
        # *** Log received voice_enabled value before saving ***
        current_app.logger.debug(f"UPDATE Chatbot {chatbot_id}: Received voice_enabled form value, converted to boolean: {voice_enabled_bool}")
        # Explicitly convert boolean to integer 1 or 0 for the database
        chatbot.voice_enabled = 1 if voice_enabled_bool else 0
        current_app.logger.debug(f"UPDATE Chatbot {chatbot_id}: Assigning integer value to chatbot.voice_enabled: {chatbot.voice_enabled}")
        chatbot.text_chat_enabled = form_to_bool_or_existing('text_chat_enabled', chatbot.text_chat_enabled)
        chatbot.file_uploads_enabled = form_to_bool_or_existing('file_uploads_enabled', chatbot.file_uploads_enabled)
        chatbot.save_history_enabled = form_to_bool_or_existing('save_history_enabled', chatbot.save_history_enabled)
        chatbot.allow_user_history_clearing = form_to_bool_or_existing('allow_user_history_clearing', chatbot.allow_user_history_clearing)
        chatbot.feedback_thumbs_enabled = form_to_bool_or_existing('feedback_thumbs_enabled', chatbot.feedback_thumbs_enabled) # Renamed from feedback_enabled
        chatbot.detailed_feedback_enabled = form_to_bool_or_existing('detailed_feedback_enabled', chatbot.detailed_feedback_enabled)
        # Convert VAD boolean to integer (0/1) for DB consistency
        vad_enabled_bool = form_to_bool_or_existing('voice_activity_detection_enabled', chatbot.vad_enabled) # Corrected form key
        chatbot.vad_enabled = 1 if vad_enabled_bool else 0
        chatbot.start_open = form_checkbox_to_bool('start_open') # Handle widget start state
        chatbot.enable_sound_notifications = form_checkbox_to_bool('enable_sound_notifications') # Add sound notification toggle
        
        # --- Consent Settings ---
        consent_message = request.form.get('consent_message', chatbot.consent_message) # Get consent message, default to existing
        consent_required = form_checkbox_to_bool('consent_required') # Get consent required flag

        # --- History Retention ---
        history_retention_days_str = request.form.get('history_retention_days')
        if history_retention_days_str is not None:
            try:
                history_retention_days_int = int(history_retention_days_str)
                if history_retention_days_int < 0:
                    return jsonify({"error": "history_retention_days must be a non-negative integer"}), 400
                chatbot.history_retention_days = history_retention_days_int
            except ValueError:
                return jsonify({"error": "history_retention_days must be a valid integer"}), 400

        # --- Source Handling ---
        source_update_requested = False # Flag to track if any source changes were sent
        use_url = request.form.get('useUrlSource', 'false').lower() == 'true'
        url_value = request.form.get('sourceValueUrl', '')
        use_sitemap = request.form.get('useSitemapSource', 'false').lower() == 'true'
        sitemap_value = request.form.get('sourceValueSitemap', '')
        selected_urls_json = request.form.get('selected_urls') # Check if key exists, don't default
        removed_files_json = request.form.get('removed_files') # Check if key exists, don't default
        new_files = request.files.getlist('new_files') # Key 'new_files' must match frontend

        # Determine if a source update was explicitly requested by checking if relevant fields were sent
        if selected_urls_json is not None or removed_files_json is not None or new_files:
            source_update_requested = True
            current_app.logger.debug(f"Source update requested for chatbot {chatbot_id}.")

        # Only process source changes if requested
        if source_update_requested:
            # Parse JSON data safely
            try:
                incoming_selected_urls = json.loads(selected_urls_json) if selected_urls_json else []
                if not isinstance(incoming_selected_urls, list): incoming_selected_urls = []
            except json.JSONDecodeError: incoming_selected_urls = []

            try:
                files_to_remove_basenames = json.loads(removed_files_json) if removed_files_json else []
                if not isinstance(files_to_remove_basenames, list): files_to_remove_basenames = []
            except json.JSONDecodeError: files_to_remove_basenames = []

            # --- Process File Changes ---
            current_details = json.loads(chatbot.source_details or '{}')
            current_files_basenames = current_details.get('files_uploaded', [])
            files_to_keep = list(current_files_basenames) # Start with current list

            # 1. Handle File Deletions (if removed_files was provided)
            if files_to_remove_basenames:
                temp_files_to_keep = []
                for basename in files_to_keep:
                    if basename in files_to_remove_basenames:
                        file_path_to_delete = os.path.join(UPLOAD_FOLDER, basename)
                        if os.path.exists(file_path_to_delete):
                            try:
                                os.remove(file_path_to_delete)
                                current_app.logger.info(f"Deleted existing file during update: {basename}")
                                files_changed = True # Mark files as changed if deletions occurred
                            except Exception as file_del_e:
                                current_app.logger.error(f"Error deleting file {basename} during update: {file_del_e}")
                                # Decide if this should be a fatal error
                        else:
                             current_app.logger.warning(f"File marked for deletion not found: {basename}")
                    else:
                        temp_files_to_keep.append(basename) # Keep this file
                files_to_keep = temp_files_to_keep # Update the list

            # 2. Handle New File Uploads (Only if enabled and new_files provided)
            if chatbot.file_uploads_enabled:
                if new_files:
                    if not os.path.exists(UPLOAD_FOLDER):
                        try: os.makedirs(UPLOAD_FOLDER)
                        except OSError as e: return jsonify({"error": f"Could not create upload dir during update: {e}"}), 500

                    for file in new_files:
                        if file and file.filename != '':
                            # --- File Type and Size Validation ---
                            allowed_types_str = chatbot.allowed_file_types
                            max_size_mb = chatbot.max_file_size_mb
                            allowed_types_set = set()
                            if allowed_types_str: allowed_types_set = {mime.strip() for mime in allowed_types_str.lower().split(',') if mime.strip()}

                            file_mimetype = file.mimetype.lower() if file.mimetype else ''
                            file.seek(0, os.SEEK_END); file_size_bytes = file.tell(); file.seek(0)
                            file_size_mb = file_size_bytes / (1024 * 1024)

                            if allowed_types_set and file_mimetype not in allowed_types_set:
                                # Clean up files uploaded so far before failing
                                for path in data_cleanup_paths:
                                    if os.path.exists(path):
                                        try: os.remove(path)
                                        except Exception: pass # Log error?
                                return jsonify({"error": f"File type '{file.filename}' ({file_mimetype}) is not allowed. Allowed: {allowed_types_str}"}), 400

                            if max_size_mb is not None and file_size_mb > max_size_mb:
                                # Clean up files uploaded so far before failing
                                for path in data_cleanup_paths:
                                    if os.path.exists(path):
                                        try: os.remove(path)
                                        except Exception: pass # Log error?
                                return jsonify({"error": f"File '{file.filename}' ({file_size_mb:.2f} MB) exceeds max size ({max_size_mb} MB)."}), 400
                            # --- End Validation ---

                            filename = secure_filename(file.filename)
                            unique_basename = f"{uuid.uuid4()}_{filename}"
                            new_file_path = os.path.join(UPLOAD_FOLDER, unique_basename)
                            try:
                                file.save(new_file_path)
                                files_to_keep.append(unique_basename)
                                newly_uploaded_basenames_for_ingestion.append(unique_basename)
                                data_cleanup_paths.append(new_file_path)
                                files_changed = True # Mark files as changed if uploads occurred
                                current_app.logger.info(f"Saved new file during update: {unique_basename}")
                            except Exception as file_save_e:
                                current_app.logger.error(f"Error saving new file {filename} during update: {file_save_e}")
                                # Clean up files uploaded so far before failing
                                for path in data_cleanup_paths:
                                    if os.path.exists(path):
                                        try: os.remove(path)
                                        except Exception as cleanup_e: current_app.logger.error(f"Error cleaning up {path}: {cleanup_e}")
                                return jsonify({"error": f"Could not save file {filename}"}), 500
            elif new_files: # Files sent but uploads disabled
                 return jsonify({"error": "File uploads are disabled for this chatbot."}), 400

            # --- Process URL Changes (if selected_urls was provided) ---
            current_selected_urls = current_details.get('selected_urls', [])
            # Only calculate changes if selected_urls were actually sent in the request
            if selected_urls_json is not None:
                urls_to_add = list(set(incoming_selected_urls) - set(current_selected_urls))
                urls_to_remove = list(set(current_selected_urls) - set(incoming_selected_urls))
                if urls_to_add or urls_to_remove:
                    urls_changed = True
            else:
                # If selected_urls not in request, keep existing URLs
                incoming_selected_urls = current_selected_urls # Ensure this is set correctly for new_source_details
                urls_to_add = []
                urls_to_remove = []
        # End of 'if source_update_requested:' block for source processing

        # --- Logo File Handling ---
        logo_file = request.files.get('logo')
        if logo_file and logo_file.filename != '':
            if allowed_file(logo_file.filename, ALLOWED_LOGO_EXTENSIONS):
                logo_file.seek(0, os.SEEK_END); file_size = logo_file.tell(); logo_file.seek(0)
                if file_size <= MAX_LOGO_SIZE_MB * 1024 * 1024:
                    secure_name = secure_filename(logo_file.filename)
                    unique_filename = f"{uuid.uuid4()}_{secure_name}"
                    relative_logo_dir = os.path.join('logos', str(chatbot_id))
                    chatbot_logo_dir_full = os.path.join(current_app.root_path, 'static', relative_logo_dir)
                    new_logo_full_path = os.path.join(chatbot_logo_dir_full, unique_filename)
                    new_logo_relative_path = os.path.join(relative_logo_dir, unique_filename).replace('\\', '/')


                    try:
                        logo_file.save(new_logo_full_path)
                        logo_file_saved = True
                        if old_logo_path and old_logo_path != new_logo_relative_path:
                            delete_old_logo = True # Mark old logo for deletion after commit
                        current_app.logger.info(f"Saved new logo for chatbot {chatbot_id} to {new_logo_full_path}")
                    except Exception as save_e: return jsonify({"error": "Could not save new logo file."}), 500
                else: return jsonify({"error": f"Logo file exceeds maximum size of {MAX_LOGO_SIZE_MB} MB."}), 400
            else: return jsonify({"error": f"Logo file type not allowed. Allowed: {ALLOWED_LOGO_EXTENSIONS}"}), 400

        # --- Avatar File Handling ---
        avatar_file_saved = False # Initialize flag
        delete_old_avatar = False # Initialize flag
        old_avatar_path = chatbot.avatar_path # Store old path before potential update
        new_avatar_relative_path = None # Initialize path variable
        avatar_file = request.files.get('avatar')
        if avatar_file and avatar_file.filename != '':
            if allowed_file(avatar_file.filename, ALLOWED_AVATAR_EXTENSIONS):
                avatar_file.seek(0, os.SEEK_END); file_size = avatar_file.tell(); avatar_file.seek(0)
                if file_size <= MAX_AVATAR_SIZE_MB * 1024 * 1024:
                    filename = secure_filename(avatar_file.filename)
                    extension = filename.rsplit('.', 1)[1].lower()
                    unique_avatar_filename = f"avatar.{extension}" # Standardize name
                    relative_avatar_dir = os.path.join('avatars', str(chatbot_id))
                    chatbot_avatar_dir_full = os.path.join(current_app.root_path, 'static', relative_avatar_dir)
                    new_avatar_full_path = os.path.join(chatbot_avatar_dir_full, unique_avatar_filename)
                    new_avatar_relative_path = os.path.join(relative_avatar_dir, unique_avatar_filename).replace('\\', '/')


                    try:
                        avatar_file.save(new_avatar_full_path)
                        avatar_file_saved = True
                        if old_avatar_path and old_avatar_path != new_avatar_relative_path:
                            delete_old_avatar = True # Mark old avatar for deletion after commit
                        current_app.logger.info(f"Saved new avatar for chatbot {chatbot_id} to {new_avatar_full_path}")
                    except Exception as save_e: return jsonify({"error": "Could not save new avatar file."}), 500
                else: return jsonify({"error": f"Avatar file exceeds maximum size of {MAX_AVATAR_SIZE_MB} MB."}), 400
            else: return jsonify({"error": f"Avatar file type not allowed. Allowed: {ALLOWED_AVATAR_EXTENSIONS}"}), 400

        # --- Launcher Icon Handling ---
        launcher_icon_file = request.files.get('launcher_icon')
        launcher_icon_file_saved = False # Initialize flag
        delete_old_launcher_icon = False # Initialize flag
        old_launcher_icon_path = chatbot.launcher_icon_path # Store old path before potential update
        new_launcher_icon_relative_path = None # Initialize path variable

        if launcher_icon_file and launcher_icon_file.filename != '':
            current_app.logger.info(f"Received launcher icon file: {launcher_icon_file.filename} for chatbot {chatbot_id}")
            if allowed_file(launcher_icon_file.filename, ALLOWED_LAUNCHER_ICON_EXTENSIONS):
                # Check file size (convert MB to bytes)
                launcher_icon_file.seek(0, os.SEEK_END); file_size = launcher_icon_file.tell(); launcher_icon_file.seek(0) # Get size efficiently
                if file_size <= MAX_LAUNCHER_ICON_SIZE_MB * 1024 * 1024:
                    # Create chatbot-specific directory
                    relative_icon_dir = os.path.join('launcher_icons', str(chatbot_id))
                    chatbot_icon_dir_full = os.path.join(current_app.root_path, 'static', relative_icon_dir)


                    # Save new icon
                    filename = secure_filename(launcher_icon_file.filename)
                    extension = filename.rsplit('.', 1)[1].lower()
                    new_filename = f"icon.{extension}" # Standardize filename
                    save_path_full = os.path.join(chatbot_icon_dir_full, new_filename)
                    new_launcher_icon_relative_path = os.path.join(relative_icon_dir, new_filename).replace('\\', '/') # Use forward slashes for URLs

                    try:
                        launcher_icon_file.save(save_path_full)
                        launcher_icon_file_saved = True
                        if old_launcher_icon_path and old_launcher_icon_path != new_launcher_icon_relative_path:
                             delete_old_launcher_icon = True # Mark old icon for deletion after commit
                        current_app.logger.info(f"Saved new launcher icon to: {save_path_full}")
                    except Exception as e:
                        current_app.logger.error(f"Failed to save launcher icon {new_filename}: {e}", exc_info=True)
                        return jsonify({"error": "Could not save launcher icon file"}), 500
                else:
                    current_app.logger.warning(f"Launcher icon file rejected: too large (> {MAX_LAUNCHER_ICON_SIZE_MB}MB)")
                    return jsonify({"error": f"Launcher icon file exceeds {MAX_LAUNCHER_ICON_SIZE_MB}MB size limit"}), 400
            else:
                current_app.logger.warning(f"Launcher icon file rejected: invalid type ({launcher_icon_file.filename})")
                return jsonify({"error": "Invalid launcher icon file type. Allowed: " + ", ".join(ALLOWED_LAUNCHER_ICON_EXTENSIONS)}), 400
        # --- End Launcher Icon Handling ---

        # --- Update Chatbot Record in DB ---
        chatbot.name = name
        chatbot.launcher_text = launcher_text
        chatbot.widget_primary_color = widget_primary_color
        chatbot.widget_text_color = widget_text_color
        chatbot.widget_background_color = widget_background_color
        chatbot.user_message_color = user_message_color
        chatbot.bot_message_color = bot_message_color
        chatbot.input_background_color = input_background_color
        chatbot.widget_welcome_message = widget_welcome_message

        # Update UI Element Visibility using the new helper
        chatbot.show_widget_header = form_checkbox_to_bool('show_widget_header')
        chatbot.show_message_timestamps = form_checkbox_to_bool('show_message_timestamps')
        chatbot.show_typing_indicator = form_checkbox_to_bool('show_typing_indicator') # Add typing indicator toggle

        # Handle image analysis flag (assuming frontend sends 'true'/'false' string)
        image_analysis_enabled_str = request.form.get('image_analysis_enabled', 'false')
        chatbot.image_analysis_enabled = image_analysis_enabled_str.lower() == 'true'
        current_app.logger.info(f"Updating image_analysis_enabled to: {chatbot.image_analysis_enabled} for chatbot {chatbot_id} based on form value '{image_analysis_enabled_str}'")

        # Handle summarization settings
        summarization_enabled_str = request.form.get('summarization_enabled', 'false')
        chatbot.summarization_enabled = summarization_enabled_str.lower() == 'true'
        chatbot.allowed_scraping_domains = request.form.get('allowed_scraping_domains', chatbot.allowed_scraping_domains)
        current_app.logger.info(f"Updating summarization_enabled to: {chatbot.summarization_enabled} for chatbot {chatbot_id}")

        # Handle summarization settings
        summarization_enabled_str = request.form.get('summarization_enabled', 'false')
        chatbot.summarization_enabled = summarization_enabled_str.lower() == 'true'
        chatbot.allowed_scraping_domains = request.form.get('allowed_scraping_domains', chatbot.allowed_scraping_domains)
        current_app.logger.info(f"Updating summarization_enabled to: {chatbot.summarization_enabled} for chatbot {chatbot_id}")

        # Update logo/avatar paths if new ones were saved
        if logo_file_saved: chatbot.logo_path = new_logo_relative_path
        if avatar_file_saved: chatbot.avatar_path = new_avatar_relative_path
        if launcher_icon_file_saved: chatbot.launcher_icon_path = new_launcher_icon_relative_path

        # Update source details and type ONLY IF a source update was requested
        if source_update_requested:
            new_source_details = {
                'original_url': url_value if use_url else current_details.get('original_url'), # Keep existing if not provided
                'original_sitemap': sitemap_value if use_sitemap else current_details.get('original_sitemap'), # Keep existing if not provided
                'selected_urls': incoming_selected_urls, # Use the processed list
                'files_uploaded': files_to_keep # Use the processed list
            }
            chatbot.source_details = json.dumps(new_source_details)

            source_types = []
            if new_source_details['selected_urls']: source_types.append('Web_Filtered')
            elif new_source_details['original_url'] or new_source_details['original_sitemap']: source_types.append('Web_Direct')
            if new_source_details['files_uploaded']: source_types.append('Files')
            chatbot.source_type = '+'.join(source_types) if source_types else 'None'
        # End of conditional source details update

        # Assign retrieved non-source values to the chatbot object
        chatbot.name = name
        chatbot.launcher_text = launcher_text
        chatbot.widget_primary_color = widget_primary_color
        chatbot.widget_text_color = widget_text_color
        chatbot.widget_background_color = widget_background_color
        chatbot.user_message_color = user_message_color
        chatbot.bot_message_color = bot_message_color
        chatbot.input_background_color = input_background_color
        chatbot.widget_welcome_message = widget_welcome_message
        chatbot.default_error_message = default_error_message # Assign the new error message
        chatbot.fallback_message = fallback_message # Assign the new fallback message
        chatbot.base_prompt = request.form.get('base_prompt', chatbot.base_prompt) # Get the base prompt

        # --- Knowledge Adherence Level ---
        knowledge_level = request.form.get('knowledge_adherence_level')
        allowed_levels = ['strict', 'moderate', 'flexible']
        if knowledge_level is not None: # Only update if provided
            if knowledge_level in allowed_levels:
                chatbot.knowledge_adherence_level = knowledge_level
            else:
                current_app.logger.warning(f"Invalid knowledge_adherence_level '{knowledge_level}' provided for chatbot {chatbot_id}")
                return jsonify({"error": f"Invalid knowledge_adherence_level. Must be one of {allowed_levels}"}), 400
        knowledge_level = request.form.get('knowledge_adherence_level')
        allowed_levels = ['strict', 'moderate', 'flexible']
        if knowledge_level is not None: # Only update if provided
            if knowledge_level in allowed_levels:
                chatbot.knowledge_adherence_level = knowledge_level
            else:
                current_app.logger.warning(f"Invalid knowledge_adherence_level '{knowledge_level}' provided for chatbot {chatbot_id}")
                return jsonify({"error": f"Invalid knowledge_adherence_level. Must be one of {allowed_levels}"}), 400

        # --- Knowledge Adherence Level ---
        knowledge_level = request.form.get('knowledge_adherence_level')
        allowed_levels = ['strict', 'moderate', 'flexible']
        if knowledge_level is not None: # Only update if provided
            if knowledge_level in allowed_levels:
                chatbot.knowledge_adherence_level = knowledge_level
            else:
                current_app.logger.warning(f"Invalid knowledge_adherence_level '{knowledge_level}' provided for chatbot {chatbot_id}")
                return jsonify({"error": f"Invalid knowledge_adherence_level. Must be one of {allowed_levels}"}), 400
        
        # Assign consent settings
        chatbot.consent_message = consent_message
        chatbot.consent_required = consent_required
 
        # --- Advanced RAG Toggle ---
        chatbot.advanced_rag_enabled = form_checkbox_to_bool('advanced_rag_enabled') # Add handling for the new flag
        current_app.logger.debug(f"UPDATE Chatbot {chatbot_id}: Setting advanced_rag_enabled to: {chatbot.advanced_rag_enabled}")
 
        # Set status if sources changed (only possible if source_update_requested was true)
        if files_changed or urls_changed:
            chatbot.status = 'Update Queued'
        # Always update timestamp regardless of source changes
        chatbot.updated_at = datetime.utcnow()

        # --- Commit DB Changes ---
        db.session.commit()
        current_app.logger.info(f"Successfully updated chatbot record for ID: {chatbot_id}")

        # --- Post-Commit Actions: Delete Old Files & Trigger Re-ingestion ---

        # Delete old logo file
        if delete_old_logo:
            try:
                old_logo_full_path_to_delete = os.path.join(current_app.root_path, 'static', old_logo_path)
                if os.path.exists(old_logo_full_path_to_delete):
                    os.remove(old_logo_full_path_to_delete)
                    current_app.logger.info(f"Deleted old logo file: {old_logo_full_path_to_delete}")
                else:
                    current_app.logger.warning(f"Old logo file not found for deletion: {old_logo_full_path_to_delete}")
            except Exception as e:
                current_app.logger.error(f"Failed to delete old logo file {old_logo_full_path_to_delete}: {e}")

        # Delete old avatar file
        if delete_old_avatar:
            try:
                old_avatar_full_path_to_delete = os.path.join(current_app.root_path, 'static', old_avatar_path)
                if os.path.exists(old_avatar_full_path_to_delete):
                    os.remove(old_avatar_full_path_to_delete)
                    current_app.logger.info(f"Deleted old avatar file: {old_avatar_full_path_to_delete}")
                else:
                    current_app.logger.warning(f"Old avatar file not found for deletion: {old_avatar_full_path_to_delete}")
            except Exception as e:
                current_app.logger.error(f"Error deleting old avatar file {old_avatar_full_path_to_delete}: {e}")

        # Delete old launcher icon file
        if delete_old_launcher_icon:
            try:
                old_icon_full_path_to_delete = os.path.join(current_app.root_path, 'static', old_launcher_icon_path)
                if os.path.exists(old_icon_full_path_to_delete):
                    os.remove(old_icon_full_path_to_delete)
                    current_app.logger.info(f"Deleted old launcher icon file: {old_icon_full_path_to_delete}")
                else:
                    current_app.logger.warning(f"Old launcher icon file not found for deletion: {old_icon_full_path_to_delete}")
            except Exception as e:
                current_app.logger.error(f"Error deleting old launcher icon file {old_icon_full_path_to_delete}: {e}")
            except Exception as e:
                current_app.logger.error(f"Failed to delete old avatar file {old_avatar_full_path_to_delete}: {e}")

        # Trigger Re-ingestion if Necessary
        if files_changed or urls_changed:
            update_source_details_for_task = {
                'original_url': new_source_details['original_url'],
                'original_sitemap': new_source_details['original_sitemap'],
                'selected_urls': new_source_details['selected_urls'],
                'files_uploaded': newly_uploaded_basenames_for_ingestion, # Only NEW files
                'is_update': True,
                'removed_files_identifiers': files_to_remove_basenames,
                'removed_urls_identifiers': urls_to_remove,
            }
            current_app.logger.debug(f"UPDATE Chatbot {chatbot_id}: Queuing re-ingestion task with details: {update_source_details_for_task}")
            task = run_ingestion_task.delay(
                chatbot_id,
                client_id,
                update_source_details_for_task
            )
            current_app.logger.info(f"Queued re-ingestion task (ID: {task.id}) for updated chatbot {chatbot_id}")
            push_status_update(chatbot_id, chatbot.status, client_id) # Push 'Update Queued' status

        # --- Prepare and Return Response ---
        # Assuming chatbot has a to_dict() method or similar for serialization
        try:
            response_data = chatbot.to_dict() # You might need to implement/adjust this method
        except AttributeError:
             # Fallback if to_dict doesn't exist
             response_data = {
                 "id": chatbot.id, "name": chatbot.name, "status": chatbot.status,
                 # Add other relevant fields manually if needed
             }

        response_data["message"] = "Chatbot updated successfully" if not (files_changed or urls_changed) else "Chatbot update initiated (re-ingestion required)."
        response_data.pop('api_key', None) # Ensure API key isn't returned
        return jsonify(response_data), 200 if not (files_changed or urls_changed) else 202

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error processing update for chatbot {chatbot_id}: {e}", exc_info=True)

        # Clean up newly uploaded DATA SOURCE files
        for path in data_cleanup_paths:
            if path and os.path.exists(path):
                try: os.remove(path)
                except Exception as cleanup_e: current_app.logger.error(f"Error cleaning up data file {path}: {cleanup_e}")

        # Clean up newly uploaded LOGO file
        if logo_file_saved and new_logo_full_path and os.path.exists(new_logo_full_path):
             try:
                 os.remove(new_logo_full_path)
                 current_app.logger.warning(f"Rolled back logo file save due to error: {new_logo_full_path}")
             except Exception as cleanup_e:
                 current_app.logger.error(f"Error cleaning up new logo file {new_logo_full_path} after error: {cleanup_e}")

        # Clean up newly uploaded AVATAR file
        if avatar_file_saved and new_avatar_full_path and os.path.exists(new_avatar_full_path):
             try:
                 os.remove(new_avatar_full_path)
                 current_app.logger.warning(f"Rolled back avatar file save due to error: {new_avatar_full_path}")
             except Exception as cleanup_e:
                 current_app.logger.error(f"Error cleaning up new avatar file {new_avatar_full_path} after error: {cleanup_e}")

        return jsonify({"error": "Internal server error during chatbot update"}), 500


# --- DELETE Chatbot Endpoint ---
@bp.route('/chatbots/<int:chatbot_id>', methods=['DELETE'])
@limiter.limit("10 per hour") # Stricter limit for deletions
def delete_chatbot(chatbot_id):
    from app.tasks.deletion_tasks import delete_chatbot_data_task # Import deletion tasks locally
    """Deletes a specific chatbot and associated data."""
    # Get client_id from query parameter for verification
    client_id = request.args.get('client_id')
    if not client_id:
        current_app.logger.warning(f"delete_chatbot request missing client_id for chatbot {chatbot_id}")
        return jsonify({"error": "client_id query parameter is required"}), 400

    chatbot = db.session.get(Chatbot, chatbot_id)
    if not chatbot:
        return jsonify({"error": "Chatbot not found"}), 404

    # Ownership check:
    user = User.query.filter_by(client_id=client_id).first()
    if not user or chatbot.client_id != user.client_id:
        current_app.logger.warning(f"Unauthorized attempt to delete chatbot {chatbot_id} by client_id {client_id}")
        return jsonify({"error": "Forbidden"}), 403

    # --- Initiate asynchronous deletion via Celery ---
    try:
        # 1. Update status to indicate deletion is in progress (optional but good UX)
        #    Make sure 'Pending Deletion' is a valid status in your Chatbot model enum if you use one.
        chatbot.status = 'Pending Deletion'
        db.session.add(chatbot)
        db.session.commit()
        current_app.logger.info(f"Marked chatbot {chatbot_id} as 'Pending Deletion'.")

        # 2. Enqueue the Celery task
        task = delete_chatbot_data_task.delay(chatbot_id, user.id)
        current_app.logger.info(f"Enqueued delete_chatbot_data_task for chatbot {chatbot_id}. Task ID: {task.id}")

        # 3. Push status update to notify frontend deletion has started
        push_status_update(chatbot_id, 'Pending Deletion', client_id)

        # 4. Return 202 Accepted
        return jsonify({"message": "Chatbot deletion process initiated", "task_id": task.id}), 202

    except Exception as e:
        # Catch errors during DB update or task enqueuing
        db.session.rollback() # Rollback status change if enqueuing failed
        current_app.logger.error(f"Failed to initiate deletion for chatbot {chatbot_id}: {e}", exc_info=True)
        # Revert status if it was changed before the error
        try:
            # Re-fetch chatbot in case session state is weird after rollback
            chatbot_revert = db.session.get(Chatbot, chatbot_id)
            if chatbot_revert and chatbot_revert.status == 'Pending Deletion':
                 # Determine previous status if possible, or set to a default like 'Error' or 'Active'
                 chatbot_revert.status = 'Active' # Or original status if tracked
                 db.session.add(chatbot_revert)
                 db.session.commit()
        except Exception as revert_e:
             current_app.logger.error(f"Failed to revert status for chatbot {chatbot_id} after deletion initiation error: {revert_e}")

        return jsonify({"error": "Failed to start chatbot deletion process"}), 500

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error deleting chatbot {chatbot_id}: {e}", exc_info=True)
        return jsonify({"error": "Internal server error during chatbot deletion"}), 500



@bp.route('/chatbots/<int:chatbot_id>/sources', methods=['DELETE'])
@require_api_key
@limiter.limit("60 per minute") # Add a reasonable rate limit
def delete_chatbot_source(chatbot_id):
    from app.tasks.deletion_tasks import delete_source_data_task # MOVED IMPORT HERE
    """
    Deletes a specific data source (URL or filename) associated with a chatbot by enqueuing a Celery task.
    Expects JSON body: {"source_identifier": "url_or_filename_to_delete"}
    """
    current_app.logger.info(f"Received request to delete source for chatbot {chatbot_id}")

    # 1. Parse Request Body
    data = request.get_json()
    if not data:
        current_app.logger.error(f"Delete source failed for chatbot {chatbot_id}: Missing JSON body")
        return jsonify({"error": "Request body must be JSON"}), 400

    source_identifier = data.get('source_identifier')
    if not source_identifier or not isinstance(source_identifier, str) or not source_identifier.strip():
        current_app.logger.error(f"Delete source failed for chatbot {chatbot_id}: Missing or invalid 'source_identifier' in JSON body")
        return jsonify({"error": "Missing or invalid 'source_identifier' in request body"}), 400

    current_app.logger.info(f"Attempting to delete source '{source_identifier}' for chatbot {chatbot_id} via Celery task.")

    # 2. Enqueue Celery Task
    try:
        # delete_source_data_task is imported locally above
        task = delete_source_data_task.delay(chatbot_id, source_identifier)
        current_app.logger.info(f"Enqueued delete_source_data_task for chatbot {chatbot_id}, source: '{source_identifier}'. Task ID: {task.id}")

        # Return 202 Accepted, indicating the task has been queued
        return jsonify({
            "message": "Source deletion process initiated",
            "chatbot_id": chatbot_id,
            "source_identifier": source_identifier,
            "task_id": task.id
        }), 202

    except Exception as e:
        # Catch errors during task enqueuing
        current_app.logger.error(f"Failed to enqueue deletion task for source '{source_identifier}' on chatbot {chatbot_id}: {e}", exc_info=True)
        return jsonify({"error": "Failed to start source deletion process"}), 500


# --- Delete Specific Chatbot Source Endpoint (Potentially Redundant Now) ---
# This function might become redundant if the above delete_chatbot_source is the primary one used.
# Consider removing if confirmed that only one /sources DELETE endpoint is needed and it should be async.
@bp.route('/chatbots/<int:chatbot_id>/sources/specific', methods=['DELETE']) # Changed path to avoid conflict if kept temporarily
@require_api_key # Ensures chatbot exists and API key is valid
@limiter.limit("10 per minute") # Limit deletion attempts
def delete_specific_chatbot_source(chatbot_id):
    from app.tasks.deletion_tasks import delete_source_data_task # MOVED IMPORT HERE
    """
    Deletes a specific data source (URL or file) and its associated embeddings
    for a given chatbot. (This is the original Celery-based version, path changed to avoid conflict)
    """
    # API key validation and chatbot existence check is handled by @require_api_key
    # The validated chatbot object is available in g.chatbot if needed, but we only need the ID here.

    data = request.get_json()
    if not data or 'source_identifier' not in data:
        return jsonify({"error": "Missing 'source_identifier' in request body"}), 400

    source_identifier = data['source_identifier']
    if not isinstance(source_identifier, str) or not source_identifier.strip():
         return jsonify({"error": "'source_identifier' must be a non-empty string"}), 400

    current_app.logger.info(f"Received request to delete source (specific path) '{source_identifier}' for chatbot ID: {chatbot_id}")

    # --- Initiate asynchronous source deletion via Celery ---
    try:
        # Basic validation already done (source_identifier exists and is string)

        # Enqueue the Celery task
        # delete_source_data_task is imported locally above
        task = delete_source_data_task.delay(chatbot_id, source_identifier)
        current_app.logger.info(f"Enqueued delete_source_data_task (specific path) for chatbot {chatbot_id}, source: '{source_identifier}'. Task ID: {task.id}")

        # Return 202 Accepted
        return jsonify({
            "message": "Source deletion process initiated (specific path)",
            "chatbot_id": chatbot_id,
            "source_identifier": source_identifier,
            "task_id": task.id
        }), 202

    except Exception as e:
        # Catch errors during task enqueuing
        current_app.logger.error(f"Failed to initiate deletion (specific path) for source '{source_identifier}' on chatbot {chatbot_id}: {e}", exc_info=True)
        return jsonify({"error": "Failed to start source deletion process (specific path)"}), 500
# Removed leftover lines from previous implementation


# --- Add Data Sources to Existing Chatbot ---

@bp.route('/chatbots/<int:chatbot_id>/sources/url', methods=['POST'])
@limiter.limit(Config.DEFAULT_RATE_LIMIT) # Apply rate limiting
# @require_api_key # Optional: Add if API key auth is strictly needed for this action
def add_source_url(chatbot_id):
    """Adds a single URL as a data source to an existing chatbot."""
    chatbot = Chatbot.query.get_or_404(chatbot_id)
    # TODO: Add authorization check if needed (e.g., check client_id against chatbot owner)
    # client_id = request.headers.get('X-Client-ID') or request.args.get('client_id')
    # if not client_id or chatbot.client_id != client_id:
    #     return jsonify({"error": "Unauthorized"}), 403

    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({"error": "Missing 'url' in request body"}), 400

    new_url = data['url']
    if not new_url: # Basic validation
         return jsonify({"error": "'url' cannot be empty"}), 400

    current_app.logger.info(f"Adding URL source '{new_url}' to chatbot {chatbot_id}")

    try:
        # Update source details
        source_details = json.loads(chatbot.source_details or '{}')
        # Decide how to store multiple URLs - append to a list? Overwrite?
        # Let's store in a dedicated list 'added_urls'
        if 'added_urls' not in source_details:
            source_details['added_urls'] = []
        if new_url not in source_details['added_urls']:
             source_details['added_urls'].append(new_url)

        chatbot.source_details = json.dumps(source_details)

        # Update source type (append if not present)
        source_types = set(chatbot.source_type.split('+') if chatbot.source_type else [])
        source_types.add('URL') # Use a simple type name
        chatbot.source_type = '+'.join(filter(None, source_types)) # Filter out potential empty strings

        chatbot.status = 'Updating' # Indicate processing is needed
        db.session.commit()

        # Trigger ingestion for the new URL
        ingestion_details = {'urls_to_ingest': [new_url]} # Pass only the new URL
        task = run_ingestion_task.delay(
            chatbot.id,
            chatbot.client_id,
            ingestion_details # Pass specific details for this task
        )
        current_app.logger.info(f"Queued ingestion task (ID: {task.id}) for URL '{new_url}' on chatbot {chatbot_id}")
        push_status_update(chatbot.id, 'Updating', chatbot.client_id) # Notify frontend

        return jsonify({"message": "URL source added and ingestion started.", "chatbot_id": chatbot.id, "task_id": task.id}), 202

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error adding URL source to chatbot {chatbot_id}: {e}", exc_info=True)
        return jsonify({"error": "Failed to add URL source"}), 500


@bp.route('/chatbots/<int:chatbot_id>/sources/files', methods=['POST'])
@limiter.limit(Config.DEFAULT_RATE_LIMIT) # Apply rate limiting
# @require_api_key # Optional: Add if API key auth is strictly needed
def add_source_files(chatbot_id):
    """Adds one or more files as data sources to an existing chatbot."""
    chatbot = Chatbot.query.get_or_404(chatbot_id)
    # TODO: Add authorization check if needed

    if 'files' not in request.files:
        return jsonify({"error": "No file part named 'files' found"}), 400

    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({"error": "No files selected"}), 400

    saved_file_basenames = []
    uploaded_file_paths_to_cleanup = []
    allowed_extensions = {'txt', 'pdf', 'docx', 'md'} # Define allowed extensions here

    # Ensure upload folder exists
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads') # Get from config or default
    if not os.path.exists(upload_folder):
        try:
            os.makedirs(upload_folder)
            current_app.logger.info(f"Created upload folder: {upload_folder}")
        except OSError as e:
            current_app.logger.error(f"Failed to create upload directory '{upload_folder}': {e}")
            return jsonify({"error": f"Could not create upload dir: {e}"}), 500

    for file in files:
        if file and file.filename != '' and allowed_file(file.filename, allowed_extensions):
            filename = secure_filename(file.filename)
            unique_basename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(upload_folder, unique_basename)
            try:
                file.save(file_path)
                current_app.logger.debug(f"Saved file {unique_basename} for chatbot {chatbot_id}")
                saved_file_basenames.append(unique_basename)
                uploaded_file_paths_to_cleanup.append(file_path)
            except Exception as e:
                current_app.logger.error(f"Failed to save file {filename} for chatbot {chatbot_id}: {e}", exc_info=True)
                # Clean up files saved *in this request* before the error
                for path in uploaded_file_paths_to_cleanup:
                    if os.path.exists(path):
                        try: os.remove(path)
                        except Exception as cleanup_e: current_app.logger.error(f"Error cleaning up {path}: {cleanup_e}")
                return jsonify({"error": f"Could not save file {filename}"}), 500
        elif file and file.filename != '':
            current_app.logger.warning(f"File type not allowed for chatbot {chatbot_id}, skipping: {file.filename}")
            # Clean up any previously saved files from this request
            for path in uploaded_file_paths_to_cleanup:
                 if os.path.exists(path):
                     try: os.remove(path)
                     except Exception as cleanup_e: current_app.logger.error(f"Error cleaning up {path}: {cleanup_e}")
            return jsonify({"error": f"File type not allowed: {file.filename}. Allowed: {', '.join(allowed_extensions)}"}), 400

    if not saved_file_basenames:
         return jsonify({"error": "No valid files were processed."}), 400

    current_app.logger.info(f"Adding {len(saved_file_basenames)} file(s) as source to chatbot {chatbot_id}")

    try:
        # Update source details
        source_details = json.loads(chatbot.source_details or '{}')
        if 'added_files' not in source_details:
            source_details['added_files'] = []
        source_details['added_files'].extend(saved_file_basenames) # Add all new files
        chatbot.source_details = json.dumps(source_details)

        # Update source type
        source_types = set(chatbot.source_type.split('+') if chatbot.source_type else [])
        source_types.add('Files')
        chatbot.source_type = '+'.join(filter(None, source_types))

        chatbot.status = 'Updating'
        db.session.commit()

        # Trigger ingestion for the new files
        # Pass only the newly added files for this specific ingestion task
        ingestion_details = {'files_to_ingest': saved_file_basenames}
        task = run_ingestion_task.delay(
            chatbot.id,
            chatbot.client_id,
            ingestion_details # Pass specific details for this task
        )
        current_app.logger.info(f"Queued ingestion task (ID: {task.id}) for {len(saved_file_basenames)} file(s) on chatbot {chatbot_id}")
        push_status_update(chatbot.id, 'Updating', chatbot.client_id)

        return jsonify({"message": f"{len(saved_file_basenames)} file(s) added and ingestion started.", "chatbot_id": chatbot.id, "task_id": task.id}), 202

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error adding file sources to chatbot {chatbot_id}: {e}", exc_info=True)
        # Clean up the files saved in this request if DB update failed
        for path in uploaded_file_paths_to_cleanup:
             if os.path.exists(path):
                 try: os.remove(path)
                 except Exception as cleanup_e: current_app.logger.error(f"Error cleaning up {path} after DB error: {cleanup_e}")
        return jsonify({"error": "Failed to add file sources"}), 500


@bp.route('/chatbots/<int:chatbot_id>/crawl/start', methods=['POST'])
@limiter.limit(Config.DEFAULT_RATE_LIMIT) # Apply rate limiting
# @require_api_key # Optional
def start_crawl(chatbot_id):
    """Starts a web crawl task for a given starting URL."""
    chatbot = Chatbot.query.get_or_404(chatbot_id)
    # TODO: Add authorization check if needed

    data = request.get_json()
    if not data or 'start_url' not in data:
        return jsonify({"error": "Missing 'start_url' in request body"}), 400

    start_url = data['start_url']
    if not start_url:
        return jsonify({"error": "'start_url' cannot be empty"}), 400

    current_app.logger.info(f"Starting crawl from '{start_url}' for chatbot {chatbot_id}")

    try:
        # Use the existing discovery task? Assume it's suitable for crawling.
        # We might need a dedicated crawl task later if discovery is too simple.
        # Pass chatbot_id for context if needed by the task
        task = run_discovery_task.delay(task_id=chatbot_id, source_url=start_url, source_type='url')

        # Note: We don't modify the chatbot record here, only start the task.
        # The results will be fetched via the status endpoint.

        current_app.logger.info(f"Queued crawl task (ID: {task.id}) from '{start_url}' for chatbot {chatbot_id}")

        return jsonify({"message": "Crawl task started.", "task_id": task.id}), 202

    except Exception as e:
        current_app.logger.error(f"Error starting crawl task for chatbot {chatbot_id}: {e}", exc_info=True)
        return jsonify({"error": "Failed to start crawl task"}), 500


# --- Crawl Status and URL Addition ---

@bp.route('/chatbots/<int:chatbot_id>/crawl/status/<string:task_id>', methods=['GET'])
@limiter.limit(Config.DEFAULT_RATE_LIMIT) # Apply rate limiting
# @require_api_key # Optional
def get_crawl_status(chatbot_id, task_id):
    """Checks the status and result of a crawl task."""
    # Optional: Check if chatbot exists, though not strictly necessary just for task status
    # chatbot = Chatbot.query.get_or_404(chatbot_id)
    # TODO: Add authorization check if needed

    current_app.logger.debug(f"Checking status for crawl task {task_id} (associated with chatbot {chatbot_id})")
    try:
        task_result = AsyncResult(task_id, app=celery_app)

        response_data = {
            "task_id": task_id,
            "status": task_result.status,
            "result": None,
            "error": None
        }

        if task_result.successful():
            task_return_value = task_result.get()
            # Check if the task returned the expected dictionary structure
            if isinstance(task_return_value, dict) and 'status' in task_return_value and 'result' in task_return_value:
                # Unpack the result from the task's return value
                response_data["status"] = task_return_value.get('status', 'completed') # Use internal status
                response_data["result"] = task_return_value.get('result', []) # Get the actual URL list
                current_app.logger.info(f"Crawl task {task_id} completed successfully with internal status '{response_data['status']}'.")
            else:
                # Handle unexpected return value from successful task
                current_app.logger.warning(f"Crawl task {task_id} succeeded but returned unexpected value: {task_return_value}")
                response_data["status"] = 'completed' # Mark as completed anyway
                response_data["result"] = [] # Return empty list as fallback
                response_data["error"] = "Task succeeded but returned unexpected data structure."
        elif task_result.failed():
            # Capture traceback or error message if possible
            error_info = str(task_result.info) if task_result.info else "Task failed without specific error info."
            response_data["error"] = error_info
            current_app.logger.error(f"Crawl task {task_id} failed: {error_info}")
        else:
            # Status is PENDING, STARTED, RETRY, etc.
             current_app.logger.debug(f"Crawl task {task_id} status: {task_result.status}")


        return jsonify(response_data), 200

    except Exception as e:
        # Handle potential issues with Celery connection or task ID format
        current_app.logger.error(f"Error checking status for task {task_id}: {e}", exc_info=True)
        return jsonify({"error": "Failed to retrieve task status"}), 500


@bp.route('/chatbots/<int:chatbot_id>/crawl/add-urls', methods=['POST'])
@limiter.limit(Config.DEFAULT_RATE_LIMIT) # Apply rate limiting
# @require_api_key # Optional
def add_crawled_urls(chatbot_id):
    """Adds a list of URLs (typically from crawl results) to a chatbot for ingestion."""
    chatbot = Chatbot.query.get_or_404(chatbot_id)
    # TODO: Add authorization check if needed

    data = request.get_json()
    if not data or 'selected_urls' not in data:
        return jsonify({"error": "Missing 'selected_urls' list in request body"}), 400

    selected_urls = data['selected_urls']
    if not isinstance(selected_urls, list):
         return jsonify({"error": "'selected_urls' must be a list"}), 400
    if not selected_urls:
         return jsonify({"message": "No URLs provided to add."}), 200 # Or 400 if empty list is invalid

    # Basic validation (optional: add more robust URL validation)
    valid_urls = [url for url in selected_urls if isinstance(url, str) and url.strip()]
    if not valid_urls:
         return jsonify({"error": "No valid URLs found in the provided list."}), 400

    current_app.logger.info(f"Adding {len(valid_urls)} crawled URLs to chatbot {chatbot_id}")

    try:
        # Update source details
        source_details = json.loads(chatbot.source_details or '{}')
        # Store in a dedicated list for crawled URLs
        if 'crawled_urls_added' not in source_details:
            source_details['crawled_urls_added'] = []
        # Avoid duplicates within this specific list
        existing_crawled = set(source_details['crawled_urls_added'])
        newly_added_count = 0
        for url in valid_urls:
            if url not in existing_crawled:
                source_details['crawled_urls_added'].append(url)
                existing_crawled.add(url)
                newly_added_count += 1

        if newly_added_count == 0:
             return jsonify({"message": "All provided URLs were already added."}), 200

        chatbot.source_details = json.dumps(source_details)

        # Update source type
        source_types = set(chatbot.source_type.split('+') if chatbot.source_type else [])
        source_types.add('CrawledURL') # Specific type for crawled URLs
        chatbot.source_type = '+'.join(filter(None, source_types))

        chatbot.status = 'Updating'
        db.session.commit()

        # Trigger ingestion for the newly added valid URLs
        ingestion_details = {'urls_to_ingest': valid_urls} # Pass the list of URLs to ingest
        task = run_ingestion_task.delay(
            chatbot.id,
            chatbot.client_id,
            ingestion_details
        )
        current_app.logger.info(f"Queued ingestion task (ID: {task.id}) for {len(valid_urls)} crawled URLs on chatbot {chatbot_id}")
        push_status_update(chatbot.id, 'Updating', chatbot.client_id)

        return jsonify({"message": f"{len(valid_urls)} crawled URLs added and ingestion started.", "chatbot_id": chatbot.id, "task_id": task.id}), 202

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error adding crawled URLs to chatbot {chatbot_id}: {e}", exc_info=True)
        return jsonify({"error": "Failed to add crawled URLs"}), 500


# --- Add GET status and POST add-urls endpoints next ---


# --- Add GET status and POST add-urls endpoints next ---


# --- Regenerate API Key Endpoint ---
@bp.route('/chatbots/<int:chatbot_id>/regenerate-key', methods=['POST'])
@limiter.limit("5 per hour") # Limit regeneration frequency
def regenerate_api_key(chatbot_id):
    """Regenerates the widget API key for a specific chatbot."""
    data = request.get_json()
    if not data or 'client_id' not in data:
        current_app.logger.warning(f"Regenerate key attempt missing client_id for chatbot {chatbot_id}")
        return jsonify({"error": "client_id is required in request body"}), 400

    request_client_id = data.get('client_id')

    # Retrieve the chatbot or return 404 if not found
    chatbot = Chatbot.query.get_or_404(chatbot_id)

    # --- Ownership Check ---
    # Verify that the client_id from the request matches the chatbot's owner
    if chatbot.client_id != request_client_id:
        current_app.logger.warning(f"Unauthorized attempt to regenerate key for chatbot {chatbot_id} by client {request_client_id} (Owner: {chatbot.client_id})")
        # Return 403 Forbidden if the client doesn't own the chatbot
        return jsonify({"error": "Unauthorized to modify this chatbot"}), 403
    # ---------------------

    current_app.logger.info(f"Regenerating API key for chatbot {chatbot_id} requested by client {request_client_id}")

    try:
        # 1. Generate a new secure plaintext API key
        new_plaintext_key = secrets.token_urlsafe(32)
        # 2. Hash the new plaintext key
        new_hashed_key = generate_password_hash(new_plaintext_key)

        # 3. Update the api_key field of the Chatbot instance with the new hashed key
        chatbot.api_key = new_hashed_key
        # 4. Commit the change to the database
        db.session.commit()

        current_app.logger.info(f"Successfully regenerated and stored new hashed API key for chatbot {chatbot_id}")

        # 5. Return a success response containing the new plaintext API key
        return jsonify({'new_api_key': new_plaintext_key}), 200

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error regenerating API key for chatbot {chatbot_id}: {e}", exc_info=True)
        return jsonify({"error": "Internal server error during key regeneration"}), 500


# --- Query Chatbot Endpoint ---
@bp.route('/chatbots/<int:chatbot_id>/query', methods=['POST'])
@limiter.limit("100 per minute") # Limit query rate per chatbot
@require_api_key # Use the decorator for API key authentication
def query_chatbot(chatbot_id):
    """Handles user queries against a specific chatbot using RAG."""
    overall_start_time = time.time()
    current_app.logger.info(f"QueryChatbot: START for chatbot {chatbot_id}")
    from app.models import ChatMessage # Local import
    chatbot = g.chatbot # Get the validated chatbot object from the decorator

    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    user_query = data['query']
    session_id = data.get('session_id') # Get session_id if provided

    # Generate a session ID if not provided by the client
    if not session_id:
        session_id = str(uuid.uuid4())
        current_app.logger.info(f"New chat session started for chatbot {chatbot_id}: {session_id}")

    # --- Language Detection Disabled ---
    # Language detection and translation logic removed as requested.
    # Passing original user_query directly to RAG.
    # ---------------------------------

    # --- Retrieve Chat History (if enabled and session_id provided) ---
    history_start_time = time.time()
    chat_history = []
    if chatbot.save_history_enabled and session_id:
        try:
            # Fetch recent messages for this session, ordered by timestamp
            recent_messages = ChatMessage.query.filter_by(
                chatbot_id=chatbot_id,
                session_id=session_id
            ).order_by(ChatMessage.timestamp.asc()).limit(10).all() # Limit history length

            chat_history = [{"role": msg.role, "content": msg.content} for msg in recent_messages]
            current_app.logger.debug(f"Retrieved {len(chat_history)} messages for session {session_id}")

        except Exception as history_err:
            current_app.logger.error(f"Error retrieving chat history for session {session_id}: {history_err}", exc_info=True)
            # Proceed without history, but log the error
    current_app.logger.info(f"PERF: Chat History Retrieval for chatbot {chatbot_id} took {time.time() - history_start_time:.4f} seconds.")

    # --- Execute RAG Pipeline ---
    try:
        rag_service = get_rag_service()
        rag_call_start_time = time.time()

        # Pass chatbot_id, user_query, and chat_history to the RAG service
        # Pass the potentially translated english_query to the RAG service
        response_data = rag_service.execute_pipeline(
            chatbot_id=str(chatbot_id), # Ensure chatbot_id is string if needed by RAG service
            query=user_query, # Use the original user query (language detection disabled)
            chat_history=chat_history, # Pass the retrieved history
            client_id=chatbot.client_id # Pass the client_id from the chatbot object
        )
        current_app.logger.info(f"PERF: RAG Pipeline execution for chatbot {chatbot_id} took {time.time() - rag_call_start_time:.4f} seconds.")
        # end_time = time.time() # Original timing, replaced by PERF log above
        # current_app.logger.info(f"Query for chatbot {chatbot_id} processed in {end_time - start_time:.2f} seconds.") # Original timing log

        # --- Process RAG service response (Dictionary) ---
        final_response_text = ""
        sources = []
        warnings = None
        error_message = None
        http_status_code = 500 # Default to error

        if isinstance(response_data, dict):
            final_response_text = response_data.get("answer", "")
            sources = response_data.get("sources", [])
            warnings = response_data.get("warnings") # Could be None or a string
            error_message = response_data.get("error")

            if error_message:
                current_app.logger.error(f"RAG service returned an error for chatbot {chatbot_id}: {error_message}")
                # Use the specific error message if available, otherwise a generic one
                final_response_text = f"Error: {error_message}"
                # Determine HTTP status code based on RAG error or default to 500
                http_status_code = response_data.get("status_code", 500)
            elif final_response_text:
                http_status_code = 200 # Success if we have an answer and no explicit error
                if warnings:
                     current_app.logger.warning(f"RAG service returned warnings for chatbot {chatbot_id}: {warnings}")
            else:
                 # No answer and no specific error message from RAG
                 current_app.logger.error(f"RAG service returned no answer and no error message for chatbot {chatbot_id}. Response: {response_data}")
                 final_response_text = "Error: Assistant did not provide a response."
                 http_status_code = 500
        else:
            # Handle unexpected response type from RAG service
            current_app.logger.error(f"RAG service returned unexpected data type for chatbot {chatbot_id}: {type(response_data)}")
            final_response_text = "Error: Received invalid response structure from assistant."
            http_status_code = 500

        # --- Save Interaction to History (if enabled) ---
        assistant_message_id = None # Initialize before try block
        if chatbot.save_history_enabled and session_id:
            try:
                # Save user message regardless of RAG success/failure
                user_message = ChatMessage(
                    chatbot_id=chatbot_id,
                    session_id=session_id,
                    role='user',
                    content=user_query # Save original user query
                )
                db.session.add(user_message)

                # Save assistant response (which might be an error message)
                assistant_message = ChatMessage(
                    chatbot_id=chatbot_id,
                    session_id=session_id,
                    role='assistant',
                    content=final_response_text # Save the final text sent to user
                )
                db.session.add(assistant_message)
                db.session.commit()
                assistant_message_id = assistant_message.id # Get ID after successful commit
                current_app.logger.debug(f"Saved user and assistant messages for session {session_id}")
            except Exception as save_hist_err:
                 db.session.rollback()
                 current_app.logger.error(f"Error saving chat history for session {session_id}: {save_hist_err}", exc_info=True)
                 # assistant_message_id remains None

        # --- Return Response ---
        response_payload = {
            # Use "answer" key for consistency, even if it contains an error message
            "answer": final_response_text,
            "sources": sources,
            "session_id": session_id,
            "message_id": assistant_message_id # Include message ID if saved
        }
        # Optionally include warnings if they exist
        if warnings:
            response_payload["warnings"] = warnings

        # Use the determined http_status_code
        return jsonify(response_payload), http_status_code

    except Exception as e: # This except corresponds to the try block starting at line 2061
        current_app.logger.error(f"Unexpected error processing query for chatbot {chatbot_id}: {e}", exc_info=True)
        # Ensure a consistent error structure is returned even for unexpected errors
        current_app.logger.info(f"PERF: QueryChatbot for chatbot {chatbot_id} overall took {time.time() - overall_start_time:.4f} seconds (Exception).")
        return jsonify({"error": "An unexpected error occurred processing your request.", "session_id": session_id, "sources": []}), 500


# --- Delete Chat Session History Endpoint ---
@bp.route('/chat-sessions/<string:session_id>/history', methods=['DELETE'])
@limiter.limit("20 per hour")
@require_api_key # Requires API key associated with the chatbot the session belongs to
def delete_session_history(session_id):
    """Deletes all messages associated with a specific chat session."""
    from app.models import ChatMessage, DetailedFeedback # Local import
    chatbot = g.chatbot # Get chatbot from decorator context

    # Check if user clearing is allowed for this chatbot
    if not chatbot.allow_user_history_clearing:
        return jsonify({"error": "User history clearing is disabled for this chatbot"}), 403

    try:
        # Delete messages matching chatbot_id and session_id
        num_deleted = ChatMessage.query.filter_by(
            chatbot_id=chatbot.id,
            session_id=session_id
        ).delete()

        # Also delete associated detailed feedback for those messages
        # This requires joining or a subquery if not using cascade delete
        feedback_to_delete = DetailedFeedback.query.filter(
             DetailedFeedback.session_id == session_id,
             # Ensure feedback belongs to messages of the authorized chatbot
             DetailedFeedback.message.has(chatbot_id=chatbot.id)
        ).delete(synchronize_session=False) # Use False if relationships might interfere

        db.session.commit()

        if num_deleted > 0 or feedback_to_delete > 0:
            current_app.logger.info(f"Deleted {num_deleted} messages and {feedback_to_delete} feedback entries for session {session_id} (Chatbot {chatbot.id})")
            return jsonify({"message": f"Chat history for session {session_id} deleted successfully."}), 200
        else:
            current_app.logger.info(f"No chat history found to delete for session {session_id} (Chatbot {chatbot.id})")
            return jsonify({"message": "No chat history found for this session."}), 200 # Or 404? 200 is okay as the state is "no history"

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error deleting chat history for session {session_id}: {e}", exc_info=True)
        return jsonify({"error": "Internal server error during history deletion"}), 500

# --- Query Chatbot with Image Endpoint ---
@bp.route('/chatbots/<int:chatbot_id>/query_with_image', methods=['POST'])
@limiter.limit("60 per minute") # Apply rate limiting
@require_api_key # Requires valid API key for the chatbot
def query_chatbot_with_image(chatbot_id):
    """
    Handles chatbot queries that include an image for analysis.
    Accepts multipart/form-data.
    Implements validation and structured error handling as per plan.
    """
    chatbot = g.chatbot # Get chatbot from decorator context
    client_id = chatbot.client_id

    # --- 1. Check if feature is enabled ---
    if not chatbot.image_analysis_enabled:
        current_app.logger.warning(f"Image query attempt for chatbot {chatbot_id} where feature is disabled.")
        return jsonify({
            "error": {
                "code": "IMAGE_ANALYSIS_DISABLED",
                "message": "Image analysis is not enabled for this chatbot."
            }
        }), 403 # Forbidden

    # --- 2. Get data from multipart/form-data ---
    original_query = request.form.get('query', '') # Text query (optional) - Renamed to original_query
    history_str = request.form.get('history', '[]') # Chat history (JSON string)
    session_id = request.form.get('session_id') # Optional session ID
    query_language = request.form.get('language', 'en') # Optional language from client, default 'en'
    image_file = request.files.get('image') # Image file

    # --- Language Detection and Translation Removed ---
    # The original_query will be used directly.
    # No translation to English is performed here anymore.
    # Leftover exception/else block from removed language processing removed.
    # -------------------------------------------------------------


    # --- 3. Validate Image (Plan Section 4.1.1) ---
    if not image_file:
        return jsonify({"error": {"code": "VALIDATION_MISSING_IMAGE", "message": "No image file provided."}}), 400

    # 3a. File Size Check
    # Need to read the file to check size reliably if not using content_length
    try:
        image_file.seek(0, os.SEEK_END) # Go to end of file
        file_size = image_file.tell() # Get size
        image_file.seek(0) # Rewind to beginning for subsequent reads
        if file_size > Config.MAX_IMAGE_SIZE_BYTES:
            current_app.logger.warning(f"Image upload failed for chatbot {chatbot_id}: Size {file_size} exceeds limit {Config.MAX_IMAGE_SIZE_BYTES}")
            return jsonify({"error": {"code": "VALIDATION_IMAGE_SIZE_EXCEEDED", "message": f"Image file size exceeds the limit ({Config.MAX_IMAGE_SIZE_MB:.1f}MB)."}}), 400
    except Exception as e:
        current_app.logger.error(f"Error checking image file size for chatbot {chatbot_id}: {e}", exc_info=True)
        return jsonify({"error": {"code": "INTERNAL_SERVER_ERROR", "message": "Failed to read image file size."}}), 500

    # 3b. MIME Type Check
    # Use file stream's mimetype if available, otherwise guess from filename
    mime_type = image_file.mimetype
    if not mime_type:
        mime_type, _ = mimetypes.guess_type(secure_filename(image_file.filename))

    if not mime_type or mime_type.lower() not in Config.ALLOWED_IMAGE_MIME_TYPES:
        current_app.logger.warning(f"Image upload failed for chatbot {chatbot_id}: Unsupported MIME type '{mime_type}'")
        return jsonify({"error": {"code": "VALIDATION_UNSUPPORTED_MIME_TYPE", "message": f"Unsupported image format ({mime_type}). Please use JPEG, PNG, WEBP, or GIF."}}), 400

    # Read image data for content validation and passing to service
    try:
        image_data = image_file.read()
        if not image_data: # Check if file is empty after reading
            raise ValueError("Image file is empty.")
    except Exception as e:
        current_app.logger.error(f"Failed to read image file data for chatbot {chatbot_id}: {e}", exc_info=True)
        return jsonify({"error": {"code": "INTERNAL_SERVER_ERROR", "message": "Failed to process image file data."}}), 500

    # 3c. Content Validation (Pillow)
    try:
        with PILImage.open(BytesIO(image_data)) as img:
            img.verify() # Verify image integrity (checks for corruption)
        # Optionally, re-open to check format if verify() isn't enough
        # with PILImage.open(BytesIO(image_data)) as img:
        #     format_check = img.format
        current_app.logger.debug(f"Pillow validation successful for image from chatbot {chatbot_id}")
    except UnidentifiedImageError:
        current_app.logger.warning(f"Image upload failed for chatbot {chatbot_id}: Pillow could not identify image format.")
        return jsonify({"error": {"code": "VALIDATION_INVALID_IMAGE", "message": "The uploaded file could not be identified as a valid image."}}), 400
    except Exception as e: # Catch other potential Pillow errors
        current_app.logger.error(f"Pillow validation failed for chatbot {chatbot_id}: {e}", exc_info=True)
        return jsonify({"error": {"code": "VALIDATION_INVALID_IMAGE", "message": "The uploaded file could not be processed as a valid image."}}), 400

    # --- 4. Parse History ---
    try:
        chat_history = json.loads(history_str)
        if not isinstance(chat_history, list):
            raise ValueError("History must be a list.")
        # Optional: Validate structure of each history item {role: str, content: str}
        for item in chat_history:
            if not isinstance(item, dict) or 'role' not in item or 'content' not in item:
                raise ValueError("Invalid history item structure.")
    except (json.JSONDecodeError, ValueError) as e:
        current_app.logger.warning(f"Invalid chat history format received for chatbot {chatbot_id}: {e}. History: '{history_str[:100]}...'")
        chat_history = [] # Default to empty history on error

    # --- 5. Prepare for RAG Service ---
    # Ensure session_id exists if needed for message saving
    if not session_id:
        session_id = str(uuid.uuid4()) # Generate one if missing
        current_app.logger.info(f"Generated new session_id for image query: {session_id}")

    # --- 6. Call RAG Service ---
    try:
        rag_service = get_rag_service()
        if rag_service is None: # Check if initialization failed
            raise RuntimeError("RAG Service is not available.")

        current_app.logger.info(f"Executing RAG pipeline with image for chatbot {chatbot_id}, client {client_id}, session {session_id}")

        # *** IMPORTANT: Assumes execute_pipeline is updated to accept image_data and mime_type ***
        # *** The actual implementation of passing image_data needs to happen in rag_service.py ***
        # Pass the original query and language directly to the RAG service
        response_data = rag_service.execute_pipeline(
            query=original_query, # Pass the original user query text
            chat_history=chat_history,
            query_language=query_language, # Pass the language provided by the client (or default 'en')
            # target_language=target_language, # Assuming RAG service determines target language internally
            client_id=client_id,
            chatbot_id=chatbot_id,
            image_data=image_data, # Pass the validated image bytes
            image_mime_type=mime_type # Pass the validated/guessed mime type
        )

        # Check if the service returned an error structure
        if isinstance(response_data, dict) and 'error' in response_data and response_data['error'] is not None: # Check if error value is actually set
            current_app.logger.error(f"RAG service returned error for chatbot {chatbot_id}: {response_data['error']}")
            # Use the error structure from the service directly
            # Determine appropriate status code based on error code if possible
            status_code = 500 # Default to internal server error
            rag_error = response_data['error'] # We know 'error' key exists, but value could be None

            # Safely check if rag_error is a dict and then check the code
            if isinstance(rag_error, dict):
                error_code = rag_error.get('code', '')
                # Ensure error_code is a string before calling startswith
                if isinstance(error_code, str) and error_code.startswith('GEMINI_'):
                    status_code = 502 # Bad Gateway for upstream API issues
            
            # Return the original error structure from the service
            return jsonify(response_data), status_code

        # Check for successful response format (dictionary with answer and ID)
        if isinstance(response_data, dict) and 'answer' in response_data and 'response_message_id' in response_data:
            english_response_text = response_data.get('answer', "Error: Missing answer in response.") # Use .get for safety
            response_message_id = response_data.get('response_message_id')
            # Log potential warnings if present
            if response_data.get('warnings'):
                 current_app.logger.warning(f"RAG service returned warnings for chatbot {chatbot_id}, session {session_id}: {response_data['warnings']}")
        # Check for older successful response format (list/tuple) - Keep for backward compatibility?
        elif isinstance(response_data, (list, tuple)) and len(response_data) == 2:
            english_response_text, response_message_id = response_data
            current_app.logger.warning(f"Received older list/tuple response format from RAG for chatbot {chatbot_id}, session {session_id}.") # Log this case
        # Check for AI refusal (string or dict with only 'response')
        elif isinstance(response_data, str):
            # Assume string is the refusal message
            english_response_text = response_data
            response_message_id = None
            current_app.logger.info(f"AI refused image for chatbot {chatbot_id}, session {session_id}. Reason: {english_response_text}") # Log refusal as info
        elif isinstance(response_data, dict) and 'response' in response_data and 'message_id' not in response_data and 'error' not in response_data:
            # Assume dict with only 'response' is the refusal message
            english_response_text = response_data['response']
            response_message_id = None
            current_app.logger.info(f"AI refused image for chatbot {chatbot_id}, session {session_id}. Reason: {english_response_text}") # Log refusal as info
        else:
            # Handle genuinely unexpected response format
            current_app.logger.error(f"Unexpected response format from RAG service for chatbot {chatbot_id}, session {session_id}. Response: {response_data}")
            english_response_text = "Error processing request." # Generic error message
            response_message_id = None # No valid message ID

        # Log only if we got a valid text response (avoid logging huge errors again)
        if response_message_id is not None or isinstance(response_data, str) or (isinstance(response_data, dict) and 'response' in response_data):
             current_app.logger.info(f"RAG pipeline with image completed for chatbot {chatbot_id}, session {session_id}. Response ID: {response_message_id}, Response length: {len(english_response_text)}")
        # Else case (unexpected format) was already logged above

        # --- Response translation removed. RAG service provides the final response text ---
        # The variable 'english_response_text' now holds the final response intended for the user.
        final_response = english_response_text # Use the direct response

        # --- 7. Save messages (optional but recommended) ---
        # Only save messages if the RAG pipeline returned a valid response ID
        if response_message_id is not None:
            try:
                # Construct user message content including image indicator
                user_message_content = f"{original_query}\n[Image: {secure_filename(image_file.filename)}]" if original_query else f"[Image: {secure_filename(image_file.filename)}]"

                user_message = ChatMessage(
                    chatbot_id=chatbot_id,
                    session_id=session_id,
                    role='user',
                    content=user_message_content # Store original query text + marker
                    # language=original_language # Removed: Field doesn't exist
                    # Removed client_id
                )
                db.session.add(user_message)

                # Let the DB assign the ID for the assistant message
                assistant_message = ChatMessage(
                    chatbot_id=chatbot_id,
                    session_id=session_id,
                    role='assistant',
                    content=final_response # Store final response from RAG
                    # language=original_language # Removed: Field doesn't exist
                    # Removed client_id and id=response_message_id
                )
                db.session.add(assistant_message)
                db.session.commit()
                current_app.logger.info(f"Saved user and assistant messages with image context for session {session_id}")
            except Exception as db_err: # Keep the original exception handling indent
                db.session.rollback()
                current_app.logger.error(f"Database error saving chat messages for session {session_id}: {db_err}")
                # Decide if you want to return an error here or just log it
        else:
            current_app.logger.warning(f"Skipping message save for session {session_id} as RAG pipeline did not return a valid response_message_id.")

        # --- 8. Return Success Response ---
        # Return the direct response from RAG service
        return jsonify({"response": final_response, "message_id": response_message_id}), 200

    except Exception as e:
        current_app.logger.error(f"Unhandled error during image query processing for chatbot {chatbot_id}: {e}", exc_info=True)
        # Check for specific exceptions from RagService if needed to return specific codes
        # For now, return a generic internal error
        return jsonify({
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred while processing your request."
            }
        }), 500

# --- Summarization Endpoint ---
@bp.route('/chatbots/<int:chatbot_id>/summarize', methods=['POST'])
@require_api_key # Ensures chatbot exists and API key is valid, sets g.chatbot
@limiter.limit("60 per minute") # Apply rate limiting
def summarize_content(chatbot_id):
    """
    Summarizes content from a given URL or text using the chatbot's configuration.
    Accepts 'url' or 'text' and an optional 'language'.
    Requires a valid API key for the chatbot.
    ---
    parameters:
      - name: chatbot_id
        in: path
        required: true
        type: integer
        description: The ID of the chatbot.
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            url:
              type: string
              description: The URL of the content to summarize (provide either url or text).
            text:
              type: string
              description: The raw text content to summarize (provide either url or text).
            language:
              type: string
              description: Optional target language code for the summary (e.g., 'en', 'es'). Defaults to 'en'.
          example:
            url: "https://example.com/article"
            language: "en"
          example2:
            text: "This is a long piece of text that needs summarizing..."
            language: "es"
    responses:
      200:
        description: Summarization successful.
      400:
        description: Invalid input (e.g., missing or conflicting source, invalid URL, missing language if required by model).
      403:
        description: Invalid or missing API key, or summarization not enabled for this chatbot.
      500:
        description: Internal server error during summarization.
      502:
        description: Bad Gateway (e.g., error fetching URL content).
      504:
        description: Gateway Timeout (e.g., timeout fetching URL content).
    """
    overall_start_time = time.time()
    current_app.logger.info(f"SummarizeContent: START for chatbot {chatbot_id}")
    # Check if the feature is enabled for this chatbot (g.chatbot is set by require_api_key)
    if not g.chatbot.summarization_enabled:
        current_app.logger.warning(f"Summarization attempt denied for disabled chatbot {chatbot_id}.")
        current_app.logger.info(f"PERF: SummarizeContent for chatbot {chatbot_id} took {time.time() - overall_start_time:.4f} seconds (Summarization Disabled).")
        return jsonify({"error": "Summarization feature is not enabled for this chatbot."}), 403

    data = request.get_json()
    if not data:
        current_app.logger.info(f"PERF: SummarizeContent for chatbot {chatbot_id} took {time.time() - overall_start_time:.4f} seconds (No JSON Body).")
        return jsonify({"error": "Request body must be JSON."}), 400

    # Get content details from the request payload
    content_type = data.get('content_type')
    content = data.get('content')
    target_language = data.get('target_language', 'en') # Match frontend key, default 'en'

    # --- Input Validation ---
    if content_type not in ['url', 'text']:
        current_app.logger.info(f"PERF: SummarizeContent for chatbot {chatbot_id} took {time.time() - overall_start_time:.4f} seconds (Invalid Content Type).")
        return jsonify({"error": "Invalid 'content_type'. Must be 'url' or 'text'."}), 400

    if not content or (isinstance(content, str) and not content.strip()):
         current_app.logger.info(f"PERF: SummarizeContent for chatbot {chatbot_id} took {time.time() - overall_start_time:.4f} seconds (Missing Content).")
         return jsonify({"error": "Missing or empty 'content' field."}), 400

    # Additional validation specific to type
    if content_type == 'url':
        # Basic URL format check
        if not isinstance(content, str) or not content.startswith(('http://', 'https://')):
             current_app.logger.info(f"PERF: SummarizeContent for chatbot {chatbot_id} took {time.time() - overall_start_time:.4f} seconds (Invalid URL).")
             return jsonify({"error": "Invalid URL format provided for 'content'."}), 400
    # No specific validation needed for 'text' content here, beyond not being empty.

    try:
        service_call_start_time = time.time()
        service = SummarizationService(current_app.logger)
        result = service.summarize(
            chatbot_id=chatbot_id,
            content_type=content_type,
            content=content,
            target_language=target_language
        )
        current_app.logger.info(f"PERF: SummarizationService.summarize call for chatbot {chatbot_id} took {time.time() - service_call_start_time:.4f} seconds.")
        current_app.logger.info(f"PERF: SummarizeContent for chatbot {chatbot_id} overall took {time.time() - overall_start_time:.4f} seconds (Success).")
        return jsonify(result), 200

    except ValueError as e: 
        current_app.logger.warning(f"Summarization ValueError for chatbot {chatbot_id}: {e}")
        current_app.logger.info(f"PERF: SummarizeContent for chatbot {chatbot_id} overall took {time.time() - overall_start_time:.4f} seconds (ValueError).")
        return jsonify({"error": str(e)}), 400
    except PermissionError as e: 
        current_app.logger.warning(f"Summarization PermissionError for chatbot {chatbot_id}: {e}")
        current_app.logger.info(f"PERF: SummarizeContent for chatbot {chatbot_id} overall took {time.time() - overall_start_time:.4f} seconds (PermissionError).")
        return jsonify({"error": str(e)}), 403
    except ConnectionError as e: 
         current_app.logger.error(f"Summarization ConnectionError for chatbot {chatbot_id}, content: {content[:100]}...: {e}")
         current_app.logger.info(f"PERF: SummarizeContent for chatbot {chatbot_id} overall took {time.time() - overall_start_time:.4f} seconds (ConnectionError).")
         return jsonify({"error": f"Failed to retrieve content: {e}"}), 502
    except TimeoutError as e: 
         current_app.logger.error(f"Summarization TimeoutError for chatbot {chatbot_id}, content: {content[:100]}...: {e}")
         current_app.logger.info(f"PERF: SummarizeContent for chatbot {chatbot_id} overall took {time.time() - overall_start_time:.4f} seconds (TimeoutError).")
         return jsonify({"error": f"Timeout retrieving content: {e}"}), 504
    except RuntimeError as e: 
        current_app.logger.error(f"Summarization RuntimeError for chatbot {chatbot_id}, content: {content[:100]}...: {e}", exc_info=True)
        current_app.logger.info(f"PERF: SummarizeContent for chatbot {chatbot_id} overall took {time.time() - overall_start_time:.4f} seconds (RuntimeError).")
        return jsonify({"error": f"An internal error occurred during summarization: {e}"}), 500
    except Exception as e: 
        current_app.logger.error(f"Unexpected error during summarization for chatbot {chatbot_id}, content: {content[:100]}...: {e}", exc_info=True)
        current_app.logger.info(f"PERF: SummarizeContent for chatbot {chatbot_id} overall took {time.time() - overall_start_time:.4f} seconds (Exception).")
        return jsonify({"error": "An unexpected internal server error occurred."}), 500

# --- End Summarization Endpoint ---

# --- Logo Settings Endpoints ---

@bp.route('/settings/logo', methods=['GET'])
# @limiter.limit("60 per minute") # Optional: Add rate limiting
# TODO: Add authentication/authorization if needed (e.g., require logged-in user via client_id)
def get_logo_settings():
    """Gets the current company logo filename."""
    client_id = request.args.get('client_id') # Or get from session/token
    if not client_id:
        # For now, we proceed without client_id, but real implementation should require it.
        # return jsonify({"error": "Client ID is required"}), 400
        pass # Remove this pass and uncomment above line in real implementation

    # --- Retrieve current logo filename ---
    # This part needs actual implementation based on how you store settings
    current_logo_filename = get_current_logo_filename(client_id) # Placeholder call

    logo_url = None
    if current_logo_filename:
         # Use url_for to generate the URL dynamically
         try:
             # Ensure the static endpoint exists and is configured correctly in Flask app setup
             logo_url = url_for('static', filename=f'logos/{current_logo_filename}', _external=True)
         except RuntimeError as e:
             current_app.logger.error(f"Could not generate URL for logo '{current_logo_filename}': {e}. Make sure the static route is configured.")
             logo_url = None # Fallback or indicate error

    current_app.logger.info(f"Returning logo info: filename='{current_logo_filename}', url='{logo_url}'")
    return jsonify({
        "logo_filename": current_logo_filename,
        "logo_url": logo_url # Return the full URL
        }), 200


@bp.route('/settings/logo', methods=['POST'])
# @limiter.limit("10 per hour") # Optional: Add rate limiting
# TODO: Add authentication/authorization (e.g., require logged-in user via client_id)
def upload_logo():
    """Uploads a new company logo."""
    client_id = request.form.get('client_id') # Get client_id from form data
    if not client_id:
        current_app.logger.warning("Logo upload attempt missing client_id.")
        return jsonify({"error": "Client ID is required"}), 400

    # --- Validate User/Client ---
    # TODO: Add check to ensure client_id corresponds to a valid, authenticated user
    # user = User.query.filter_by(client_id=client_id).first()
    # if not user:
    #     current_app.logger.warning(f"Logo upload attempt for invalid client_id: {client_id}")
    #     return jsonify({"error": "Invalid client ID"}), 403

    current_app.logger.info(f"Logo upload attempt received for client_id: {client_id}")

    if 'logo' not in request.files:
        current_app.logger.warning(f"Logo upload failed for client {client_id}: 'logo' file part missing.")
        return jsonify({"error": "No logo file part in the request"}), 400

    file = request.files['logo']

    if file.filename == '':
        current_app.logger.warning(f"Logo upload failed for client {client_id}: No file selected.")
        return jsonify({"error": "No selected file"}), 400

    # --- File Validation ---
    if file and allowed_file(file.filename, ALLOWED_LOGO_EXTENSIONS):
        # Basic size check (using Content-Length header is approximate but quick)
        if request.content_length > MAX_LOGO_SIZE_MB * 1024 * 1024:
             current_app.logger.warning(f"Logo upload failed for client {client_id}: File too large (>{MAX_LOGO_SIZE_MB}MB).")
             return jsonify({"error": f"File exceeds maximum size of {MAX_LOGO_SIZE_MB}MB"}), 413 # Payload Too Large

        # More robust validation (optional but recommended): Check magic numbers/PIL
        try:
            # Read some bytes to check type without loading whole file
            file.seek(0) # Ensure reading from the start
            img = PILImage.open(BytesIO(file.read(1024*1024))) # Read up to 1MB for check
            img.verify() # Check if Pillow can identify it as an image
            file.seek(0) # Reset file pointer for saving
            # Optional: Check img.format against allowed types if needed
            current_app.logger.debug(f"Logo file validated using Pillow for client {client_id}: format {img.format}")
        except (UnidentifiedImageError, ValueError, TypeError) as img_err:
             current_app.logger.warning(f"Logo upload failed for client {client_id}: Invalid image file. Error: {img_err}")
             return jsonify({"error": "Invalid or corrupted image file"}), 400
        except Exception as e:
             current_app.logger.error(f"Unexpected error during logo image validation for client {client_id}: {e}", exc_info=True)
             return jsonify({"error": "Server error during file validation"}), 500


        # --- Save File ---
        filename = secure_filename(file.filename)
        # Create a unique filename (e.g., using client_id and timestamp or UUID)
        # Example: f"{client_id}_{int(time.time())}_{filename}"
        # For simplicity now, let's just use the client_id prefix if available
        unique_filename = f"{client_id}_{filename}" if client_id else f"{uuid.uuid4()}_{filename}"

        # Ensure the target directory exists
        logo_dir = LOGO_UPLOAD_FOLDER # Defined globally
        if not os.path.exists(logo_dir):
            try:
                os.makedirs(logo_dir)
                current_app.logger.info(f"Created logo upload directory: {logo_dir}")
            except OSError as e:
                current_app.logger.error(f"Failed to create logo directory '{logo_dir}': {e}")
                return jsonify({"error": "Server error creating storage directory"}), 500

        save_path = os.path.join(logo_dir, unique_filename)
        current_app.logger.info(f"Attempting to save logo for client {client_id} to: {save_path}")

        try:
            # TODO: Before saving, potentially delete the *old* logo file for this client_id
            # old_filename = get_current_logo_filename(client_id)
            # if old_filename:
            #    old_path = os.path.join(logo_dir, old_filename)
            #    if os.path.exists(old_path):
            #        os.remove(old_path)
            #        current_app.logger.info(f"Removed old logo file: {old_path}")

            file.save(save_path)
            current_app.logger.info(f"Successfully saved logo for client {client_id} as: {unique_filename}")

            # --- Update Database/Settings ---
            set_current_logo_filename(client_id, unique_filename) # Placeholder call

            # --- Generate URL for response ---
            logo_url = None
            try:
                 logo_url = url_for('static', filename=f'logos/{unique_filename}', _external=True)
            except RuntimeError as e:
                 current_app.logger.error(f"Could not generate URL for newly uploaded logo '{unique_filename}': {e}")


            return jsonify({
                "message": "Logo uploaded successfully",
                "filename": unique_filename,
                "url": logo_url
                }), 200
        except Exception as e:
            current_app.logger.error(f"Failed to save logo file '{save_path}' for client {client_id}: {e}", exc_info=True)
            # Attempt to clean up partially saved file if it exists
            if os.path.exists(save_path):
                try: os.remove(save_path)
                except Exception as cleanup_e: current_app.logger.error(f"Error cleaning up failed logo save {save_path}: {cleanup_e}")
            return jsonify({"error": "Server error saving file"}), 500

    else:
        current_app.logger.warning(f"Logo upload failed for client {client_id}: File type not allowed ('{file.filename}').")
        return jsonify({"error": "File type not allowed"}), 400

# Ensure necessary imports are present at the top of the file:
# import os, uuid, time # uuid and time might be needed for unique filenames
# from werkzeug.utils import secure_filename
# from flask import request, jsonify, current_app, url_for
# from PIL import Image as PILImage, UnidentifiedImageError # For validation
# from io import BytesIO # For validation

# Ensure constants are defined (adjust paths as needed):
# LOGO_UPLOAD_FOLDER = os.path.join(current_app.root_path, 'static', 'logos') # Example using app context
# ALLOWED_LOGO_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'svg'}
# MAX_LOGO_SIZE_MB = 2

# --- Helper function to get current logo (can be expanded later) ---
def get_current_logo_filename(client_id):
    # TODO: Implement logic to retrieve the actual logo filename associated
    # with the client_id from a database or configuration.
    # For now, assume no logo is set or return a default if applicable.
    # Example: Check a User or Settings model associated with the client_id.
    # user_settings = UserSettings.query.filter_by(client_id=client_id).first()
    # return user_settings.logo_filename if user_settings else None
    return None # Placeholder

# --- Helper function to set current logo (can be expanded later) ---
def set_current_logo_filename(client_id, filename):
    # TODO: Implement logic to save the logo filename associated
    # with the client_id in a database or configuration.
    # Example: Update a User or Settings model.
    # user_settings = UserSettings.query.filter_by(client_id=client_id).first()
    # if user_settings:
    #     user_settings.logo_filename = filename
    #     db.session.commit()
    # else:
    #     # Handle case where settings don't exist? Create new record?
    #     pass
    current_app.logger.info(f"Logo filename '{filename}' set for client '{client_id}' (placeholder logic).")
    pass # Placeholder
