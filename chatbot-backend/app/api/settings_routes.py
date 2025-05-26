# chatbot-backend/app/api/settings_routes.py
import os
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import logging

# Define allowed extensions and max size (e.g., 2MB)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 2 * 1024 * 1024 # 2MB

bp = Blueprint('settings', __name__)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @bp.route('/logo', methods=['POST']) # Removed - Handled by PUT /chatbots/{id}
# def upload_logo():
#     """Handles company logo uploads."""
#     logger.info("Received request to /api/settings/logo")
#
#     # Check if the post request has the file part
#     if 'logo' not in request.files:
#         logger.warning("No 'logo' file part in request")
#         return jsonify({"error": "No file part"}), 400
#
#     file = request.files['logo']
#
#     # If the user does not select a file, the browser submits an
#     # empty file without a filename.
#     if file.filename == '':
#         logger.warning("No selected file")
#         return jsonify({"error": "No selected file"}), 400
#
#     # Check file size (using request.content_length for efficiency before reading)
#     # Note: request.content_length might not be perfectly accurate for multipart/form-data
#     # A more robust check might involve reading the stream up to the limit.
#     # However, this provides a basic check.
#     if request.content_length and request.content_length > MAX_CONTENT_LENGTH:
#          logger.warning(f"File size ~{request.content_length} exceeds limit {MAX_CONTENT_LENGTH}")
#          return jsonify({"error": f"File exceeds maximum size of {MAX_CONTENT_LENGTH // 1024 // 1024}MB"}), 413 # Payload Too Large
#
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename) # Sanitize filename
#         # Construct the upload path relative to the app directory
#         # Assumes 'app' is the root package directory where __init__.py resides
#         # Adjust if your project structure is different
#         # current_app.root_path gives the path to the 'app' directory
#         upload_folder = os.path.join(current_app.root_path, '..', 'uploads', 'logos')
#
#         # Ensure the upload directory exists
#         try:
#             os.makedirs(upload_folder, exist_ok=True)
#             logger.info(f"Ensured upload directory exists: {upload_folder}")
#         except OSError as e:
#             logger.error(f"Could not create upload directory {upload_folder}: {e}", exc_info=True)
#             return jsonify({"error": "Could not create upload directory"}), 500
#
#         save_path = os.path.join(upload_folder, filename)
#
#         try:
#             # Consider adding a check here to prevent overwriting existing files if needed
#             # e.g., by adding a UUID or timestamp to the filename
#             file.save(save_path)
#             logger.info(f"Logo saved successfully to {save_path}")
#             # Store the filename in a simple text file for persistence
#             config_file_path = os.path.join(upload_folder, 'current_logo.txt')
#             try:
#                 with open(config_file_path, 'w') as f:
#                     f.write(filename)
#                 logger.info(f"Current logo filename '{filename}' saved to {config_file_path}")
#             except IOError as e:
#                 # Log the error but proceed, as the upload itself was successful
#                 logger.error(f"Could not write current logo filename to {config_file_path}: {e}", exc_info=True)
#
#             return jsonify({"message": "Logo uploaded successfully", "filename": filename}), 200
#         except Exception as e:
#             logger.error(f"Could not save file to {save_path}: {e}", exc_info=True)
#             return jsonify({"error": "Could not save file"}), 500
#     else:
#         logger.warning(f"File type not allowed: {file.filename}")
#         return jsonify({"error": "File type not allowed"}), 400


# @bp.route('/logo', methods=['GET']) # Removed - Widget gets logo URL from /widget-config
# def get_logo():
#     """Retrieves the filename of the currently configured company logo."""
#     logger.info("Received request to GET /api/settings/logo")
#     logo_filename = None
#     try:
#         # Determine the path to the config file storing the current logo filename
#         # This path needs to be consistent with where the POST endpoint saves it
#         upload_folder = os.path.join(current_app.root_path, '..', 'uploads', 'logos')
#         config_file_path = os.path.join(upload_folder, 'current_logo.txt')
#
#         if os.path.exists(config_file_path):
#             with open(config_file_path, 'r') as f:
#                 logo_filename = f.read().strip()
#             if not logo_filename: # Handle empty file case
#                 logo_filename = None
#                 logger.info("Found current_logo.txt, but it was empty.")
#             else:
#                  logger.info(f"Retrieved current logo filename '{logo_filename}' from {config_file_path}")
#         else:
#             logger.info(f"Current logo config file not found: {config_file_path}")
#
#     except IOError as e:
#         logger.error(f"Could not read current logo filename from {config_file_path}: {e}", exc_info=True)
#         # Decide if an error should be returned or just null filename
#         # Returning null seems more appropriate if the file can't be read
#         logo_filename = None
#     except Exception as e:
#         logger.error(f"An unexpected error occurred while retrieving logo filename: {e}", exc_info=True)
#         # Return 500 for unexpected errors
#         return jsonify({"error": "An internal error occurred"}), 500
#
#     return jsonify({"logo_filename": logo_filename}), 200