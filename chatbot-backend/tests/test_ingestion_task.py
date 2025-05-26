# chatbot-backend/tests/test_ingestion_task.py

import unittest
from unittest.mock import patch, MagicMock, call # Import necessary mocking tools
import os

# Import the task function we want to test
# Need to adjust path if tests directory is not directly under chatbot-backend
# Assuming tests is a sibling of app, adjust import path:
import sys
# Add the parent directory (chatbot-backend) to the Python path
# Use absolute path from this file's location
tests_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(tests_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import from app
# Import the internal function and exceptions
from app.services.ingestion import _perform_ingestion, RETRYABLE_EXCEPTIONS 
# Also need db for patching
from app import db

# Import exceptions we might need to mock raising/catching
from google.api_core.exceptions import GoogleAPICallError, ServiceUnavailable, ResourceExhausted
import requests
from flask import Flask # Import Flask to create a test app instance

# Mock the Flask app context dependencies if needed directly
# For simplicity, we'll often mock functions within the service directly

# --- Test Class ---
class TestIngestionTaskRetries(unittest.TestCase):

    # --- Test Scenario 1: Success on First Try ---
    # Patch targets relative to the module where they are LOOKED UP
    @patch('app.services.ingestion.initialize_gcp_clients')
    @patch('app.services.ingestion.process_uploaded_files')
    @patch('app.services.ingestion.process_web_source')
    @patch('app.services.ingestion.generate_and_trigger_batch_update')
    @patch('app.services.ingestion.update_chatbot_status')
    @patch('app.services.ingestion.os.remove') # Mock file cleanup
    @patch('app.services.ingestion.os.path.exists') # Mock file check for cleanup
    @patch('app.db.session.get') # Correct patch target: where db is looked up
    @patch('app.models.Chatbot') # Correct patch target: where Chatbot is looked up
    # No longer need to patch create_app, we'll manage context manually
    def test_success_on_first_try(self,
                                  # mock_create_app removed
                                  mock_Chatbot,    # Patched 'app.models.Chatbot'
                                  mock_db_get,     # Patched 'app.db.session.get'
                                  mock_path_exists,
                                  mock_os_remove,
                                  mock_update_status,
                                  mock_generate_batch,
                                  mock_process_web,
                                  mock_process_files,
                                  mock_init_clients):

        # --- Mock Setup ---
        # 1. Create a real Flask app for context
        test_app = Flask(__name__)
        test_app.config['TESTING'] = True
        # Add other necessary config if needed by the code under test
        # e.g., test_app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        # Mock logger directly onto the test_app instance if needed,
        # otherwise current_app.logger will use Flask's default logger
        test_app.logger = MagicMock()

        # 2. Mock GCP Clients Initialization
        mock_storage_client = MagicMock()
        mock_bucket = MagicMock()
        mock_embedding_model = MagicMock()
        mock_index_client = MagicMock()
        mock_init_clients.return_value = (mock_storage_client, mock_bucket, mock_embedding_model, mock_index_client)

        # 3. Mock File/Web Processing Results (return some dummy chunk data)
        mock_process_files.return_value = [{'id': 'file_vec_1', 'text': 'chunk from file', 'source': 'file://test.txt', 'chunk_index': 0}]
        mock_process_web.return_value = [{'id': 'web_vec_1', 'text': 'chunk from web', 'source': 'http://example.com', 'chunk_index': 0}]

        # 4. Mock Embedding/Indexing (return True for success)
        mock_generate_batch.return_value = True

        # 5. Mock File Cleanup
        mock_path_exists.return_value = True # Assume files exist for cleanup

        # 6. Mock Final Status DB Update
        mock_chatbot_instance = MagicMock()
        # Add the necessary attributes/methods used by complete_index_operation
        mock_chatbot_instance.status = 'Processing'
        mock_chatbot_instance.client_id = 'client123' # Needed for potential SSE push in complete_index_operation
        mock_db_get.return_value = mock_chatbot_instance # Simulate finding the chatbot
        # We mocked the Chatbot class itself, so no need to mock Chatbot() call

        # 7. Mock Celery Task Instance ('self')
        mock_task_self = MagicMock()
        mock_task_self.request = MagicMock()
        mock_task_self.request.retries = 0 # Start with 0 retries
        mock_task_self.max_retries = 3
        mock_task_self.default_retry_delay = 30
        # Ensure retry method is *not* called
        mock_task_self.retry = MagicMock(side_effect=Exception("Retry should not be called in success scenario"))

        # --- Test Data ---
        test_chatbot_id = 1
        test_client_id = 'client123'
        test_source_details = {
            'files_uploaded': ['test.txt'], # Basename used by task
            'selected_urls': ['http://example.com']
        }
        # Define UPLOAD_FOLDER as expected by the task logic
        # Patching os.path.join might be safer if UPLOAD_FOLDER location is uncertain
        test_upload_folder = 'uploads' # Matches the hardcoded value in task

        # --- Execute Task ---
        # --- Execute Task within App Context ---
        # Push the context of the real test app
        with test_app.app_context():
            # Call the internal logic function directly, passing the mocked task instance
            _perform_ingestion(mock_task_self, test_chatbot_id, test_client_id, test_source_details)

        # --- Assertions ---
        # 1. Check external functions were called correctly
        mock_init_clients.assert_called_once()
        # Need to construct the expected full path for process_uploaded_files call
        expected_file_path = os.path.join(test_upload_folder, 'test.txt')
        mock_process_files.assert_called_once_with(
            test_chatbot_id, test_client_id, [expected_file_path], mock_storage_client, mock_bucket, mock_embedding_model, mock_index_client, task_instance=mock_task_self
        )
        mock_process_web.assert_called_once_with(
            test_chatbot_id, test_client_id, ['http://example.com'], 'web_filtered', mock_storage_client, mock_bucket, mock_embedding_model, mock_index_client, task_instance=mock_task_self
        )
        # Check generate_and_trigger_batch_update call args
        expected_chunks_data = [
            {'id': 'file_vec_1', 'text': 'chunk from file', 'source': 'file://test.txt', 'chunk_index': 0},
            {'id': 'web_vec_1', 'text': 'chunk from web', 'source': 'http://example.com', 'chunk_index': 0}
        ]
        mock_generate_batch.assert_called_once_with(
            task_instance=mock_task_self,
            chatbot_id=test_chatbot_id,
            client_id=test_client_id,
            embedding_model=mock_embedding_model,
            bucket=mock_bucket,
            index_client=mock_index_client,
            chunks_data=expected_chunks_data, # Check combined list
            storage_client=mock_storage_client
        )

        # 2. Check cleanup was attempted
        mock_path_exists.assert_called_once_with(expected_file_path)
        mock_os_remove.assert_called_once_with(expected_file_path)

        # 3. Check retry was NOT called
        mock_task_self.retry.assert_not_called()

        # 4. Check final status update
        # Check intermediate status updates
        mock_update_status.assert_has_calls([
            call(test_chatbot_id, "Processing - Starting"),
            # Calls from process_uploaded_files/process_web_source might be mocked out
            # Add calls expected from generate_and_trigger_batch_update if not mocked
        ], any_order=True) # Order might vary slightly

        # Check the final call via the mocked Chatbot instance method
        # Ensure the mock_Chatbot class itself is used for the get call
        mock_db_get.assert_called_once_with(mock_Chatbot, test_chatbot_id)
        mock_chatbot_instance.complete_index_operation.assert_called_once_with(success=True, total_chunks=2) # 2 chunks total

# --- Add more test methods for other scenarios later ---

# --- Boilerplate to run tests ---
if __name__ == '__main__':
    # Ensure the path modification is effective if run directly
    # (Though typically tests are run via a test runner like 'python -m unittest discover')
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(tests_dir, '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    unittest.main()
