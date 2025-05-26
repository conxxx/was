# chatbot-backend/tests/test_ingestion_task.py

import unittest
from unittest.mock import patch, MagicMock, call # Import necessary mocking tools
import os

# Import the task function we want to test
# Need to adjust path if tests directory is not directly under chatbot-backend
# Assuming tests is a sibling of app, adjust import path:
import sys
# Add the parent directory (chatbot-backend) to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.ingestion import run_ingestion_task, RETRYABLE_EXCEPTIONS

# Import exceptions we might need to mock raising/catching
from google.api_core.exceptions import GoogleAPICallError, ServiceUnavailable, ResourceExhausted
import requests

# Mock the Flask app context dependencies if needed directly
# For simplicity, we'll often mock functions within the service directly

# --- Test Class ---
class TestIngestionTaskRetries(unittest.TestCase):

    # --- Test Scenario 1: Success on First Try ---
    # Patch targets relative to the module where they are LOOKED UP, not where they are defined
    @patch('app.services.ingestion.initialize_gcp_clients')
    @patch('app.services.ingestion.process_uploaded_files')
    @patch('app.services.ingestion.process_web_source')
    @patch('app.services.ingestion.generate_and_trigger_batch_update')
    @patch('app.services.ingestion.update_chatbot_status')
    @patch('app.services.ingestion.os.remove') # Mock file cleanup
    @patch('app.services.ingestion.os.path.exists') # Mock file check for cleanup
    @patch('app.services.ingestion.db.session.get') # Mock DB get for final status update
    @patch('app.services.ingestion.Chatbot') # Mock the Chatbot model used in final update
    @patch('app.services.ingestion.create_app') # Mock app creation
    def test_success_on_first_try(self,
                                  mock_create_app,
                                  mock_Chatbot,
                                  mock_db_get,
                                  mock_path_exists,
                                  mock_os_remove,
                                  mock_update_status,
                                  mock_generate_batch,
                                  mock_process_web,
                                  mock_process_files,
                                  mock_init_clients):

        # --- Mock Setup ---
        # 1. Mock Flask App and Context
        mock_app = MagicMock()
        mock_context = MagicMock()
        mock_app.app_context.return_value = mock_context
        # Configure the context manager mock
        mock_context.__enter__.return_value = None # Simulate entering 'with' block
        mock_context.__exit__.return_value = None # Simulate exiting 'with' block
        mock_create_app.return_value = mock_app
        # Mock logger on the app context
        mock_app.logger = MagicMock()
        # Mock config on the app context if initialize_gcp_clients uses it
        mock_app.config = {} # Add necessary config keys if needed by tested code

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
        # Call the task function directly, passing the mocked 'self'
        # Note: The task function itself handles the app context now
        run_ingestion_task(mock_task_self, test_chatbot_id, test_client_id, test_source_details)

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
        mock_db_get.assert_called_once_with(mock_Chatbot, test_chatbot_id)
        mock_chatbot_instance.complete_index_operation.assert_called_once_with(success=True, total_chunks=2) # 2 chunks total

# --- Add more test methods for other scenarios later ---

# --- Boilerplate to run tests ---
if __name__ == '__main__':
    # Ensure the path modification is effective if run directly
    # (Though typically tests are run via a test runner like 'python -m unittest discover')
    if '..' not in sys.path:
         sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    unittest.main()
