import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import google.api_core.exceptions

# Assuming RagService is in app.services.rag_service
# Adjust the import path if necessary
from chatbot_backend.app.services.rag_service import RagService
# Assuming Flask app context is used for config, or direct config import
# from chatbot_backend.app import create_app
from chatbot_backend import config # Or adjust as needed

# Mock the TextEmbeddingModel class and its methods
MockTextEmbeddingModel = MagicMock()
MockEmbeddingResponse = MagicMock()
MockEmbedding = MagicMock()

class TestRagService(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        # Reset mocks for each test
        MockTextEmbeddingModel.reset_mock()
        MockEmbeddingResponse.reset_mock()
        MockEmbedding.reset_mock()

        # Mock the config values needed by RagService
        self.mock_config = {
            'EMBEDDING_MODEL_NAME': 'test-model-from-config',
            'GCP_PROJECT_ID': 'test-project',
            'GCP_LOCATION': 'test-location',
            # Add other necessary config mocks
        }

        # Patch the config access within the rag_service module
        # This assumes rag_service imports config directly. Adjust if it uses app.config
        self.config_patcher = patch('chatbot_backend.app.services.rag_service.config', MagicMock(**self.mock_config))
        self.mock_config_obj = self.config_patcher.start()

        # Patch the Vertex AI SDK class
        self.model_patcher = patch('chatbot_backend.app.services.rag_service.TextEmbeddingModel', MockTextEmbeddingModel)
        self.mock_model_cls = self.model_patcher.start()
        self.mock_model_instance = MagicMock()
        self.mock_model_cls.from_pretrained.return_value = self.mock_model_instance

    def tearDown(self):
        """Tear down test fixtures."""
        self.config_patcher.stop()
        self.model_patcher.stop()

    def test_rag_service_initializes_correct_embedding_model(self):
        """
        Test RagService initializes the correct embedding model from config.
        """
        RagService() # Initialize the service
        self.mock_model_cls.from_pretrained.assert_called_once_with('test-model-from-config')

    @patch('chatbot_backend.app.services.rag_service.logging') # Mock logging
    def test_rag_service_init_handles_model_load_error(self, mock_logging):
        """
        Test RagService handles errors during embedding model initialization.
        """
        self.mock_model_cls.from_pretrained.side_effect = google.api_core.exceptions.NotFound("Model not found")

        with self.assertRaises(google.api_core.exceptions.NotFound): # Or expect specific handling
             RagService()
        # Optionally check logging:
        # mock_logging.error.assert_called_with(expected_error_message)


    def test_generate_embeddings_uses_retrieval_query_task_type(self):
        """
        Test generate_embeddings calls the SDK with task_type='RETRIEVAL_QUERY'.
        """
        service = RagService()
        queries = ["query 1", "query 2"]
        service.generate_embeddings(queries)
        self.mock_model_instance.get_embeddings.assert_called_once_with(queries, task_type="RETRIEVAL_QUERY")

    def test_generate_embeddings_returns_vectors(self):
        """
        Test generate_embeddings extracts and returns embedding vectors correctly.
        """
        service = RagService()
        queries = ["query 1"]
        expected_vector = [0.1, 0.2, 0.3]

        # Mock the response structure
        mock_embedding = MagicMock()
        type(mock_embedding).values = PropertyMock(return_value=expected_vector) # Mocking the 'values' property
        mock_response = [mock_embedding] # Simulate the list response
        self.mock_model_instance.get_embeddings.return_value = mock_response

        embeddings, error = service.generate_embeddings(queries)

        self.assertIsNone(error)
        self.assertEqual(embeddings, [expected_vector])
        self.mock_model_instance.get_embeddings.assert_called_once_with(queries, task_type="RETRIEVAL_QUERY")

    @patch('chatbot_backend.app.services.rag_service.logging')
    def test_generate_embeddings_handles_api_error(self, mock_logging):
        """
        Test generate_embeddings handles API errors during embedding generation.
        """
        service = RagService()
        queries = ["query 1"]
        self.mock_model_instance.get_embeddings.side_effect = google.api_core.exceptions.InternalServerError("API error")

        embeddings, error = service.generate_embeddings(queries)

        self.assertIsNone(embeddings)
        self.assertIsNotNone(error)
        self.assertIn("API error", error)
        # Optionally check logging:
        # mock_logging.error.assert_called_with(expected_error_message)

    @patch('chatbot_backend.app.services.rag_service.logging')
    def test_generate_embeddings_handles_partial_failure(self, mock_logging):
        """
        Test generate_embeddings handles cases where fewer embeddings are returned than expected.
        """
        service = RagService()
        queries = ["query 1", "query 2"] # Request 2 embeddings

        # Mock the response structure - return only 1 embedding
        mock_embedding = MagicMock()
        type(mock_embedding).values = PropertyMock(return_value=[0.1, 0.2, 0.3])
        mock_response = [mock_embedding] # Only one embedding returned
        self.mock_model_instance.get_embeddings.return_value = mock_response

        embeddings, error = service.generate_embeddings(queries)

        self.assertIsNone(embeddings)
        self.assertIsNotNone(error)
        self.assertIn("Mismatch", error) # Check for a specific error message
        # Optionally check logging:
        # mock_logging.warning.assert_called_with(expected_warning_message)


if __name__ == '__main__':
    unittest.main()