import os
import unittest
from unittest.mock import patch

# Import the Flask app factory
from app import create_app
# Import the base config to potentially pass modifications
from config import Config

class TestConfig(unittest.TestCase):

    def setUp(self):
        """Set up a test app instance for each test."""
        # Create a base app with testing flag
        # We might need to override settings for specific tests
        self.app = create_app(Config) # Use the base Config initially
        self.app.config.update({
            "TESTING": True,
            # Add other overrides if necessary, e.g., disable CSRF
            "WTF_CSRF_ENABLED": False,
            "SERVER_NAME": "localhost.test" # Often needed for context
        })
        self.app_context = self.app.app_context()
        self.app_context.push()

    def tearDown(self):
        """Clean up the app context."""
        self.app_context.pop()

    def test_config_loads_correct_embedding_model_default(self):
        """
        Test that the default EMBEDDING_MODEL_NAME is loaded correctly via app config.
        """
        # Access config through the test app's context
        self.assertEqual(self.app.config['EMBEDDING_MODEL_NAME'], "text-embedding-large-exp-03-07")

    @patch.dict(os.environ, {"EMBEDDING_MODEL_NAME": "override-model-name"})
    def test_config_loads_correct_embedding_model_override(self):
        """
        Test that EMBEDDING_MODEL_NAME can be overridden by an environment variable.
        """
        # To test env var override, we need to create a *new* app instance
        # *after* the environment variable has been patched.
        # The config object reads env vars when the app is created.

        # Create a new app instance *within* the patched environment
        test_app_override = create_app(Config)
        test_app_override.config.update({"TESTING": True, "SERVER_NAME": "localhost.test"})

        with test_app_override.app_context():
             # Check the config value in the new app instance
             self.assertEqual(test_app_override.config['EMBEDDING_MODEL_NAME'], "override-model-name")

        # The original self.app instance created in setUp will still have the default value
        # as it was created before the patch. This is expected.
        self.assertEqual(self.app.config['EMBEDDING_MODEL_NAME'], "text-embedding-large-exp-03-07")


if __name__ == '__main__':
    unittest.main()