# --- GEVENT PATCHING ---
# Apply patch *before* importing Flask, app factory, or anything using networking/ssl.
try:
    import gevent.monkey
    gevent.monkey.patch_all()
    print("Gevent monkey patching applied in run.py.") # Confirmation log
except ImportError:
    print("Gevent not installed in run.py, skipping monkey patching.")
# -----------------------

import os
# Now import your app and other necessary modules
from app import create_app, db # Assuming create_app is your Flask app factory
from app.models import User, Chatbot # Import models if needed directly here
from app.api.routes import get_rag_service
from celery_worker import celery_app # Import the celery instance
from celery import Task # Keep Task import if needed elsewhere, otherwise remove

app = create_app()

# --- Removed Celery Context Task Setup ---
# Context will be handled manually within each task function
# -----------------------------------------

# Initialize services only in the main process or when not in debug mode
if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or not app.debug:
    with app.app_context():
        print("Attempting to eagerly initialize services...")
        try:
            get_rag_service() # Call the function to trigger initialization
            print("Services initialized.")
        except Exception as e:
            print(f"ERROR: Service initialization failed during startup: {e}")
            # Decide if you want the app to exit if critical services fail
            # import sys
            # sys.exit(1)
    # ------------------------------------------------------
@app.shell_context_processor
def make_shell_context():
    return {'db': db, 'User': User, 'Chatbot': Chatbot}

if __name__ == '__main__':
    # Use Flask's built-in development server (threaded by default)
    # Use host='0.0.0.0' to allow external access if needed
    # Use debug=False for production or based on config
    # Use threaded=True if not using gevent for the web server itself,
    # but since you use gevent heavily, relying on gevent's concurrency might be better.
    # Consider using a proper WSGI server like gunicorn with gevent workers for production.
    host = os.environ.get('FLASK_RUN_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_RUN_PORT', 5001))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

    print(f"Starting Flask development server on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug) # Removed threaded=True, let gevent handle it
    # http_server = WSGIServer(('0.0.0.0', port), app) # Removed gevent server
    # http_server.serve_forever() # Removed gevent server
