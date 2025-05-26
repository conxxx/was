# app/sse_utils.py
import json
import redis
from flask import current_app
from config import Config # Import Config to get Redis URL

# --- Redis Pub/Sub Configuration ---
REDIS_URL = Config.CELERY_BROKER_URL # Use the same Redis as Celery
SSE_CHANNEL = "chatbot-status-updates" # Define a channel name
# -----------------------------------

# --- Redis Connection ---
# Create a connection pool for efficiency
# decode_responses=True handles decoding from bytes to strings
try:
    redis_pool = redis.ConnectionPool.from_url(REDIS_URL, decode_responses=True)
    redis_client = redis.Redis(connection_pool=redis_pool)
    # Test connection
    redis_client.ping()
    print(f"SSE Utils: Connected to Redis at {REDIS_URL}")
except Exception as e:
    print(f"ERROR: SSE Utils: Failed to connect to Redis at {REDIS_URL} - {e}")
    # Fallback or handle error appropriately - maybe disable SSE?
    redis_client = None
# ------------------------


def push_status_update(chatbot_id, status, client_id):
    """Publishes a status update message to the Redis Pub/Sub channel."""
    logger = current_app.logger if current_app else None # Get logger if in app context

    if not redis_client:
        if logger:
            logger.error(f"SSE Push: Cannot publish status for chatbot {chatbot_id}, Redis client not available.")
        else:
            print(f"ERROR: SSE Push: Cannot publish status for chatbot {chatbot_id}, Redis client not available.")
        return

    message = json.dumps({
        'chatbot_id': chatbot_id,
        'status': status,
        'client_id': client_id # Include client_id for filtering on subscriber side
    })

    try:
        if logger:
            logger.debug(f"SSE Push: Publishing update to channel '{SSE_CHANNEL}' for chatbot {chatbot_id} (Status: {status}, Client: {client_id})")
        
        published_count = redis_client.publish(SSE_CHANNEL, message)
        
        if logger:
            if published_count > 0:
                logger.debug(f"SSE Push: Message published to {published_count} subscriber(s).")
            else:
                logger.debug(f"SSE Push: Message published, but no active subscribers on channel '{SSE_CHANNEL}'.")

    except Exception as e:
        log_msg = f"SSE Push: FAILED to publish update for chatbot {chatbot_id} to Redis channel '{SSE_CHANNEL}': {e}"
        if logger:
             logger.error(log_msg, exc_info=True)
        else:
             print(f"ERROR: {log_msg}")

# Note: The sse_message_queue = queue.Queue() line is removed.
