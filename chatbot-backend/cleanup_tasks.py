# chatbot-backend/cleanup_tasks.py
import logging
from datetime import datetime, timedelta
from app import db, create_app  # Assuming create_app is your Flask app factory
from celery_worker import celery_app # Import the Celery app instance
from app.models import Chatbot, ChatMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@celery_app.task(name='cleanup_tasks.cleanup_old_chat_messages')
def cleanup_old_chat_messages():
    """
    Deletes chat messages older than the configured retention period for each chatbot
    where history saving is enabled and a retention period is set.

    This function is designed to be run periodically (e.g., daily via a scheduler like
    Celery Beat, APScheduler, or a cron job).
    """
    app = create_app() # Create an app instance to work within the app context
    with app.app_context():
        logger.info("Starting chat message cleanup task...")
        deleted_count_total = 0
        chatbots_processed = 0

        try:
            # Find chatbots with history saving enabled and a specific retention period set
            chatbots_to_clean = Chatbot.query.filter(
                Chatbot.save_history_enabled == True,
                Chatbot.history_retention_days.isnot(None) # Ensure retention days is set
            ).all()

            chatbots_processed = len(chatbots_to_clean)
            logger.info(f"Found {chatbots_processed} chatbots with history retention policies.")

            for chatbot in chatbots_to_clean:
                retention_days = chatbot.history_retention_days
                if retention_days <= 0: # Skip if retention is zero or negative (shouldn't happen ideally)
                    logger.warning(f"Skipping chatbot ID {chatbot.id}: Invalid retention period ({retention_days} days).")
                    continue

                cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
                logger.info(f"Processing chatbot ID {chatbot.id} (Name: {chatbot.name}): Deleting messages older than {cutoff_date.strftime('%Y-%m-%d %H:%M:%S')} UTC ({retention_days} days).")

                try:
                    # Query and delete old messages for this specific chatbot
                    # Using lazy='dynamic' on chat_messages relationship allows filtering before loading
                    messages_to_delete = chatbot.chat_messages.filter(ChatMessage.timestamp < cutoff_date)
                    
                    # Count before deleting (more efficient than loading all then counting)
                    deleted_count_chatbot = messages_to_delete.count() 

                    if deleted_count_chatbot > 0:
                        # Perform the deletion
                        messages_to_delete.delete(synchronize_session=False) # Use False for bulk deletes
                        db.session.commit() # Commit after each chatbot's deletion
                        deleted_count_total += deleted_count_chatbot
                        logger.info(f"Deleted {deleted_count_chatbot} messages for chatbot ID {chatbot.id}.")
                    else:
                        logger.info(f"No messages older than retention period found for chatbot ID {chatbot.id}.")
                        # No need to commit if nothing was deleted

                except Exception as e:
                    db.session.rollback() # Rollback changes for this specific chatbot on error
                    logger.error(f"Error deleting messages for chatbot ID {chatbot.id}: {e}", exc_info=True)

            logger.info(f"Chat message cleanup task finished. Processed {chatbots_processed} chatbots. Total messages deleted: {deleted_count_total}.")

        except Exception as e:
            db.session.rollback() # Rollback any potential partial commits if the outer loop fails
            logger.error(f"Critical error during chat message cleanup task: {e}", exc_info=True)

if __name__ == '__main__':
    # This allows running the cleanup manually via `python cleanup_tasks.py`
    # Ensure your Flask app environment (e.g., DATABASE_URL) is configured
    # when running this script directly.
    logger.info("Running cleanup manually...")
    cleanup_old_chat_messages()
    logger.info("Manual cleanup finished.")