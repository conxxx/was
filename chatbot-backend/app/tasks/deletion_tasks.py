# tasks/deletion_tasks.py

import logging
import hashlib # Added import
import json # Added for source_details manipulation
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import select

from celery_worker import celery_app
from app import db, create_app
from app.models import Chatbot, VectorIdMapping, ChatMessage, DetailedFeedback, UsageLog
from app.services.rag_service import RagService, VertexAIDeletionError # Import only needed exceptions
from app.api.routes import get_rag_service
from google.cloud.exceptions import NotFound as GoogleNotFound

flask_app = create_app()
logger = logging.getLogger(__name__)

# --- Helper function to construct GCS path ---
def _construct_gcs_path_from_vector_id(vector_id: str, chatbot_id: int) -> str | None:
    """
    Constructs the expected GCS blob path from a vector ID and chatbot ID.
    Vector ID format: chatbot_<chatbot_id>_source_<source_hash>_chunk_<chunk_index>
    GCS Path format: chatbot_<chatbot_id>/source_<source_hash>/<chunk_index>.txt
    Returns: GCS path string or None if format is invalid.
    """
    parts = vector_id.split('_')
    # Expecting format: chatbot_<id>_source_<hash>_chunk_<index> (6 parts)
    if len(parts) == 6 and parts[0] == 'chatbot' and parts[2] == 'source' and parts[4] == 'chunk':
        # We already have chatbot_id passed in, but could verify parts[1] if needed
        hash_part = parts[3]
        index_part = parts[5]
        # Construct the GCS path using the passed chatbot_id
        return f"chatbot_{chatbot_id}/source_{hash_part}/{index_part}.txt"
    else:
        logger.warning(f"Invalid vector ID format for GCS path construction: {vector_id}")
        return None
# ---------------------------------------------

@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def delete_source_data_task(self, chatbot_id: int, source_identifier: str):
    """
    Celery task to delete all data associated with a specific source identifier
    for a given chatbot, including GCS files, Vector Store entries, and DB records.
    """
    logger.info(f"Task {self.request.id}: Starting deletion for source '{source_identifier}' from chatbot {chatbot_id}.")
    with flask_app.app_context():
        rag_service = None
        try:
            rag_service = get_rag_service()
            if rag_service is None:
                 logger.error(f"Task {self.request.id}: get_rag_service() returned None unexpectedly.")
                 raise self.retry(exc=Exception("Failed to obtain RagService instance via get_rag_service()."))

            # Determine the correct identifier for DB query and GCS hashing
            db_query_identifier = source_identifier
            gcs_hash_identifier = source_identifier
            details_removal_identifier = source_identifier # This is what's in source_details.files_uploaded
            is_file_source = False

            if not source_identifier.startswith(('http://', 'https://')):
                is_file_source = True
                # Input source_identifier is like "uuid_filename.txt"
                # For DB query and GCS hash, we need "file://uuid_filename.txt"
                db_query_identifier = f"file://{source_identifier}"
                gcs_hash_identifier = f"file://{source_identifier}"
                logger.info(f"Task {self.request.id}: Identified file source. DB/GCS identifier: '{db_query_identifier}', Details removal ID: '{details_removal_identifier}'")
            
            # 1. Identify Data in Database
            mappings_to_delete = VectorIdMapping.query.filter_by(
                chatbot_id=chatbot_id,
                source_identifier=db_query_identifier # Use the potentially prefixed identifier
            ).all()

            if not mappings_to_delete:
                logger.warning(f"Task {self.request.id}: No vector mappings found for chatbot {chatbot_id} and DB query source '{db_query_identifier}' (original input: '{source_identifier}').")
                # Even if no mappings, try to clean up source_details as it might be orphaned
                try:
                    chatbot = db.session.get(Chatbot, chatbot_id)
                    if chatbot and chatbot.source_details:
                        logger.info(f"Task {self.request.id}: Attempting to clean up source_details for '{details_removal_identifier}' despite no vector mappings found.")
                        source_details_dict = json.loads(chatbot.source_details)
                        removed_from_details = False
                        if is_file_source and 'files_uploaded' in source_details_dict and details_removal_identifier in source_details_dict['files_uploaded']:
                            source_details_dict['files_uploaded'].remove(details_removal_identifier)
                            removed_from_details = True
                        elif not is_file_source and 'selected_urls' in source_details_dict and details_removal_identifier in source_details_dict['selected_urls']:
                            source_details_dict['selected_urls'].remove(details_removal_identifier)
                            removed_from_details = True
                        
                        if removed_from_details:
                            new_source_types = []
                            if source_details_dict.get('selected_urls') or source_details_dict.get('original_url'):
                                if source_details_dict.get('selected_urls'): new_source_types.append('Web_Filtered')
                                elif source_details_dict.get('original_url'): new_source_types.append('Web_Direct')
                            if source_details_dict.get('files_uploaded'): new_source_types.append('Files')
                            chatbot.source_type = '+'.join(new_source_types) if new_source_types else 'None'
                            chatbot.source_details = json.dumps(source_details_dict)
                            db.session.add(chatbot)
                            db.session.commit()
                            logger.info(f"Task {self.request.id}: Successfully cleaned up '{details_removal_identifier}' from source_details for chatbot {chatbot_id}.")
                            return f"Success: Source '{details_removal_identifier}' removed from details (no vector data found)."
                        else:
                            logger.info(f"Task {self.request.id}: Source '{details_removal_identifier}' not found in source_details for chatbot {chatbot_id}.")
                            return f"Success: No data found for source '{db_query_identifier}' (original input: '{source_identifier}')."
                    else:
                        logger.info(f"Task {self.request.id}: Chatbot or source_details not found for cleanup of '{details_removal_identifier}'.")
                        return f"Success: No data found for source '{db_query_identifier}' (original input: '{source_identifier}')."
                except Exception as e_cleanup_details:
                    logger.error(f"Task {self.request.id}: Error during source_details cleanup for '{details_removal_identifier}': {e_cleanup_details}", exc_info=True)
                    db.session.rollback() # Rollback if cleanup fails
                    return f"Error during source_details cleanup for '{details_removal_identifier}' (no vector data found)."


            vector_ids_to_delete = [mapping.vector_id for mapping in mappings_to_delete]
            logger.info(f"Task {self.request.id}: Found {len(vector_ids_to_delete)} vector IDs to delete for DB query source '{db_query_identifier}'.")

            # --- 1b. Identify and Delete Associated GCS Files (Revised to delete prefix) ---
            gcs_init_success = False
            if rag_service:
                try:
                    logger.info(f"Task {self.request.id}: Explicitly ensuring clients initialized before GCS deletion...")
                    if rag_service._ensure_clients_initialized():
                         gcs_init_success = True
                         logger.info(f"Task {self.request.id}: Client initialization check successful.")
                    else:
                         logger.error(f"Task {self.request.id}: Client initialization check failed: {rag_service.initialization_error}")
                except Exception as init_ex:
                    logger.error(f"Task {self.request.id}: Exception during client initialization check: {init_ex}", exc_info=True)

            deleted_files_count = 0
            failed_files_count = 0
            if gcs_init_success and rag_service.bucket:
                bucket_name = rag_service.bucket.name
                try:
                    # Calculate GCS prefix using gcs_hash_identifier
                    hashed_source = hashlib.sha256(gcs_hash_identifier.encode()).hexdigest()[:16]
                    source_gcs_prefix = f"chatbot_{chatbot_id}/source_{hashed_source}/"
                    logger.info(f"Task {self.request.id}: Attempting to delete all GCS objects under prefix '{source_gcs_prefix}' (from GCS hash ID: '{gcs_hash_identifier}') from bucket '{bucket_name}'...")

                    blobs_to_delete = list(rag_service.bucket.list_blobs(prefix=source_gcs_prefix))
                    if not blobs_to_delete:
                        logger.info(f"Task {self.request.id}: No GCS objects found under prefix '{source_gcs_prefix}'. Nothing to delete from GCS.")
                    else:
                        logger.info(f"Task {self.request.id}: Found {len(blobs_to_delete)} GCS objects to delete under prefix '{source_gcs_prefix}'.")
                        for blob in blobs_to_delete:
                            try:
                                blob.delete()
                                deleted_files_count += 1
                                logger.info(f"Task {self.request.id}: Successfully deleted GCS object: {blob.name}")
                            except GoogleNotFound:
                                logger.warning(f"Task {self.request.id}: GCS object not found (already deleted?): {blob.name}")
                                deleted_files_count += 1 # Count as success if not found
                            except Exception as e:
                                failed_files_count += 1
                                logger.error(f"Task {self.request.id}: Failed to delete GCS object '{blob.name}': {e}", exc_info=True)
                        logger.info(f"Task {self.request.id}: GCS prefix deletion summary for '{source_gcs_prefix}' - Deleted/Not Found: {deleted_files_count}, Failed: {failed_files_count}")
                        if failed_files_count > 0:
                            logger.warning(f"Task {self.request.id}: Proceeding despite {failed_files_count} GCS deletion failures for prefix '{source_gcs_prefix}'.")
                except Exception as prefix_calc_err:
                    logger.error(f"Task {self.request.id}: Error calculating GCS prefix or listing blobs for GCS hash ID '{gcs_hash_identifier}': {prefix_calc_err}", exc_info=True)
                    # This is critical, as we might not delete GCS data.
                    # Depending on policy, could raise self.retry or just log and continue with vector/DB deletion.
                    # For now, log and continue, as vector/DB deletion is separate.
                    failed_files_count = -1 # Indicate a general GCS failure
            else:
                logger.error(f"Task {self.request.id}: CRITICAL - Cannot delete GCS files for GCS hash ID '{gcs_hash_identifier}'. Bucket initialization failed or rag_service is None.")

            # 2. Delete from Vector Store
            if vector_ids_to_delete:
                try:
                    rag_service.delete_datapoints(vector_ids_to_delete)
                    logger.info(f"Task {self.request.id}: Successfully requested deletion of {len(vector_ids_to_delete)} vectors from Vertex AI for DB query source '{db_query_identifier}'.")
                except VertexAIDeletionError as e:
                    logger.error(f"Task {self.request.id}: Failed to delete vectors from Vertex AI for DB query source '{db_query_identifier}': {e}")
                    db.session.rollback() # Rollback before retry
                    raise self.retry(exc=e, countdown=60)
                except Exception as e:
                    logger.error(f"Task {self.request.id}: Unexpected error during Vertex AI deletion for DB query source '{db_query_identifier}': {e}", exc_info=True)
                    db.session.rollback() # Rollback before retry
                    raise self.retry(exc=e)

            # 3. Update Chatbot.source_details in Database
            try:
                chatbot = db.session.get(Chatbot, chatbot_id)
                if chatbot:
                    logger.info(f"Task {self.request.id}: Updating source_details for chatbot {chatbot_id} to remove '{source_identifier}'.")
                    current_source_details_str = chatbot.source_details
                    if current_source_details_str:
                        try:
                            source_details_dict = json.loads(current_source_details_str)
                            removed_from_dict = False

                            if is_file_source:
                                # details_removal_identifier is the plain "uuid_filename.txt"
                                if 'files_uploaded' in source_details_dict and isinstance(source_details_dict['files_uploaded'], list):
                                    if details_removal_identifier in source_details_dict['files_uploaded']:
                                        source_details_dict['files_uploaded'].remove(details_removal_identifier)
                                        removed_from_dict = True
                                        logger.info(f"Task {self.request.id}: Removed file '{details_removal_identifier}' from files_uploaded in source_details.")
                            elif details_removal_identifier.startswith(('http://', 'https://')): # It's a URL
                                if 'selected_urls' in source_details_dict and isinstance(source_details_dict['selected_urls'], list):
                                    if details_removal_identifier in source_details_dict['selected_urls']:
                                        source_details_dict['selected_urls'].remove(details_removal_identifier)
                                        removed_from_dict = True
                                        logger.info(f"Task {self.request.id}: Removed URL '{details_removal_identifier}' from selected_urls in source_details.")
                            
                            if removed_from_dict:
                                # Recalculate source_type
                                new_source_types = []
                                # Check selected_urls first, then original_url as a fallback for Web type
                                if source_details_dict.get('selected_urls'):
                                     new_source_types.append('Web_Filtered')
                                elif source_details_dict.get('original_url'): # Only consider original_url if selected_urls is empty
                                     new_source_types.append('Web_Direct') # Or a generic 'Web'
                                if source_details_dict.get('files_uploaded'):
                                    new_source_types.append('Files')
                                
                                chatbot.source_type = '+'.join(new_source_types) if new_source_types else 'None'
                                chatbot.source_details = json.dumps(source_details_dict)
                                db.session.add(chatbot)
                                logger.info(f"Task {self.request.id}: Updated source_details and source_type for chatbot {chatbot_id}. New type: {chatbot.source_type}, Details (first 200 chars): {chatbot.source_details[:200]}")
                            else:
                                logger.warning(f"Task {self.request.id}: Identifier '{details_removal_identifier}' not found within parsed source_details of chatbot {chatbot_id}.")

                        except json.JSONDecodeError:
                            logger.error(f"Task {self.request.id}: Failed to parse source_details JSON for chatbot {chatbot_id} during update: '{current_source_details_str}'", exc_info=True)
                        except Exception as e_sd_update:
                             logger.error(f"Task {self.request.id}: Error updating source_details for chatbot {chatbot_id}: {e_sd_update}", exc_info=True)
                    else:
                        logger.warning(f"Task {self.request.id}: Chatbot {chatbot_id} has no source_details to update.")
                else:
                    logger.warning(f"Task {self.request.id}: Chatbot {chatbot_id} not found during source_details update step.")
            except Exception as e_chatbot_fetch:
                logger.error(f"Task {self.request.id}: Error fetching chatbot {chatbot_id} for source_details update: {e_chatbot_fetch}", exc_info=True)
                # This is a non-critical error for the deletion flow, log and continue with VectorIdMapping deletion.

            # 4. Delete VectorIdMapping from Database
            try:
                # Use the IDs from the mappings list we already fetched
                mapping_ids_to_delete = [m.id for m in mappings_to_delete]
                num_deleted = VectorIdMapping.query.filter(
                    VectorIdMapping.id.in_(mapping_ids_to_delete)
                ).delete(synchronize_session=False)

                if num_deleted != len(mappings_to_delete):
                     logger.warning(f"Task {self.request.id}: Expected to delete {len(mappings_to_delete)} mappings from DB for DB query source '{db_query_identifier}', but deleted {num_deleted}.")

                logger.info(f"Task {self.request.id}: Successfully deleted {num_deleted} vector mapping records from database for DB query source '{db_query_identifier}'.")
                db.session.commit() # Commit DB changes (includes chatbot source_details update and VectorIdMapping deletion)
                logger.info(f"Task {self.request.id}: Successfully deleted data for source '{details_removal_identifier}' (DB query ID: '{db_query_identifier}') from chatbot {chatbot_id}.")
                return f"Success: Source data for '{details_removal_identifier}' deleted."

            except SQLAlchemyError as e:
                logger.error(f"Task {self.request.id}: Database error deleting vector mappings for DB query source '{db_query_identifier}': {e}", exc_info=True)
                db.session.rollback()
                raise self.retry(exc=e, countdown=30) # Retry DB errors

        except SQLAlchemyError as e: # Catch DB errors during initial query
            logger.error(f"Task {self.request.id}: Database error finding vector mappings for DB query source '{db_query_identifier}': {e}", exc_info=True)
            db.session.rollback()
            raise self.retry(exc=e, countdown=30)
        except Exception as e:
            logger.error(f"Task {self.request.id}: Unexpected error during source data deletion for '{details_removal_identifier}' (DB query ID: '{db_query_identifier}'): {e}", exc_info=True)
            db.session.rollback()
            raise self.retry(exc=e)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=120)
def delete_chatbot_data_task(self, chatbot_id: int, user_id: int):
    """
    Celery task to delete ALL data associated with a specific chatbot.
    Does NOT delete the Chatbot record itself.
    """
    logger.info(f"Task {self.request.id}: Starting deletion for ALL data for chatbot {chatbot_id} (triggered by user {user_id}).")
    with flask_app.app_context():
        rag_service = None
        try:
            rag_service = get_rag_service()
            if rag_service is None:
                 logger.error(f"Task {self.request.id}: get_rag_service() returned None unexpectedly.")
                 raise self.retry(exc=Exception("Failed to obtain RagService instance via get_rag_service()."))

            # Verify ownership
            chatbot = db.session.get(Chatbot, chatbot_id)
            if not chatbot:
                 logger.warning(f"Task {self.request.id}: Chatbot {chatbot_id} not found.")
                 return f"Success: Chatbot {chatbot_id} not found, no data to delete."
            if chatbot.user_id != user_id:
                 logger.error(f"Task {self.request.id}: User {user_id} does not own chatbot {chatbot_id}. Deletion forbidden.")
                 return f"Failed: Permission denied."

            # 1. Identify All Vector Data in Database
            # Fetch mappings FIRST to use for both GCS and Vector deletion
            mappings = VectorIdMapping.query.filter_by(chatbot_id=chatbot_id).all()
            vector_ids_to_delete = [mapping.vector_id for mapping in mappings]
            logger.info(f"Task {self.request.id}: Found {len(vector_ids_to_delete)} vector IDs to delete for chatbot {chatbot_id}.")

            # --- 1b. Identify and Delete Associated GCS Files ---
            gcs_paths_to_delete = set() # Use a set to avoid duplicates
            for mapping in mappings:
                # Pass chatbot_id to the helper function
                gcs_path = _construct_gcs_path_from_vector_id(mapping.vector_id, chatbot_id)
                if gcs_path:
                    gcs_paths_to_delete.add(gcs_path)
                # else: # Already logged in helper function
                #    logger.warning(f"Task {self.request.id}: Could not construct GCS path for vector ID {mapping.vector_id}")

            logger.info(f"Task {self.request.id}: Identified {len(gcs_paths_to_delete)} unique GCS paths to attempt deletion for.")

            gcs_init_success = False
            if rag_service:
                try:
                    logger.info(f"Task {self.request.id}: Explicitly calling rag_service._ensure_clients_initialized() before GCS deletion...")
                    if rag_service._ensure_clients_initialized():
                         gcs_init_success = True
                         logger.info(f"Task {self.request.id}: rag_service._ensure_clients_initialized() call reported success.")
                    else:
                         logger.error(f"Task {self.request.id}: rag_service._ensure_clients_initialized() call reported failure. Initialization error: {rag_service.initialization_error}")
                except Exception as init_ex:
                    logger.error(f"Task {self.request.id}: Exception during explicit call to rag_service._ensure_clients_initialized(): {init_ex}", exc_info=True)
            else:
                 logger.error(f"Task {self.request.id}: Cannot attempt GCS initialization because rag_service is None.")

            # --- Attempt GCS Deletion ---
            deleted_files_count = 0
            failed_files_count = 0
            if gcs_init_success and rag_service.bucket:
                bucket_name = rag_service.bucket.name
                logger.info(f"Task {self.request.id}: RAG service bucket '{bucket_name}' appears initialized. Attempting to delete associated chunk files...")

                if not gcs_paths_to_delete:
                    logger.info(f"Task {self.request.id}: No GCS paths derived from vector IDs, skipping GCS deletion.")
                else:
                    logger.info(f"Task {self.request.id}: Attempting deletion for {len(gcs_paths_to_delete)} GCS paths from bucket '{bucket_name}'...")
                    for gcs_path in gcs_paths_to_delete:
                        try:
                            blob = rag_service.bucket.blob(gcs_path) # Use the constructed path
                            if blob.exists():
                                blob.delete()
                                deleted_files_count += 1
                                logger.info(f"Task {self.request.id}: Successfully deleted GCS object: {gcs_path}")
                            else:
                                logger.warning(f"Task {self.request.id}: GCS object not found (already deleted?): {gcs_path}")
                                deleted_files_count += 1 # Treat as success
                        except GoogleNotFound:
                             logger.warning(f"Task {self.request.id}: GCS object not found (NotFound Exception): {gcs_path}")
                             deleted_files_count += 1
                        except Exception as e:
                            failed_files_count += 1
                            logger.error(f"Task {self.request.id}: Failed to delete GCS object '{gcs_path}': {e}", exc_info=True)

                    logger.info(f"Task {self.request.id}: GCS deletion attempt summary - Successfully Deleted/Not Found: {deleted_files_count}, Failed: {failed_files_count}")
                    if failed_files_count > 0:
                        logger.warning(f"Task {self.request.id}: Proceeding with vector/DB deletion despite {failed_files_count} GCS deletion failures.")
            else:
                logger.error(f"Task {self.request.id}: CRITICAL - Cannot delete GCS files. Bucket initialization failed or rag_service is None.")
                # Decide if this should be a hard failure/retry
                # raise self.retry(exc=Exception("RAG service bucket not initialized for GCS deletion."), countdown=60)

            # --- 2. Delete from Vector Store ---
            # (Keep this section as is)
            if vector_ids_to_delete:
                try:
                    rag_service.delete_datapoints(vector_ids_to_delete)
                    logger.info(f"Task {self.request.id}: Successfully requested deletion of {len(vector_ids_to_delete)} vectors from Vertex AI for chatbot {chatbot_id}.")
                except VertexAIDeletionError as e:
                    logger.error(f"Task {self.request.id}: Failed to delete vectors from Vertex AI for chatbot {chatbot_id}: {e}")
                    db.session.rollback() # Rollback before retrying
                    raise self.retry(exc=e, countdown=60)
                except Exception as e:
                    logger.error(f"Task {self.request.id}: Unexpected error during Vertex AI deletion for chatbot {chatbot_id}: {e}", exc_info=True)
                    db.session.rollback() # Rollback before retrying
                    raise self.retry(exc=e) # Retry unexpected errors

            # --- 3. Delete from Database ---
            # (Keep this section as is, but ensure it runs AFTER vector deletion attempt)
            try:
                # Use transaction management provided by Flask-SQLAlchemy context if appropriate
                # or manage explicitly with db.session.begin(), commit(), rollback()

                # Delete DetailedFeedback
                feedback_id_select = select(DetailedFeedback.id)\
                    .join(ChatMessage, DetailedFeedback.message_id == ChatMessage.id)\
                    .where(ChatMessage.chatbot_id == chatbot_id)
                num_feedback_deleted = db.session.query(DetailedFeedback)\
                    .filter(DetailedFeedback.id.in_(feedback_id_select))\
                    .delete(synchronize_session=False)
                logger.info(f"Task {self.request.id}: Deleted {num_feedback_deleted} detailed feedback records for chatbot {chatbot_id}.")

                # Delete ChatMessages
                num_messages_deleted = ChatMessage.query.filter_by(chatbot_id=chatbot_id).delete(synchronize_session=False)
                logger.info(f"Task {self.request.id}: Deleted {num_messages_deleted} chat message records for chatbot {chatbot_id}.")

                # Delete UsageLogs
                num_logs_deleted = UsageLog.query.filter_by(chatbot_id=chatbot_id).delete(synchronize_session=False)
                logger.info(f"Task {self.request.id}: Deleted {num_logs_deleted} usage log records for chatbot {chatbot_id}.")

                # Delete VectorIdMappings - Use the IDs from the mappings list fetched earlier
                if mappings: # Only delete if mappings were found
                    mapping_ids_to_delete = [m.id for m in mappings]
                    num_mappings_deleted = VectorIdMapping.query.filter(
                        VectorIdMapping.id.in_(mapping_ids_to_delete)
                    ).delete(synchronize_session=False)
                    logger.info(f"Task {self.request.id}: Deleted {num_mappings_deleted} vector mapping records for chatbot {chatbot_id}.")
                else:
                    logger.info(f"Task {self.request.id}: No vector mapping records to delete from DB for chatbot {chatbot_id}.")


                # Delete the main Chatbot record itself
                if chatbot: # Check if chatbot object still exists
                    db.session.delete(chatbot)
                    logger.info(f"Task {self.request.id}: Marked main Chatbot record {chatbot_id} for deletion from database.")
                else:
                    logger.warning(f"Task {self.request.id}: Chatbot object {chatbot_id} was not found when attempting to mark for main record deletion.")


                db.session.commit() # Commit all DB changes together (including chatbot deletion)
                logger.info(f"Task {self.request.id}: Successfully deleted all associated data AND the main record for chatbot {chatbot_id}.")
                return f"Success: All associated data AND the main record for chatbot {chatbot_id} deleted."

            except SQLAlchemyError as e:
                logger.error(f"Task {self.request.id}: Database error during cleanup for chatbot {chatbot_id}: {e}", exc_info=True)
                db.session.rollback()
                raise self.retry(exc=e, countdown=30)

        except Exception as e:
            logger.error(f"Task {self.request.id}: Unexpected error during chatbot data deletion for {chatbot_id}: {e}", exc_info=True)
            # Rollback any potential uncommitted changes
            db.session.rollback()
            raise self.retry(exc=e) # Retry generic unexpected errors
