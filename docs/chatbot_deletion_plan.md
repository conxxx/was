# Chatbot Deletion Implementation Plan

This document outlines the plan for implementing the chatbot deletion feature, covering frontend, backend API, data cleanup in Vertex AI and the database, error handling, and logging.

## 1. Frontend (`chatbot-frontend/src/pages/DashboardPage.jsx` or similar)

*   **Trigger:** Add a "Delete" button next to each chatbot listed on the dashboard.
*   **Confirmation:**
    *   Clicking the "Delete" button should trigger a confirmation modal (e.g., using a library like `react-modal` or a custom component).
    *   The modal should clearly state the irreversible nature of the action (e.g., "Are you sure you want to delete chatbot '[Chatbot Name]'? This action cannot be undone.").
    *   Include "Confirm Delete" and "Cancel" buttons.
*   **API Call:**
    *   Upon confirmation, the frontend will send an asynchronous `DELETE` request to the backend API endpoint: `/api/chatbots/<chatbot_id>`.
    *   The `<chatbot_id>` should be obtained from the chatbot data associated with the clicked button.
    *   Include necessary authentication tokens (e.g., JWT in the `Authorization` header).
*   **Response Handling:**
    *   **Success (e.g., 200 OK or 204 No Content):**
        *   Remove the corresponding chatbot entry from the UI list.
        *   Display a temporary success notification (e.g., "Chatbot '[Chatbot Name]' deleted successfully.").
    *   **Error (e.g., 4xx, 5xx):**
        *   Keep the chatbot entry in the UI.
        *   Display an error message to the user (e.g., "Failed to delete chatbot. Please try again later or contact support."). Log the detailed error received from the backend for debugging.

## 2. Backend API (`DELETE /api/chatbots/<chatbot_id>` in `chatbot-backend/app/api/routes.py`)

*   **Authentication & Authorization:**
    *   Verify the user is authenticated (e.g., using Flask-Login or JWT).
    *   Verify the authenticated user is the owner of the chatbot identified by `<chatbot_id>`. Query the `Chatbot` table and check the `user_id`. Return 403 Forbidden if not authorized.
*   **Validation:**
    *   Validate that `<chatbot_id>` is a valid identifier format (e.g., integer or UUID).
    *   Check if a chatbot with the given `<chatbot_id>` exists. Return 404 Not Found if it doesn't.

## 3. Backend Deletion Logic (Service Layer: e.g., `chatbot-backend/app/services/chatbot_service.py`)

This logic should be encapsulated in a dedicated function (e.g., `delete_chatbot_data`) called by the API route handler.

*   **Step 1: Retrieve Data:**
    *   Fetch the `Chatbot` object from the database using `chatbot_id`.
    *   Query the `VectorIdMapping` table to retrieve all `vector_id`s associated with the `chatbot_id`.
    *   If no `vector_id`s are found, proceed directly to Step 3 (Database Cleanup).
*   **Step 2: Vertex AI Cleanup:**
    *   Initialize the Vertex AI client (`aiplatform.init(...)`).
    *   Get the `IndexEndpoint` reference using the configured `INDEX_ENDPOINT_ID` and `PROJECT_ID`, `REGION`.
    *   Iterate through the list of `vector_id`s retrieved in Step 1.
    *   For each `vector_id`:
        *   Call the `index_endpoint.delete_datapoints(datapoint_ids=[vector_id])` method (or the equivalent method in the SDK version used).
        *   **Error Handling:**
            *   Log any errors encountered during the deletion of a specific `vector_id` (e.g., "Failed to delete vector_id '{vector_id}' for chatbot_id '{chatbot_id}': {error_message}").
            *   **Policy:** Decide on a failure policy. Recommended: Log the error and continue trying to delete the remaining `vector_id`s. Track the number of successful and failed deletions.
*   **Step 3: Database Cleanup (Conditional on Vertex AI Success):**
    *   **Condition:** Proceed only if the Vertex AI cleanup was deemed "successful". Define success criteria:
        *   *Option A (Strict):* All `vector_id`s were successfully deleted from Vertex AI.
        *   *Option B (Lenient):* A high percentage (e.g., >95%) of `vector_id`s were deleted, or no critical errors occurred during the Vertex AI API calls. (Requires careful consideration).
        *   *Recommendation:* Start with Option A (Strict) for maximum consistency. If Vertex AI cleanup failed critically (e.g., API unavailable, authentication issues, zero successful deletions), abort here, log the failure, and return a 500 error (See Section 4).
    *   **Transaction:** Use a database transaction (`db.session.begin()` or equivalent with the ORM/DB library):
        *   `DELETE FROM VectorIdMapping WHERE chatbot_id = :chatbot_id`
        *   `DELETE FROM DetailedFeedback WHERE chat_message_id IN (SELECT id FROM ChatMessage WHERE chatbot_id = :chatbot_id)` (Ensure correct cascading or explicit deletion order)
        *   `DELETE FROM ChatMessage WHERE chatbot_id = :chatbot_id`
        *   `DELETE FROM UsageLog WHERE chatbot_id = :chatbot_id`
        *   `DELETE FROM Chatbot WHERE id = :chatbot_id`
    *   **Commit:** If all deletions within the transaction succeed, commit the transaction (`db.session.commit()`).
    *   **Rollback:** If any error occurs during the database operations, roll back the transaction (`db.session.rollback()`) and proceed to error handling (See Section 4).
*   **Step 4: Response:**
    *   **Success:** If both Vertex AI (Step 2) and Database (Step 3) cleanups are successful, return a success response (e.g., `jsonify({"message": "Chatbot deleted successfully"}), 200` or just `"", 204`).
    *   **Failure:** If any step failed critically, return an appropriate error response (See Section 4).

## 4. Error Handling & Consistency

*   **Vertex AI Failure (Critical):**
    *   **Scenario:** Unable to connect to Vertex AI, authentication failure, or a significant number of datapoint deletions fail (based on the chosen policy in Step 2).
    *   **Action:**
        *   Log the critical error details extensively.
        *   **Do NOT proceed** to database deletion (Step 3).
        *   Return a `500 Internal Server Error` or `502 Bad Gateway` to the frontend with a generic error message (e.g., "Failed to clean up AI data. Chatbot deletion aborted.").
    *   **State:** The chatbot record and all associated database entries remain. The link to Vertex AI data might be partially broken if some deletions succeeded before the critical failure. Consider adding a status field (e.g., `deletion_status = 'cleanup_failed'`) to the `Chatbot` model to flag it for potential manual intervention or a retry mechanism.
*   **Database Transaction Failure:**
    *   **Scenario:** Vertex AI cleanup (Step 2) succeeded, but the database transaction (Step 3) failed (e.g., connection error, constraint violation, deadlock).
    *   **Action:**
        *   The transaction automatically rolls back, leaving the database state unchanged *for this attempt*.
        *   Log the critical error: "Database transaction failed for chatbot_id '{chatbot_id}' after successful Vertex AI cleanup. Inconsistent state potential." Include the specific database error.
        *   Return a `500 Internal Server Error` to the frontend.
    *   **State:** This is the most problematic state: Vertex AI data is deleted, but database records still exist.
    *   **Mitigation Strategies:**
        *   **Retry Logic:** Implement retry logic (e.g., using Celery) for the database deletion part.
        *   **Manual Cleanup Flag:** Mark the chatbot with a status like `deletion_status = 'db_cleanup_pending'`. A separate process or administrator action would be needed.
        *   **Idempotency:** Ensure the deletion logic (especially Vertex AI calls) is reasonably idempotent if retries are implemented. Deleting a non-existent datapoint in Vertex AI usually doesn't cause an error, which helps.
*   **Consistency:** The primary consistency mechanisms are:
    *   Conditional execution: Only attempting database deletion after successful Vertex AI cleanup.
    *   Database transactions: Ensuring atomicity for all related database record deletions.

## 5. Logging

Implement detailed logging throughout the process using the application's standard logging framework (e.g., Python's `logging` module).

*   `INFO`: Received request to delete chatbot_id: {chatbot_id} by user_id: {user_id}.
*   `INFO`: Attempting to retrieve vector IDs for chatbot_id: {chatbot_id}. Found {count} IDs.
*   `INFO`: Attempting to delete vector_id: {vector_id} from Vertex AI for chatbot_id: {chatbot_id}.
*   `INFO`: Successfully deleted vector_id: {vector_id} from Vertex AI.
*   `WARNING` or `ERROR`: Failed to delete vector_id: {vector_id} from Vertex AI. Reason: {error_message}.
*   `INFO`: Vertex AI cleanup completed for chatbot_id: {chatbot_id}. Success: {success_count}, Failures: {failure_count}.
*   `ERROR`: Vertex AI cleanup failed critically for chatbot_id: {chatbot_id}. Aborting deletion. Reason: {error_summary}.
*   `INFO`: Attempting database cleanup transaction for chatbot_id: {chatbot_id}.
*   `INFO`: Database transaction committed successfully for chatbot_id: {chatbot_id}.
*   `CRITICAL`: Database transaction failed and rolled back for chatbot_id: {chatbot_id}. Reason: {db_error_message}. Potential inconsistent state.
*   `INFO`: Successfully completed deletion process for chatbot_id: {chatbot_id}.
*   `ERROR`: Failed to complete deletion process for chatbot_id: {chatbot_id}. Final status: {status_message}.

## 6. Integration

*   Modify the existing `DELETE /api/chatbots/<chatbot_id>` route handler in `app/api/routes.py` to incorporate the new service layer function (`delete_chatbot_data`).
*   Ensure existing authentication, authorization, and validation logic is preserved or enhanced.
*   Test thoroughly to ensure no regressions in other API functionalities.
*   Update API documentation (e.g., Swagger/OpenAPI specs) if necessary.