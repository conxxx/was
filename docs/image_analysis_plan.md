
# Plan: Chatbot Image Analysis Feature Implementation

**Version:** 1.0
**Date:** 2025-04-12

## 1. Overview

This document outlines the plan for implementing an image analysis feature within the chatbot project. Users will be able to upload images through the chat interface (both the main frontend page and the embeddable widget), and the chatbot will leverage multimodal AI to understand the image content and incorporate it into its responses, alongside its existing knowledge base retrieved via RAG. The feature will be configurable per chatbot instance.

## 2. Goals

*   Enable image uploads in the chat interfaces (`ChatPage.jsx`, `widget.js`).
*   Implement a backend endpoint to receive images and chat context.
*   Integrate image analysis using a multimodal LLM (Gemini Vision).
*   Combine image analysis results with the existing RAG pipeline for context-aware responses.
*   Add a configuration toggle (`WidgetCustomizationSettings.jsx`) to enable/disable the feature per chatbot.
*   Implement comprehensive logging for debugging.
*   Identify and list necessary dependencies.

## 3. Frontend Changes

### 3.1. `chatbot-frontend/src/pages/ChatPage.jsx`

*   **UI:**
    *   Add a new button (e.g., paperclip icon) next to the text input and microphone button within the input form (`<form onSubmit={handleSendMessage} ...>`).
    *   This button will trigger a hidden file input element (`<input type="file" accept="image/*" ...>`).
    *   Optionally, implement drag-and-drop functionality onto the chat input area or the entire chat window.
    *   Display a thumbnail preview of the selected image near the input area before sending. Add a way to remove/cancel the selected image.
*   **State Management:**
    *   Add new state variables:
        *   `selectedImageFile`: To hold the `File` object of the image chosen by the user.
        *   `imagePreviewUrl`: To store a temporary URL for the thumbnail preview (using `URL.createObjectURL`).
        *   `isUploadingImage`: Boolean flag for loading state during image upload/processing.
*   **API Interaction:**
    *   Modify `handleSendMessage`:
        *   If `selectedImageFile` is present, instead of calling `apiService.queryChatbot`, call a *new* API service function (e.g., `apiService.queryChatbotWithImage`).
        *   This new function will need to send `FormData` containing:
            *   The text query (`input`).
            *   The image file (`selectedImageFile`).
            *   Chat history (`historyToSend`).
            *   `chatbotId`.
        *   Clear `selectedImageFile` and `imagePreviewUrl` after successful sending.
        *   Disable the text input and send button while `isUploadingImage` is true.
*   **Conditional Rendering:**
    *   The new image upload button should only be visible if `chatbotDetails?.image_analysis_enabled` is true (this field needs to be added to the backend response and fetched).

### 3.2. `chatbot-widget/widget.js`

*   **UI:**
    *   Leverage the existing `fileUploadButton` (created in `createUI`, lines 402-409).
    *   **Interaction Clarification:** When `image_analysis_enabled` is true, this button's primary function shifts to image uploads. If general file uploads are also supported by the widget, the `handleFileUploadClick` and `handleFileSelection` logic must differentiate based on context or potentially require separate UI elements. For this plan, we assume this button handles *only* images when image analysis is enabled.
    *   Modify its appearance/icon if needed (e.g., ensure it's a paperclip or image icon).
    *   Modify its `title` attribute to "Upload Image for Analysis".
    *   Ensure it's only displayed if `chatbotConfig.image_analysis_enabled === true`.
    *   Modify `handleFileUploadClick` (line 1253) to trigger a hidden file input specifically for images (`accept="image/*"`).
    *   Modify `handleFileSelection` (line 1318) to handle the selected image file. Consider adding a preview mechanism similar to `ChatPage.jsx`.
*   **State Management:**
    *   Add widget-specific state variables:
        *   `widgetSelectedImageFile`: Holds the selected `File` object.
        *   `widgetImagePreviewUrl`: For the preview.
        *   `widgetIsUploadingImage`: Loading state.
*   **API Interaction:**
    *   Modify `handleSendMessage` (line 662):
        *   If `widgetSelectedImageFile` exists, call the new backend endpoint (via a new function in the widget's internal API handling, similar to `apiService.queryChatbotWithImage`).
        *   Send `FormData` including the text input, image file, chat history (from `messages`), `chatbotId`, and `sessionId`.
        *   Clear image state after sending.
        *   Update `updateInputDisabledState` (line 620) to account for `widgetIsUploadingImage`.
*   **Configuration:**
    *   Modify `fetchChatbotConfig` (line 122) to fetch a new `image_analysis_enabled` boolean field from the `/widget-config` endpoint. Store it in `chatbotConfig`.
    *   Use `chatbotConfig.image_analysis_enabled` to conditionally display the image upload button.

### 3.3. `chatbot-frontend/src/components/WidgetCustomizationSettings.jsx`

*   **UI:**
    *   Add a new toggle switch section labeled "Enable Image Analysis".
    *   Use the same styling/structure as the "Enable Voice Interaction" or "Enable File Uploads" toggles.
*   **State Management:**
    *   Add a new state variable: `imageAnalysisEnabled`, initialized to `false`.
*   **API Interaction:**
    *   In `useEffect` (line 62), fetch the value for `image_analysis_enabled` from `apiService.getChatbotDetails` and update the state. Add `image_analysis_enabled: details.image_analysis_enabled ?? false` around line 90.
    *   In `handleSave` (line 124), append the `imageAnalysisEnabled` state to the `FormData` object: `formData.append('image_analysis_enabled', imageAnalysisEnabled);` (around line 152).

## 4. Backend Changes

### 4.1. API Endpoint

*   **New Endpoint:** Create a new route, e.g., `POST /api/chatbots/<chatbot_id>/query_with_image`.
    *   This endpoint will accept `multipart/form-data`.
    *   Expected form fields:
        *   `query` (text, optional if only image is sent)
        *   `image` (file upload)
        *   `history` (JSON string representing chat history, optional)
        *   `session_id` (string, optional, mainly for widget)
        *   `language` (string, optional, user's preferred response language)
*   **Request Handling:**
    *   Authenticate the request (e.g., using API key or user session).
    *   Validate `chatbot_id`.
    *   Retrieve the image file from the request data.
    *   Parse the `query`, `history`, `session_id`, and `language`.
    *   Fetch the chatbot's configuration settings, specifically checking if `image_analysis_enabled` is true for this `chatbot_id`. If not, return an error.
    *   Perform backend validation on the image (See Section 4.1.1).
    *   If validation passes, pass the validated data (query, image data, history, language, client_id, chatbot settings) to the `RagService`.
 
### 4.1.1. Backend Image Validation (New Section)
 
*   Before passing the image data to `RagService` or the Gemini API, the backend endpoint handler (`routes.py`) must perform validation:
    *   **File Size Check:** Verify the uploaded image size against a predefined limit (e.g., 10MB) configured in the application settings. Reject files exceeding the limit.
    *   **MIME Type Check:** Determine the actual MIME type of the uploaded file (e.g., using Python's `mimetypes` library or inspecting the request's `Content-Type` if reliable). Validate it against a list of allowed types (e.g., `image/jpeg`, `image/png`, `image/webp`, `image/gif`). Reject unsupported types.
    *   **Content Validation (Optional but Recommended):** Use a library like `Pillow` to attempt opening the image file. This helps ensure it's a valid, non-corrupted image format.
    *   Return specific error responses to the frontend upon validation failure (See Section 6.1).
### 4.2. Database Schema (`app/models.py`)

*   Modify the `Chatbot` model (or its associated settings model, e.g., `WidgetSettings` if separate):
    *   Add a new boolean column: `image_analysis_enabled` (default: `False`).
*   Run database migrations (e.g., using Flask-Migrate) to apply the schema change.

### 4.3. Image Processing/Analysis Service (`app/services/rag_service.py`)

*   **Modify `RagService.execute_pipeline`:**
    *   Add a new parameter to accept image data (e.g., `image_data: bytes = None`).
    *   Inside the pipeline:
        *   Check if `image_data` is provided and if the feature is enabled for the chatbot.
        *   **Image-Only Query Handling:** If `image_data` is present but the text `query` is empty or whitespace:
            *   Skip the RAG steps: Do not generate query embeddings, perform vector search (`find_neighbors`), or fetch context text (`fetch_chunk_texts`).
            *   Proceed directly to `generate_response`, passing the `image_data` and potentially a generic instruction or the chat `history` if available.
        *   If both `image_data` and `query` are present:
            *   Perform RAG steps as usual (vector search based on text `query`).
            *   Proceed to `generate_response` with `image_data`, `query`, `history`, and retrieved `context_texts`.
        *   If only `query` is present (no image):
            *   Perform RAG steps as usual.
            *   Proceed to `generate_response` with `query`, `history`, and `context_texts`.
*   **Modify `RagService.construct_prompt`:**
    *   Add a new parameter `image_analysis_result: str = None`.
    *   Modify the prompt structure to include the image analysis result as additional context. Example section:
      ```
      === IMAGE ANALYSIS START ===
      {image_analysis_result}
      === IMAGE ANALYSIS END ===

      === RETRIEVED CONTEXT DOCUMENTS START ===
      {context_str}
      === RETRIEVED CONTEXT DOCUMENTS END ===
      ```
*   **Modify `RagService.generate_response`:**
    *   Ensure the `GenerativeModel` used (`self.generation_model`) is a multimodal model (e.g., `gemini-2.5-flash-preview-04-17` or `gemini-2.5-flash-preview-04-17`). Update `GENERATION_MODEL_NAME` constant.
    *   Modify the call to `self.generation_model.generate_content` to potentially include the image data directly alongside the text prompt if the chosen approach involves direct multimodal input rather than just text description. This requires structuring the `contents` argument correctly according to the Vertex AI SDK for multimodal inputs.
      *   *Chosen Approach Refinement:* Pass the image data directly to the final generation model. Modify `generate_response` to accept `image_data` and construct the `contents` list with both text (the prompt) and image parts. Modify `execute_pipeline` to pass `image_data` down to `generate_response`. Remove the intermediate `image_analysis_result` from `construct_prompt` if the image is passed directly to `generate_response`.

### 4.4. Image Analysis Approach

*   **Method:** Utilize Google Cloud Vertex AI's Gemini multimodal models (e.g., Gemini 1.5 Flash/Pro).
*   **Implementation:**
    *   Within `RagService.execute_pipeline` (or potentially `generate_response`):
        *   If image data is present and the feature is enabled:
            *   Prepare the image data in a format acceptable by the Vertex AI SDK (e.g., base64 encoded string or `Part.from_data`).
            *   Pass this image data along with the constructed text prompt to the `generate_content` method of the multimodal `GenerativeModel`.
*   **Justification:**
    *   Leverages existing Vertex AI integration.
    *   Gemini models offer state-of-the-art multimodal understanding.
    *   Avoids adding separate image analysis APIs/libraries if Gemini can handle it directly within the generation step.
    *   Allows the model to directly correlate the image content with the text query and retrieved context for more nuanced responses.

### 4.5. `rag_service.py` Integration Strategy

1.  Modify `execute_pipeline` signature to accept `image_data: bytes = None`.
2.  Fetch chatbot settings within `execute_pipeline` (or ensure they are passed in) to check `image_analysis_enabled`.
3.  Modify `generate_response` signature to accept `image_data: bytes = None`.
4.  Update the `GENERATION_MODEL_NAME` constant to a multimodal Gemini model (e.g., "gemini-2.5-flash-preview-04-17"). Re-initialize `self.generation_model` accordingly.
5.  Inside `generate_response`:
    *   If `image_data` is present:
        *   **Dynamically Determine MIME Type:** Get the MIME type from the validated uploaded file data (e.g., from validation step 4.1.1 or request metadata). Do *not* hardcode it. Example: `mime_type = determine_mime_type(image_data)`
        *   Prepare the image `Part` using `Part.from_data(mime_type=dynamic_mime_type, data=image_data)`.
        *   Construct the `contents` argument for `generate_content`.
            *   If text `prompt` (from `construct_prompt` or a default for image-only) exists: `contents = [prompt, image_part]`
            *   If no text `prompt` (e.g., image-only with no default instruction): `contents = [image_part]` (Verify Gemini API supports image-only content parts). Consider adding a default prompt like "Describe this image." if needed.
    *   If `image_data` is not present, call `generate_content` with only the text prompt as before: `contents = [prompt]`
6.  Update the call to `generate_response` within `execute_pipeline` to pass `image_data`.
7.  Remove the need to add image analysis text to `construct_prompt` if the image is passed directly to `generate_response`.

## 5. Data Flow

```mermaid
sequenceDiagram
    participant User
    participant Frontend (ChatPage/Widget)
    participant Backend API
    participant RagService
    participant Gemini (Multimodal)
    participant Vector DB (Vertex ME)
    participant Text DB (GCS)

    User->>+Frontend: Selects image & types query
    Frontend->>Frontend: Show image preview
    User->>+Frontend: Clicks Send
    Frontend->>+Backend API: POST /query_with_image (query, image, history)
    Backend API->>+RagService: execute_pipeline(query, image_data, history, chatbot_id, client_id)
    RagService->>RagService: Check image_analysis_enabled
    alt Image Analysis Enabled AND Text Query Present
        RagService->>+Vector DB: find_neighbors(text_query_embedding, client_id) # RAG based on text
        Vector DB-->>-RagService: chunk_ids
        RagService->>+Text DB (GCS): fetch_chunk_texts(chunk_ids)
        Text DB (GCS)-->>-RagService: context_texts
        RagService->>RagService: construct_prompt(context_texts, query, history)
        RagService->>+Gemini (Multimodal): generate_content([prompt_text, image_data]) # Multimodal generation
        Gemini (Multimodal)-->>-RagService: response_text
    else Image Analysis Enabled AND Image ONLY (No Text Query)
        RagService->>RagService: construct_prompt(query="Describe the image.", history=history) # No RAG context
        RagService->>+Gemini (Multimodal): generate_content([prompt_text, image_data]) # Multimodal generation (Image focus)
        Gemini (Multimodal)-->>-RagService: response_text
    else Image Analysis Disabled OR No Image (Text Query Only)
        RagService->>+Vector DB: find_neighbors(text_query_embedding, client_id) # Standard RAG
        Vector DB-->>-RagService: chunk_ids
        RagService->>+Text DB (GCS): fetch_chunk_texts(chunk_ids)
        Text DB (GCS)-->>-RagService: context_texts
        RagService->>RagService: construct_prompt(context_texts, query, history)
        RagService->>+Gemini (Multimodal): generate_content([prompt_text]) # Text-only generation
        Gemini (Multimodal)-->>-RagService: response_text
    end
    RagService-->>-Backend API: result {response: response_text}
    Backend API-->>-Frontend: API Response
    Frontend->>User: Display bot response
## 6. Error Handling and Logging

### 6.1. Error Propagation Strategy

Clear communication of backend errors to the frontend is crucial for user experience. The backend API (`/query_with_image`) should return structured JSON error responses.

**Error Response Structure:**
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "User-friendly error message.",
    "details": "Optional technical details (for logging/debugging)."
  }
}
```

**Specific Error Codes & Scenarios:**

*   **`VALIDATION_IMAGE_SIZE_EXCEEDED`**: Image file size is too large (from Section 4.1.1).
    *   Message: "Image file size exceeds the limit (e.g., 10MB)."
*   **`VALIDATION_UNSUPPORTED_MIME_TYPE`**: Image format is not allowed (from Section 4.1.1).
    *   Message: "Unsupported image format. Please use JPEG, PNG, WEBP, or GIF."
*   **`VALIDATION_INVALID_IMAGE`**: File is corrupted or not a valid image (from Section 4.1.1 Pillow check).
    *   Message: "The uploaded file could not be processed as a valid image."
*   **`GEMINI_API_ERROR`**: General error communicating with the Vertex AI Gemini API.
    *   Message: "Could not analyze the image due to a temporary issue. Please try again later."
    *   Details: Include underlying API error message if available.
*   **`GEMINI_CONTENT_SAFETY_BLOCKED`**: Gemini blocked the request or response due to safety filters (prompt or image content).
    *   Message: "The request could not be completed due to content safety restrictions."
    *   Details: Include safety ratings/reasons if provided by the API.
*   **`IMAGE_ANALYSIS_DISABLED`**: User attempted image upload for a chatbot where the feature is off.
    *   Message: "Image analysis is not enabled for this chatbot."
*   **`INTERNAL_SERVER_ERROR`**: Catch-all for unexpected backend issues.
    *   Message: "An unexpected error occurred. Please try again later."

**Frontend Handling:**
*   The frontend (`ChatPage.jsx`, `widget.js`) must check the API response for this error structure.
*   If an error object is present, display the `message` to the user in the chat interface (e.g., as a bot error message).
*   Log the full error object (including `code` and `details`) to the console for debugging.
*   Reset any loading states (`isUploadingImage`).

### 6.2. Logging Strategy
Frontend (ChatPage.jsx, widget.js):
Log when the image upload button is clicked.
Log when an image is selected (include file name and size).
Log when the image upload/query starts.
Log successful API response reception.
Log any errors during file selection, preview generation, or API calls.
Backend API (routes.py or similar):
Log incoming request details (chatbot ID, presence of query/image/history).
Log check for image_analysis_enabled status.
Log successful call to RagService.
Log any request validation errors or exceptions.
Backend Service (rag_service.py):
Log entry into execute_pipeline indicating if image data is present.
Log decision points based on image_analysis_enabled.
Log start and end of image analysis step (if separated) or modification of generate_content call.
Log size of image data being processed.
Log construction of multimodal contents for Gemini.
Log any errors during image processing or interaction with the multimodal model.
Maintain existing detailed logging for RAG steps (embedding, retrieval, text fetching, prompt construction, generation).
7. Dependencies
Frontend (React - chatbot-frontend/package.json):
No major new libraries strictly required if using standard file input. Consider react-dropzone for enhanced drag-and-drop UI.
Frontend (Widget - chatbot-widget/widget.js):
No new dependencies expected.
Backend (Python - chatbot-backend/requirements.txt):
google-cloud-aiplatform: Already likely present, ensure version supports multimodal Gemini models. Verify version >= 1.38.1 (or later recommended).
google-cloud-storage: Already present.
vertexai: Already likely present, ensure up-to-date.
Pillow: **Required** for backend image validation (Section 4.1.1) to reliably check image integrity before sending to Vertex AI. Add `Pillow` to `requirements.txt`.
8. Potential Challenges
API Costs: Multimodal models like Gemini Vision can be more expensive than text-only models. Monitor usage and costs.
Latency: Image analysis adds processing time. Ensure the user experience remains acceptable. Optimize image handling and API calls.
Image Size/Formats: Define limits on upload size (e.g., via frontend checks and backend validation) and supported formats (JPEG, PNG, WEBP common). Gemini has its own limits.
Prompt Engineering: Crafting effective prompts that instruct the LLM to utilize both the image content and the retrieved text context effectively will require iteration.
Error Handling: Robust handling for failed image uploads, analysis errors, or unsupported formats is crucial.
Security: Ensure proper validation of uploaded files on the backend to prevent potential security risks (though sending directly to Vertex AI mitigates some risks compared to processing locally).
Gemini Model Version: Ensure the selected Gemini model version (gemini-2.5-flash-preview-04-17 or gemini-2.5-flash-preview-04-17-001) is available in the specified GCP region and supports the required multimodal input format.
9. Next Steps
Review and approve this plan.
Create necessary database migrations.
Implement backend API endpoint and RagService modifications.
Implement frontend changes in ChatPage.jsx, widget.js, and WidgetCustomizationSettings.jsx.
Thoroughly test the end-to-end flow with various images and queries.
Monitor performance and costs after deployment.
