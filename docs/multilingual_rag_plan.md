# Multilingual RAG Pipeline Enhancement Plan (Refined)

**Date:** 2025-04-17

**Status:** Approved

## 1. Goal

Enable the RAG pipeline to handle user queries in languages different from the source documents, ensuring accurate retrieval and responses in the user's original language.

## 2. Core Strategy

1.  **Store Source Language:** Store the primary language of source documents associated with each chatbot.
2.  **Detect Query Language:** Detect the language of incoming user queries.
3.  **Translate Query:** Translate the query to the source document language *before* embedding retrieval if languages differ.
4.  **Instruct LLM:** Instruct the LLM to generate the final response in the *original* query language, using the retrieved source-language context.

## 3. Prerequisites & Setup

*   **Choose & Configure Language Service:**
    *   **Recommendation:** Google Cloud Translation API (`google-cloud-translate`).
    *   **Action:** Add `google-cloud-translate` to `chatbot-backend/requirements.txt`.
    *   **Action:** Obtain and securely store API credentials (e.g., environment variables).
*   **Install Dependencies:** Run `pip install -r chatbot-backend/requirements.txt`.

## 4. Database Schema Modification

*   **File:** `chatbot-backend/app/models.py`
*   **Model:** `Chatbot`
*   **Change:** Add a new field:
    ```python
    source_document_language = db.Column(db.String(10), nullable=True, default='en') # Language of the ingested RAG source documents (ISO 639-1)
    ```
    *(Ensure this is distinct from the existing `text_language` field which likely controls UI/interaction language).*
*   **Migration:**
    *   Generate migration: `flask db migrate -m "Add source_document_language to Chatbot"`
    *   Apply migration: `flask db upgrade`

## 5. Backend API Modification

*   **File(s):** Relevant route files in `chatbot-backend/app/routes/` (e.g., `chatbot_routes.py`).
*   **Endpoints:**
    *   **Get Chatbot Details:** Modify the endpoint that returns chatbot details (used by frontend's `apiService.getChatbotDetails`) to include the new `source_document_language` field in its response.
    *   **Update Chatbot Settings:** Modify the endpoint that saves chatbot settings (used by frontend's `DataManagementView` or related save actions) to:
        *   Accept `source_document_language` in the request payload.
        *   Validate the incoming language code (optional but recommended).
        *   Save the value to the `Chatbot.source_document_language` field in the database.

## 6. Frontend UI Modification

*   **File:** `chatbot-frontend/src/components/DataManagementView.jsx`
*   **Changes:**
    *   **Fetch Data:** Ensure the `fetchSources` function (or equivalent data fetching logic) retrieves and stores the `source_document_language` from the API response into the component's state.
    *   **State:** Add state variable (e.g., `sourceDocumentLanguage`, `setSourceDocumentLanguage`) initialized with the fetched value.
    *   **UI Element:** Add a labeled `<select>` dropdown element within the component's returned JSX.
        *   Label: "Source Document Language" (or similar).
        *   Options: Populate with common languages and their ISO 639-1 codes (e.g., `{ value: 'en', label: 'English' }`, `{ value: 'es', label: 'Spanish' }`, `{ value: 'fr', label: 'French' }`, etc.).
        *   Value: Bind the dropdown's value to the `sourceDocumentLanguage` state variable.
        *   `onChange`: Set the handler to update the `sourceDocumentLanguage` state using `setSourceDocumentLanguage`.
    *   **Save Logic:** Modify the function responsible for saving chatbot settings (e.g., triggered by a save button or passed down from `EditChatbotPage`) to include the current value of the `sourceDocumentLanguage` state variable in the data payload sent to the backend update API endpoint.

## 7. Backend RAG Service Modification

*   **File:** `chatbot-backend/app/services/rag_service.py`
*   **Class:** `RagService`
*   **Method:** `execute_pipeline`
*   **Detailed Steps within `execute_pipeline`:**
    1.  **Import & Init:**
        *   `from google.cloud import translate_v2 as translate`
        *   Instantiate `translate_client = translate.Client()` (consider doing this in `_initialize_clients` for efficiency).
    2.  **Retrieve Source Language:**
        *   Fetch the `Chatbot` object using `chatbot_id`.
        *   `source_lang = chatbot.source_document_language if chatbot and chatbot.source_document_language else 'en'`
        *   `self.logger.info(f"Chatbot source document language: {source_lang}")`
    3.  **Detect Query Language:**
        *   `detection_result = translate_client.detect_language(query)`
        *   `detected_lang = detection_result['language']`
        *   `confidence = detection_result.get('confidence', 'N/A')`
        *   `self.logger.info(f"Detected query language: {detected_lang} (Confidence: {confidence})")`
    4.  **Translate Query (Conditional):**
        *   `translated_query = query`
        *   `translation_performed = False`
        *   `if detected_lang != source_lang and detected_lang != 'und':`
            *   `self.logger.info(f"Translating query from '{detected_lang}' to '{source_lang}'...")`
            *   `try:`
                *   `translation_result = translate_client.translate(query, target_language=source_lang)`
                *   `translated_query = translation_result['translatedText']`
                *   `translation_performed = True`
                *   `self.logger.info(f"Translation successful. Query for RAG: '{translated_query[:50]}...'")`
            *   `except Exception as e:`
                *   `self.logger.error(f"Translation failed: {e}. Using original query for RAG.", exc_info=True)`
                *   `# Fallback to original query`
        *   `else:`
            *   `self.logger.info("Query language matches source or is undetermined. No translation needed.")`
    5.  **Generate Rephrased Queries:**
        *   Pass `translated_query` to `self.generate_rephrased_queries()`.
    6.  **Embeddings & Retrieval:**
        *   The existing logic for `generate_multiple_embeddings` and `retrieve_chunks_multi_query` will now use embeddings derived from `translated_query` (or its rephrasings).
    7.  **Construct Prompt:**
        *   When calling `self.construct_prompt(...)`, ensure the `query_language` parameter is set to the *original* `detected_lang`.
        *   Example: `prompt_parts = self.construct_prompt(..., query=query, query_language=detected_lang, ...)`
    8.  **Generate Response:**
        *   Call `self.generate_response(...)` with the constructed prompt.

## 8. Logging Implementation

*   **Location:** Primarily within `rag_service.py` (`execute_pipeline`).
*   **Key Log Points:**
    *   Start of pipeline execution.
    *   Retrieved `source_document_language`.
    *   Detected query language and confidence.
    *   Decision on whether translation is needed.
    *   Result of translation attempt (success/failure, translated query snippet).
    *   Query being used for rephrasing/embedding (original or translated).
    *   Language being passed to `construct_prompt` (should be original detected language).
    *   End of pipeline execution.
*   **Tool:** Use Python's standard `logging` module, configured within the Flask application.

## 9. Testing Strategy

*   **Unit Tests:**
    *   `rag_service.py`: Test language detection mocking, translation mocking, conditional logic, correct language propagation to `construct_prompt`.
    *   API Endpoints: Test update endpoint saves `source_document_language`, detail endpoint returns it.
    *   Frontend (`DataManagementView.jsx`): Test state updates, dropdown rendering, correct value included in save payload. Mock `apiService`.
*   **Integration Tests:**
    *   Create/edit a chatbot, setting `source_document_language` (e.g., 'es').
    *   Ingest Spanish documents.
    *   Query in English: Verify query is translated to Spanish for RAG, LLM prompt requests English response, final response is in English.
    *   Query in Spanish: Verify no translation occurs, RAG uses Spanish query, LLM prompt requests Spanish response, final response is in Spanish.
    *   Query in French: Verify query is translated to Spanish for RAG, LLM prompt requests French response, final response is in French.
    *   Test edge cases (undetermined language, translation API errors).

## 10. Deployment & Operational Considerations

*   **API Keys:** Use secure methods (environment variables, secrets manager) for the Translation API key.
*   **Cost:** Monitor Google Cloud Translation API usage and costs. Set budget alerts if necessary.
*   **Latency:** Translation adds latency; monitor overall response time.
*   **Error Handling:** Implement robust try/except blocks around API calls for detection and translation. Define fallback behavior (e.g., use original query if translation fails).

## 11. Edge Cases & Challenges

*   **Translation Accuracy:** Imperfect translations might affect retrieval.
*   **Detection Confidence:** Decide how to handle low-confidence detections.
*   **Undetermined Language ('und'):** Decide behavior (e.g., proceed without translation).
*   **Unsupported Languages:** Handle errors if the translation service doesn't support a detected language.
*   **Mixed-Language Sources:** This plan assumes a single primary source language. Handling mixed sources requires more complex chunk-level detection.

## 12. Visualization (Mermaid Diagram)

```mermaid
graph TD
    A[User Query (Original Language)] --> B{Detect Query Language};
    B -- Language Detected --> C{Retrieve Chatbot Source Language};
    C --> D{Languages Match?};
    D -- Yes --> E[Query for RAG = Original Query];
    D -- No --> F{Translate Query to Source Language};
    F -- Translated Query --> G[Query for RAG = Translated Query];
    E --> H[Generate Rephrased Queries];
    G --> H;
    H --> I[Generate Embeddings];
    I --> J[Retrieve Document Chunks (Source Language)];
    J --> K{Construct LLM Prompt};
    K -- Include Chunks (Source Lang) & Instruct Response in Original Detected Lang --> L[LLM Generates Response];
    L -- Response (Original Query Language) --> M[Return Response to User];

    subgraph Logging Points
        L1[Log Query Received]
        L2[Log Detected Query Lang]
        L3[Log Source Lang]
        L4[Log Translation Decision/Result]
        L5[Log Query Used for RAG]
        L6[Log LLM Prompt Details (incl. target lang)]
        L7[Log LLM Response Received]
    end

    A --> L1;
    B --> L2;
    C --> L3;
    D --> L4;
    F --> L4;
    E --> L5;
    G --> L5;
    K --> L6;
    L --> L7;

    style F fill:#f9f,stroke:#333,stroke-width:2px
    style K fill:#ccf,stroke:#333,stroke-width:2px