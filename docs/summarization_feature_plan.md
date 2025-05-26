# Final Implementation Plan: Multi-Language Summarization Feature

**(Version 2 - Incorporating User Feedback)**

**1. Introduction**

The goal is to implement a multi-language summarization feature accessible from both the main React chat application (`ChatPage.jsx`) and the embeddable JavaScript widget (`widget.js`). Users should be able to summarize content obtained primarily via URL scraping (with configurable domain restrictions) or secondarily via pasted text. Summaries will be provided in user-selected target languages, matching the existing voice feature's language list, utilizing Cloud Translation APIs for quality. The feature will be configurable per chatbot via a toggle and domain list in the widget customization settings page.

**2. Content Sourcing Evaluation**

*   **a. Current Webpage (Widget):** **Not Recommended** (Security risks).
*   **b. URL Scraping (Backend - Manual):** **Primary Method.** Requires robust manual implementation with domain restrictions, error handling, and security checks (SSRF prevention).
*   **c. Pasted Text:** **Secondary Method.** Simple and secure fallback.
*   **d. Existing Knowledge (RAG Enhancement):** **Deferred.**

**3. Proposed Solution**

1.  **Primary Method:** Implement **Manual URL Scraping** with configurable domain whitelisting (input via textarea).
2.  **Secondary Method:** Implement **Pasted Text** summarization.
3.  **Configuration:** Add `summarization_enabled` (Boolean) and `allowed_scraping_domains` (Text) settings to the `Chatbot` model and `WidgetCustomizationSettings.jsx`.
4.  **Languages:** Support all languages defined in `ChatPage.jsx`'s `supportedLanguages` array for output, using **Cloud Translation APIs**.

**4. Detailed Implementation Steps**

```mermaid
sequenceDiagram
    participant User
    participant FE (ChatPage/Widget/Settings)
    participant BE (Flask API)
    participant SummarizationService
    participant CloudTranslationAPI
    participant ManualScraper

    %% Configuration %%
    User->>+FE: Accesses Widget Customization Settings
    FE->>+BE: GET /api/chatbots/{id} (fetch details)
    BE-->>-FE: Return chatbot details (incl. summarization_enabled, allowed_domains)
    FE->>FE: Display settings (toggle, domain list textarea)
    User->>FE: Enables Summarization, Enters Allowed Domains (in textarea)
    User->>+FE: Clicks 'Save'
    FE->>+BE: PUT /api/chatbots/{id} (FormData with summarization_enabled, allowed_domains)
    BE->>BE: Update Chatbot model in DB
    BE-->>-FE: Return success/updated details
    FE-->>-User: Show success message

    %% Summarization Flow %%
    User->>+FE: Clicks 'Summarize' button (ChatPage/Widget)
    FE->>FE: Show Summarization UI (URL/Text input, Lang select)
    User->>FE: Enters Content (URL/Text) & Target Language
    User->>+FE: Clicks 'Generate Summary'
    FE->>+BE: POST /api/chatbots/{id}/summarize (content, type, target_lang, api_key)
    BE->>BE: Fetch chatbot's allowed_scraping_domains
    alt Content Type is URL
        BE->>BE: Validate URL against allowed_scraping_domains (parsed from textarea string)
        alt Validation Fails
             BE-->>-FE: Return Error: Domain not allowed
        else Validation OK
            BE->>+ManualScraper: Scrape URL (requests, beautifulsoup, etc.)
            ManualScraper-->>-BE: Return Page Content (Text) / Error
        end
    end
    alt Error during scraping or validation
        BE-->>-FE: Return Error
    else Content Available (from URL or Text)
        BE->>+CloudTranslationAPI: Detect/Translate Content to Model Language (e.g., EN)
        CloudTranslationAPI-->>-BE: Translated Content
        BE->>+SummarizationService: Summarize Content (using chosen model/API)
        SummarizationService-->>-BE: Summary (in Model Language)
        BE->>+CloudTranslationAPI: Translate Summary to Target Language
        CloudTranslationAPI-->>-BE: Translated Summary
        BE-->>-FE: Return { summary: "...", original_language: "...", target_language: "..." }
        FE->>FE: Display Summary in Chat
        FE-->>-User: Shows Summary
    end

```

*   **A. Backend (Python/Flask - `chatbot-backend`)**
    1.  **New Service (`app/services/summarization_service.py`):**
        *   Class `SummarizationService`.
        *   Method `summarize(chatbot_id, content_type, content, target_language, source_language=None)`: Fetch chatbot, validate domain (parse `allowed_scraping_domains` from Text field), perform manual scraping (implement robustly), detect language, call Cloud Translation API for input translation if needed, call Summarization model/API, call Cloud Translation API for output translation, return result/error.
    2.  **API Endpoint (`app/api/routes.py`):**
        *   `POST /api/chatbots/<int:chatbot_id>/summarize`: Apply decorators, check `g.chatbot.summarization_enabled`, get params, call service, return JSON.
    3.  **Configuration (`config.py`, `.env`):** Add API keys/endpoints for Cloud Translation Service, Summarization Service (if external).
    4.  **Models (`app/models.py`):** Add `summarization_enabled = db.Column(db.Boolean, default=False, nullable=False)` and `allowed_scraping_domains = db.Column(db.Text, nullable=True)`. Run migrations.
    5.  **Update Routes (`app/api/routes.py`):** Update `get_chatbot_details`, `get_widget_config`, `update_chatbot` to handle the new fields. Ensure `update_chatbot` parses the boolean and text fields correctly from FormData.

*   **B. Frontend (React - `chatbot-frontend`)**
    1.  **Widget Settings (`src/components/WidgetCustomizationSettings.jsx`):** Add state for `summarizationEnabled`, `allowedScrapingDomains`. Fetch/set in `useEffect`. Add UI toggle and textarea. Update `handleSave` to append these fields to `FormData`.
    2.  **Chat Page (`src/pages/ChatPage.jsx`):** Conditionally render button. Implement modal with URL/Text inputs and language dropdown (using `supportedLanguages`). Call API service on submit.
    3.  **API Service (`src/services/api.js`):** Add `summarizeContent` function. Ensure `updateChatbot` handles FormData.

*   **C. Widget (Vanilla JS - `chatbot-widget`)**
    1.  **Configuration (`widget.js`):** Fetch `summarization_enabled`.
    2.  **UI (`widget.js`):** Conditionally create "Summarize" button.
    3.  **Interaction (`widget.js`):** Implement UI toggle for summarization mode.
    4.  **API Call (`widget.js`):** Call summarize endpoint via `fetch`. Handle loading/results.

**5. UI/UX Proposals**

*   **Settings Page:** Toggle switch for enable/disable. Textarea for allowed domains (comma or newline separated).
*   **Chat/Widget:** Distinct icon. Modal (ChatPage) or inline input modification (Widget). Language dropdown. Clear prefix for summary messages. Loading indicators. Specific error messages (e.g., "Domain not allowed for summarization").

**6. Technology Choices**

*   **Summarization:** Hugging Face `transformers` (e.g., `google/mt5-small`) or Cloud API (Google/AWS/Azure).
*   **Language Detection:** `langdetect` or `fasttext`.
*   **Translation:** **Cloud Translation API (e.g., Google Cloud Translate, AWS Translate, DeepL API)**.
*   **Web Scraping (Manual):** `requests`, `beautifulsoup4`, `html5lib`, `validators`, `urllib.parse`, potentially `trafilatura`. Implement carefully with error handling, timeouts, user-agent rotation, and SSRF prevention.

**7. Integration Strategy**

*   Additive feature controlled by DB flag. Requires migration. New service class. Updates to existing components/routes follow established patterns.

**8. Logging (`chatbot-backend`)**

*   Detailed logging in API route, Summarization Service, and Manual Scraper covering requests, validation, processing steps, errors, and results.

**9. Feasibility & Challenges**

*   Manual URL scraping remains the primary challenge (reliability, anti-bot measures, security). Domain whitelisting mitigates some risk.
*   Multi-language support is feasible with Cloud Translation APIs.
*   Performance needs user feedback (loading states).