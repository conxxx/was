# Project Summary: Customizable RAG Chatbot Platform (Revised and Verified)

## 1. Overview & Purpose

This project develops a platform for creating and deploying customizable chatbot widgets for websites. The core functionality is Retrieval Augmented Generation (RAG), enabling chatbots to answer user queries based on the content of a customer's website.

*   **Problem Solved:** Provides a method for websites to offer customer support and information derived directly from their existing content.
*   **Core Goal:** To deliver a platform where users can create, customize, and embed content-aware chatbots.
*   **Scope:** Includes a backend system for data ingestion and RAG processing, a frontend dashboard for chatbot management and customization, and the embeddable chatbot widget.

## 2. Platforms

The application operates across three key platforms:

*   **Web Platform (Frontend):** A web-based dashboard (`chatbot-frontend`) for administrators/users to create, configure, and manage their chatbots.
*   **Web Platform (Backend):** Server-side infrastructure (`chatbot-backend`) hosting the API, data processing, and RAG capabilities.
*   **Customer Websites (Widget):** The chatbot widget (`chatbot-widget`) is designed to be embedded into external customer websites.

## 3. Architecture & Structure

The system utilizes a three-tier architecture:

*   **Backend (`chatbot-backend`):**
    *   **API:** Exposes RESTful HTTP endpoints (verified in `chatbot-backend/app/api/routes.py`) for the frontend and widget.
    *   **Services:** Handles business logic, including data ingestion, user authentication, chatbot configuration, and RAG processing (e.g., `RagService`, `AdvancedRagService`, `SummarizationService`, `VoiceService` found in `chatbot-backend/app/services/`).
    *   **Tasks:** Asynchronous operations (e.g., website crawling, data embedding) are managed by Celery and Redis (confirmed by `celery` and `redis` in `chatbot-backend/requirements.txt`).
    *   **RAG Pipeline:**
        *   **Standard RAG:** Core functionality for answering questions based on ingested website data, utilizing LLMs.
        *   **Advanced RAG Pipeline:** An optional, user-enabled feature for sophisticated query processing, implemented with modular components (verified in `chatbot-backend/app/services/advanced_rag_service.py`):
            *   Query Processing (e.g., rephrasing, decomposition - `_rephrase_query`, `_decompose_query_into_sub_questions`)
            *   Retrieval (multi-step retrieval using `rag_service_instance.retrieve_chunks_multi_query`)
            *   Re-ranking (using `CrossEncoder` in `_rerank_chunks`)
            *   Context Management (e.g., formatting, compression - `_format_context`, `_compress_context`)
            *   Orchestration (handled by the `AdvancedRagProcessor.process` method).
    *   **Database:** SQLite. (Inferred from the comment "For SQLite, it's built-in." in `chatbot-backend/requirements.txt` line 5, and the presence of `app.db` files in the `chatbot-backend` directory).
    *   **Vector Store:** Implicitly required for RAG; specific implementation details are managed by the RAG services.
    *   **Cloud Storage:** Google Cloud Storage is used for file storage (confirmed by `google-cloud-storage` in `chatbot-backend/requirements.txt`).

*   **Frontend (`chatbot-frontend`):**
    *   **User Interface:** A dashboard built with React (confirmed in `chatbot-frontend/package.json`).
    *   Communicates with the backend RESTful API via HTTP requests, using Axios (confirmed in `chatbot-frontend/package.json`).

*   **Widget (`chatbot-widget`):**
    *   **Embeddable UI:** A JavaScript-based component.
    *   Communicates with the backend RAG endpoint.

## 4. Programming Languages & Frameworks/Libraries (Verified)

*   **Backend (`chatbot-backend` - from `requirements.txt`):**
    *   **Language:** Python
    *   **Web Framework:** Flask
    *   **ORM/Database:** Flask-SQLAlchemy, Flask-Migrate (SQLite is used).
    *   **API & Web:** Flask-Cors, Flask-Login, Flask-Bcrypt, Flask-Limiter, Requests.
    *   **Task Queue:** Celery, Redis.
    *   **Cloud Integration:** `google-cloud-storage`, `google-cloud-speech`, `google-cloud-texttospeech`, `google-generativeai` (for Gemini models), `google-cloud-translate`.
    *   **NLP/RAG/Text Processing:** `beautifulsoup4`, `lxml`, `python-docx`, `pdfminer.six`, `chardet`, `html5lib`, `langdetect`, `tiktoken`, `langchain-text-splitters`, `rank_bm25`, `sentence-transformers`.
    *   **Utilities:** `python-dotenv`, `uuid` (built-in), `pydub`, `Pillow`, `validators`.

*   **Frontend (`chatbot-frontend` - from `package.json`):**
    *   **Language:** JavaScript (ES Modules, specified by `"type": "module"`).
    *   **Framework/Library:** React.
    *   **Build Tool:** Vite.
    *   **Styling:** Tailwind CSS, `tailwindcss-rtl`, styled-components.
    *   **Routing:** `react-router-dom`.
    *   **HTTP Client:** Axios.
    *   **UI Components/Utils:** `@uiw/react-color-wheel`, `react-colorful`, `react-icons`, `react-markdown` (with `remark-gfm`), `react-modal`, `react-toastify`, `motion`, `uuid`.
    *   **Linting:** ESLint.

*   **Widget (`chatbot-widget` - from `package.json`):**
    *   **Language:** JavaScript.
    *   **Build Tool:** Webpack (with `css-loader`, `style-loader`, `copy-webpack-plugin`).
    *   **Markdown Processing:** `marked`.
    *   **HTML Sanitization:** `dompurify`.
    *   Custom JavaScript modules for UI (`ui.js`), state (`state.js`), API (`api.js`), and handlers (`handlers.js`) are present in `chatbot-widget/src/`.

## 5. How the Application Works & Key Features (Verified)

*   **Chatbot Creation & Management (Frontend Dashboard):**
    *   Users log in to create and manage chatbots.
    *   Configuration includes data sources (website URLs for crawling), appearance, and toggling features like Advanced RAG.

*   **Data Ingestion (Backend):**
    *   The backend crawls websites, extracts text from HTML, PDFs, DOCX files.
    *   Content is processed, chunked, and converted into vector embeddings for RAG.

*   **Chat Interaction (Widget & Backend):**
    1.  Visitor interacts with the embedded widget.
    2.  Widget captures user input (text, or voice transcribed to text via Google Cloud STT by the backend).
    3.  Query sent to the backend API (`/chatbots/<chatbot_id>/query`).
    4.  Backend RAG pipeline (standard or advanced) retrieves relevant chunks, constructs a prompt, and sends it to a Google Gemini LLM.
    5.  LLM generates an answer.
    6.  Answer returned to the widget. If voice output is enabled, text is synthesized by the backend (Google Cloud TTS) and played.

*   **Key Widget Features (Verified in `chatbot-widget/src/`):**
    *   **Voice Interaction:**
        *   Input: Uses `MediaRecorder` (e.g., `voice.js` line 64) and Voice Activity Detection (`vad-processor.js`). Audio is sent to the backend for STT.
        *   Output: Backend uses Google Cloud TTS; widget plays received audio with controls (play/pause, stop, seek bar, time display - implemented in `ui.js` and `handlers.js`).
    *   **Customizable Appearance:** Supports theming (e.g., night mode) and positioning. Preferences are saved in `localStorage` (e.g., `localStorage.getItem(darkModeKey)` in `index.js` line 97; `localStorage.setItem(darkModeKey, ...)` in `handlers.js` line 88).
    *   **Night Mode Synchronization with Host Page: NOT IMPLEMENTED.**
        *   The widget does **not** infer initial night mode from `document.body.classList.contains('dark-mode')`.
        *   The host page can**not** control the widget's night mode via `window.chatbotWidget.setNightMode()` as this function is not exposed.
    *   **Request Cancellation:** Users can cancel long-running requests. Uses `AbortController` (e.g., `handlers.js` line 141, `voice.js` line 195).
    *   **Summarization:** A feature to summarize content, invoked via an API call to the backend.
        *   *Note on LLM Token Limits for Summarization:* While LLM token limits are a general constraint, the backend's user-facing summarization service (`summarization_service.py`) currently implements generic error logging for LLM API calls, without specific handling or logging for "token limit exceeded" as a distinct managed "known issue."

*   **Frontend Chat Page (`chatbot-frontend/src/pages/ChatPage.jsx` - Verified):**
    *   Provides a full-featured chat interface within the dashboard.
    *   Includes voice playback controls similar to the widget (e.g., `playBase64Audio` function line 693, audio event handlers lines 736-810).
    *   Handles image uploads (`handleImageUploadClick` function line 1004).

## 6. Code Design & Patterns (Verified)

*   **Modular Design:**
    *   Backend: Services are modular (e.g., `AdvancedRagService`, `IngestionService`). The Advanced RAG pipeline components are distinct within `advanced_rag_service.py`.
*   **State Management:**
    *   Frontend (React): Uses React state and context.
    *   Widget: Employs a custom `state.js` for shared state across its modules.
*   **Event-Driven UI Updates:** HTML5 Audio events (`loadedmetadata`, `timeupdate`, `ended`, etc.) drive updates to playback controls in both frontend (`ChatPage.jsx` lines 736-810) and widget.
*   **Request Cancellation:** Uses `AbortController` and `AbortSignal` for cancellable `fetch`/`axios` requests (e.g., `chatbot-widget/src/handlers.js` line 141).
*   **Client-Side Persistence:** `localStorage` is used by the widget to store UI preferences like night mode (`chatbot_night_mode_<id>`) and chat history (if enabled).
*   **External Widget Control (`window.chatbotWidget`):**
    *   The global `window.chatbotWidget` namespace is initialized in `chatbot-widget/src/index.js` (line 14), providing a base for potential external interaction.
    *   Currently, specific control functions like `setNightMode` are **not implemented** on this namespace.
*   **Resource Management:** Evidence of `URL.revokeObjectURL()` and nullifying audio object references in `chatbot-frontend/src/pages/ChatPage.jsx` (e.g., lines 790, 807) to manage resources.

## 7. Workflow/Pipeline (High-Level - Verified)

*   **User Onboarding & Chatbot Setup (Frontend Dashboard):**
    1.  User logs in.
    2.  User creates and configures a chatbot (data sources, appearance, features).
    3.  Backend (via Celery tasks) ingests and processes data from specified sources.

*   **End-User Chat Interaction (Widget on Customer Website):**
    1.  Visitor interacts with the widget (text/voice).
    2.  Widget sends query to the backend RESTful API endpoint for the specific chatbot.
    3.  Backend RAG pipeline processes query, retrieves context, generates response via LLM (Google Gemini).
    4.  Backend returns response; widget displays it (plays audio if TTS active).

*   **Asynchronous Backend Processes (Celery & Redis):**
    *   Data ingestion (crawling, parsing, embedding) and other long-running operations are handled by Celery tasks.