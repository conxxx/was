# Plan: Widget Enhancements

This document outlines the plan for implementing the "Widget Enhancements" TODO item for the chatbot project.

## Phase 1: Verification & Refinement of Existing Features

1.  **Verify Markdown Rendering:**
    *   **Action:** Manually test the widget with queries expected to return Markdown (lists, links, bold).
    *   **Check:** Ensure correct formatting in the chat window.
    *   **Inspect:** Check browser console for `marked.min.js` loading errors or `marked.parse()` errors (widget.js line 146).
    *   **Security Review:** Current `marked.setOptions({ breaks: true, sanitize: false });` (widget.js line 67) disables sanitization.
        *   **Recommendation:** Integrate DOMPurify for sanitization after parsing with Marked.
        *   **Conceptual Code:**
            ```javascript
            // Load DOMPurify (similar to loadMarked)
            // ...

            // Inside addMessage, after marked.parse:
            if (role === 'assistant' && typeof marked === 'function' && !isError) {
                try {
                    const rawHtml = marked.parse(content);
                    contentSpan.innerHTML = typeof DOMPurify === 'function' ? DOMPurify.sanitize(rawHtml) : rawHtml; // Sanitize if loaded
                } catch (e) {
                    console.error("Widget Error: Markdown parse/sanitize failed.", e);
                    contentSpan.textContent = content; // Fallback
                }
            } else {
                contentSpan.textContent = content;
            }
            ```

2.  **Verify Chatbot Name Display:**
    *   **Action:** Load a page with the widget.
    *   **Check:** Observe the header title.
    *   **Inspect:** Check network tab for `/api/chatbots/<id>/widget-config` request. Verify 200 OK response with correct `name` JSON payload. Check console for `fetchChatbotConfig` errors (widget.js lines 74-85).

## Phase 2: Implementation of Customization Options

```mermaid
graph TD
    A[User sets customization in Frontend Platform] --> B(Frontend Platform sends data to Backend API);
    B --> C{Backend API (Update Chatbot Endpoint)};
    C --> D[Save options to Chatbot Model in DB];

    subgraph Backend Changes
        D -- Stores --> E(Chatbot Model: Add color/welcome fields);
        F[DB Migration: Add new columns] -- Updates --> E;
        G[Widget Config API Endpoint] -- Reads --> E;
        C -- Updates --> E;
    end

    subgraph Widget Interaction
        H(Widget JS requests config) --> G;
        G -- Returns JSON with name, colors, welcome msg --> H;
        H --> I{Widget JS (widget.js)};
        I -- Applies --> J[Widget UI: Header Name];
        I -- Applies --> K[Widget UI: Colors via CSS Vars];
        I -- Applies --> L[Widget UI: Welcome Message];
    end

    style Backend Changes fill:#f9f,stroke:#333,stroke-width:2px
    style Widget Interaction fill:#ccf,stroke:#333,stroke-width:2px
```

**Detailed Steps:**

1.  **Backend (`chatbot-backend`):**
    *   **Modify `app/models.py`:** Add nullable fields: `widget_primary_color` (String(7)), `widget_text_color` (String(7)), `widget_welcome_message` (String(500)).
    *   **Database Migration:** Run `flask db migrate -m "Add widget customization fields"` and `flask db upgrade`.
    *   **Modify API (`app/api/routes.py`):**
        *   **Update `/chatbots/<id>/widget-config`:** Return new fields with defaults (e.g., `#007bff`, `#ffffff`, 'Hello! ...').
        *   **Update `POST /chatbots` & `PUT /chatbots/<id>`:** Accept new fields in JSON body and save to the `Chatbot` object.
    *   **(Optional) Frontend Platform (`chatbot-frontend`):** Modify `CreateChatbotPage.jsx` to include input fields for these settings.

2.  **Widget (`chatbot-widget/widget.js`):**
    *   **Update `fetchChatbotConfig` (lines 74-85):** Retrieve and store new fields (`widget_primary_color`, `widget_text_color`, `widget_welcome_message`) in the `config` object.
    *   **Update `init` (lines 348-355):** After fetching `config`, apply colors using CSS variables:
        ```javascript
        // Apply Colors
        if (config.widget_primary_color) {
            chatWindow.style.setProperty('--chatbot-primary-color', config.widget_primary_color);
            chatToggleButton.style.setProperty('--chatbot-primary-color', config.widget_primary_color);
        }
        if (config.widget_text_color) {
            chatWindow.style.setProperty('--chatbot-text-color', config.widget_text_color);
            chatToggleButton.style.setProperty('--chatbot-text-color', config.widget_text_color);
        }
        loadMessages(config); // Pass config to loadMessages
        ```
    *   **Update `loadMessages` (lines 207-213):** Modify to accept `config` and use `config.widget_welcome_message` or a default.
        ```javascript
        function loadMessages(config = {}) {
            const welcomeMessage = config.widget_welcome_message || 'Hello! How can I assist you today?';
            // ... rest of the function ...
            if (!storedHistory) {
                addMessage(welcomeMessage, 'assistant');
            }
            // ... handle errors ...
            if (messages.length === 0) { // Ensure welcome message is added if history load fails or is empty
                 addMessage(welcomeMessage, 'assistant');
            }
        }
        ```

## Phase 3: Testing

*   Test Markdown rendering thoroughly.
*   Test name display with different chatbots.
*   Test customization: Set custom values, verify widget reflects changes, verify defaults work.