# Speech-to-Text (STT) Service Documentation

This document details the implementation of the Speech-to-Text (STT) service, responsible for transcribing audio input and processing it with an AI service.

## 1. Overview

The `STTService` handles the workflow of:
1.  Receiving raw audio data (as a Buffer).
2.  Uploading the audio to Fal AI's storage.
3.  Calling the Fal AI Wizper model to transcribe the audio.
4.  Fetching an appropriate AI service instance (OpenAI or Gemini) via `AiServiceFactory`.
5.  Sending the transcription text to the AI service for analysis (specifically using the `debugSolution` method with context type 'interview').
6.  Returning the AI's analysis.

## 2. File Location

The core logic resides in:
`electron/stt/STTService.ts`

## 3. Core Class: `STTService`

```typescript
export class STTService {
  // ... implementation ...
}
```

### 3.1 Key Method: `transcribeAndProcess`

This is the primary method exposed by the service.

**Signature:**

```typescript
async transcribeAndProcess(audioBuffer: Buffer): Promise<string>
```

**Parameters:**

*   `audioBuffer: Buffer`: A Node.js Buffer containing the raw audio data to be transcribed.

**Returns:**

*   `Promise<string>`: A promise that resolves to a string containing the analysis provided by the configured AI service after processing the transcription.

**Throws:**

*   `Error`:
    *   If the `FAL_KEY` environment variable is not set.
    *   If uploading the audio buffer to Fal storage fails.
    *   If the Fal Wizper API call fails or returns an invalid format.
    *   If the AI service analysis fails for a non-API specific reason.
*   `AiApiException`: If the call to the AI service's `debugSolution` method fails (see `electron/ai/types.ts`).

**Workflow:**

1.  Retrieves the Fal AI API key using `configService.getFalApiKey()`. Throws an error if not found.
2.  Creates a `File` object from the input `audioBuffer`.
3.  Uploads the audio file to Fal storage using `fal.storage.upload(audioFile as any)`. Stores the returned URL. Throws an error on failure.
4.  Calls the Fal Wizper transcription service using `fal.run('fal-ai/wizper', ...)` with the uploaded audio URL. Specifies `task: 'transcribe'` and `language: 'en'`. Throws an error on failure or invalid response format.
5.  Retrieves the configured AI service provider instance (OpenAI or Gemini) using `AiServiceFactory.getServiceProvider()`.
6.  Calls the `debugSolution` method on the AI service instance, passing the transcription text as `contextData` and specifying `contextType: 'interview'`.
7.  Returns the `analysis` field from the `DebugInfo` object returned by `debugSolution`.
8.  Catches potential errors during AI processing, including specific `AiApiException` instances, and throws appropriate errors.

## 4. Dependencies and Interactions

### 4.1 External Libraries

*   `@fal-ai/client`: Used for interacting with Fal AI services (storage upload and Wizper transcription).
    *   Import: `import { fal } from '@fal-ai/client';`
*   `node:buffer`: Used for handling binary audio data.
    *   Import: `import { Buffer, File } from 'node:buffer';`

### 4.2 Internal Modules

*   **`electron/configService.ts`**:
    *   Import: `import { configService } from '../configService';`
    *   Usage: Calls `configService.getFalApiKey()` to retrieve the Fal API key from environment variables.
*   **`electron/ai/aiServiceFactory.ts`**:
    *   Import: `import { AiServiceFactory } from '../ai/aiServiceFactory';`
    *   Usage: Calls `AiServiceFactory.getServiceProvider()` to get an instance of the currently configured AI service (OpenAI or Gemini).
*   **`electron/ai/types.ts`**:
    *   Import: `import { IAiServiceProvider, AiApiException, DebugInfo } from '../ai/types';`
    *   Usage: Uses the `IAiServiceProvider` interface contract (specifically the `debugSolution` method), the `DebugInfo` return type, and the `AiApiException` for error handling.
*   **`electron/store.ts`** (Indirect via `AiServiceFactory`):
    *   The `AiServiceFactory` uses `getCurrentAiProvider()` from the store to determine whether to instantiate `OpenAIClient` or `GeminiClient`.

## 5. Configuration

The `STTService` relies on several configuration elements:

1.  **Fal AI API Key**:
    *   Must be set as an environment variable `FAL_KEY`.
    *   Accessed via `configService.getFalApiKey()`.
    *   **Setup**: Add `FAL_KEY=YOUR_FAL_API_KEY` to your `.env` file.
2.  **AI Service Provider**:
    *   Determined by the value stored in `electron-store` under the key `aiProvider` (either `"openai"` or `"gemini"`).
    *   Read by `AiServiceFactory` when `getServiceProvider()` is called.
3.  **AI Service API Keys**:
    *   The `AiServiceFactory` reads the corresponding API key from environment variables:
        *   `OPENAI_API_KEY` for the OpenAI provider.
        *   `GEMINI_API_KEY` for the Gemini provider.
    *   **Setup**: Add `OPENAI_API_KEY=YOUR_OPENAI_KEY` and/or `GEMINI_API_KEY=YOUR_GEMINI_KEY` to your `.env` file, depending on the provider(s) you intend to use.

## 6. Setup Summary

To ensure the STTService functions correctly:

1.  Install dependencies: `npm install @fal-ai/client` (or `bun install`).
2.  Create or update the `.env` file in the project root with:
    ```dotenv
    FAL_KEY=YOUR_FAL_API_KEY
    OPENAI_API_KEY=YOUR_OPENAI_KEY # If using OpenAI
    GEMINI_API_KEY=YOUR_GEMINI_KEY   # If using Gemini
    ```
    *(Replace placeholders with actual keys)*
3.  Ensure the desired AI provider (`"openai"` or `"gemini"`) is correctly set in the application's configuration store (managed by `electron/store.ts` and potentially `configService.ts`).