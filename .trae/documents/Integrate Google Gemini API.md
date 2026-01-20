# Implement Google Gemini Integration Plan

I will upgrade the `AutoFusionGenerator` to support Google's Gemini API and provide a verification script.

## 1. Upgrade `src/generator.py`
I will modify `src/generator.py` to integrate the `google-generativeai` library.
-   **Imports**: Add `google.generativeai` and `tenacity` for retries.
-   **Initialization**: Update `__init__` to accept `api_key` (or read `GOOGLE_API_KEY` from env) and configure `genai`.
-   **API Call Implementation**: Implement `_call_real_api` method:
    -   Instantiate `genai.GenerativeModel`.
    -   Set **Safety Settings** to `BLOCK_NONE` for all categories to prevent code generation blocking.
    -   Set **Generation Config** with `temperature=0.9` for creativity.
    -   Return generated text and usage metadata.
-   **Generate Method**: Update `generate` to call `_call_real_api` when `mock_mode=False`.

## 2. Create `scripts/verify_gemini.py`
I will create a script to verify the connection to the Gemini API.
-   **Functionality**:
    -   Loads `GOOGLE_API_KEY` from environment.
    -   Initializes `AutoFusionGenerator` with `model_name='gemini-3-flash-preview'` (user specified) or default.
    -   Sends a test prompt: "Write a PyTorch specific fusion layer class."
    -   Prints the generated code and token usage.
-   **Dependencies**: Requires `google-generativeai` package (I will assume it's installed or user will install).

## 3. Update Project Status
I will update `.ai_status.md` to reflect the integration of the Gemini API.
