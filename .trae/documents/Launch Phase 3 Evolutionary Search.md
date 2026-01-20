# Phase 3 Full-Scale Search Launch Plan

I will launch the full-scale evolutionary search for Auto-Fusion, transitioning from testing to production operation.

## 1. Verify Environment & Data
I will perform a final check to ensure:
-   `GOOGLE_API_KEY` is available (I will prompt the user to provide it or verify it's set).
-   Feature data exists in `data/processed/scienceqa_features`.
-   `src/search_runner.py` is ready.

## 2. Execute Search Command
I will construct and execute the command to run the search process in the background using `nohup`.
-   **Iterations**: 20
-   **Epochs**: 3 (per user instruction in the prompt example, though input requirements said 5, I'll stick to the explicit command example of 3 unless directed otherwise. *Correction*: Input requirements said "Epochs: 5", but the example code showed 3. I will prioritize the explicit requirement of **5 epochs** for better accuracy).
-   **Log**: Redirect output to `search.log`.

## 3. Provide Monitoring Instructions
After launching, I will provide the user with the command to monitor the progress (`tail -f search.log`).

## 4. Update Project Status
I will update `.ai_status.md` to mark the start of Phase 3.
