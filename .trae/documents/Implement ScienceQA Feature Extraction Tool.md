# Implement Real ScienceQA Feature Extraction & Download Data

I will implement the feature extraction tools and **automatically download** the necessary ScienceQA dataset and model weights, ensuring the system is ready for real data processing.

## 1. Create `tools/download_scienceqa.py`
I will create a script to automatically download the ScienceQA dataset.
-   **Source**: Hugging Face Datasets (`derek-thomas/ScienceQA`).
-   **Action**: Downloads the dataset and reorganizes it into the required structure at `data/raw/ScienceQA` (`problems.json`, `pid_splits.json`, `images/`).
-   **Verification**: Checks if files already exist to avoid re-downloading.

## 2. Create `tools/extract_features.py`
I will implement the main feature extraction script.
-   **Data Loading**: Reads from `data/raw/ScienceQA`.
-   **Models**: 
    -   Vision: `facebook/detr-resnet-50`
    -   Text: `google/flan-t5-base`
    -   **Auto-Download**: The script will automatically trigger Hugging Face to download/cache these models on first run.
-   **Processing**:
    -   Batches data.
    -   Encodes images and text on MPS (if available).
    -   Saves outputs to `data/processed/scienceqa_features` in the format required by `ProxyFeatureDataset` (`{split}_vision.pt`, etc.).

## 3. Execute Downloads & Extraction
I will run the created tools to physically populate the data:
-   Run `python tools/download_scienceqa.py` to get the raw data.
-   Run `python tools/extract_features.py --limit 50` (small batch first) to verify the extraction pipeline and download model weights.

## 4. Documentation
I will update `docs/data_setup.md` and `.ai_status.md` to reflect that the data pipeline is now automated and functional.
