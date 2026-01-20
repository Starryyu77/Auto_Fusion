# Dataset & Dummy Data Tools Implementation Plan

I will implement the `ProxyFeatureDataset` and the dummy data generator tool to support the proxy task training in `Evaluator`.

## 1. `src/dataset.py` (Proxy Feature Dataset)
I will create a class `ProxyFeatureDataset` that inherits from `torch.utils.data.Dataset`.
-   **Functionality**: Loads pre-computed feature tensors from disk (`.pt` files).
-   **Features**:
    -   Loads `train_vision.pt` (Visual Features), `train_text.pt` (Text Features), and `train_labels.pt`.
    -   `__getitem__` returns a dictionary: `{'visual': ..., 'text': ..., 'label': ...}`.
    -   Includes a helper function `create_dataloaders` to return Train/Val DataLoaders.
    -   **Optimization**: Uses `mmap=True` for `torch.load` if supported/beneficial, or loads entirely into RAM if size permits (controlled by a flag).
    -   **MPS Compatibility**: Data remains on CPU during loading; transfer to MPS happens in the training loop (not here).

## 2. `tools/make_dummy_features.py` (Dummy Data Generator)
I will create a standalone script to generate fake feature data for testing.
-   **Logic**:
    -   Generates random tensors matching the dimensions required by `UnifiedFusionAdapter` and T5.
    -   **Visual**: `[Batch, Seq_Len_V, Dim_V]` (e.g., `[100, 49, 2048]`). Note: `adapter.py` defaults to `vis_dim=2048`.
    -   **Text**: `[Batch, Seq_Len_T, Dim_T]` (e.g., `[100, 512, 768]`).
    -   **Labels**: Random integer labels or embedding targets.
    -   Saves these as `.pt` files in a created `data/processed/` directory.
-   **Usage**: Run this once to populate `data/processed/` so `Evaluator` has something to load.

## 3. Update Project Status
I will update `.ai_status.md` to reflect the addition of data handling modules.
