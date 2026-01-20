# Auto-Fusion: Neural Architecture Search for Multimodal Fusion

Auto-Fusion is an automated system for discovering optimal multimodal fusion architectures. It leverages Reinforcement Learning (RL) to guide a Large Language Model (LLM) in generating PyTorch code for fusion modules, which are then evaluated on a proxy task (ScienceQA) using Apple Silicon (MPS) acceleration.

## 🚀 Features

-   **Closed-Loop Evolution**: Integrates RL Controller, LLM Generator, and Proxy Evaluator.
-   **MPS Optimization**: Optimized for Apple Silicon (M-series chips) with `mps_patch`.
-   **Safe Code Execution**: Dynamic loading with AST-based security validation.
-   **Proxy Task Acceleration**: Pre-computed feature caching (DETR + T5) for fast evaluation.
-   **Mock & Real Modes**: Supports dry-runs without LLM costs and real training on ScienceQA.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/Auto-Fusion.git
    cd Auto-Fusion
    ```

2.  **Install dependencies**:
    ```bash
    pip install torch torchvision torchaudio transformers datasets pillow tqdm
    ```

3.  **Set PYTHONPATH**:
    ```bash
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    ```

## 📊 Data Preparation (ScienceQA)

To run real experiments, you need to prepare the ScienceQA dataset.

1.  **Download Raw Data**:
    ```bash
    python tools/download_scienceqa.py
    ```
    This fetches the dataset from Hugging Face and organizes it into `data/raw/ScienceQA`.

2.  **Extract Features**:
    ```bash
    python tools/extract_features.py --batch_size 32
    ```
    This encodes images (DETR-ResNet50) and text (Flan-T5-Base) into `.pt` tensors saved in `data/processed/scienceqa_features`.

## 🏃‍♂️ Usage

### 1. Quick Verification (Mock Mode)
Test the entire pipeline using dummy data and mock generation (no API costs).
```bash
# 1. Generate dummy data
python tools/make_dummy_features.py --out_dir data/processed_verify

# 2. Run verification script
python scripts/verify_pipeline.py
```

### 2. Evolutionary Search (Real Data)
Run the architecture search loop using extracted ScienceQA features.

```bash
python src/search_runner.py \
  --data_dir data/processed/scienceqa_features \
  --epochs 2 \
  --iterations 5 \
  --mock  # Remove this flag to use real LLM API
```

-   `--data_dir`: Path to feature tensors.
-   `--iterations`: Number of search rounds.
-   `--mock`: If set, uses a hardcoded generator (saves tokens). Remove to use GPT-4.

## 🧠 System Architecture

-   **Controller (`src/controller.py`)**: A2C Agent that proposes high-level actions (e.g., "Mutation", "Crossover").
-   **Bridge (`src/bridge.py`)**: Translates RL actions into prompt instructions.
-   **Generator (`src/generator.py`)**: LLM interface that writes PyTorch code (`FusionModule`).
-   **Evaluator (`src/evaluator.py`)**: Trains the generated module on the proxy task and returns a reward.
-   **Adapter (`src/adapter.py`)**: Aligns visual (2048-dim) and text (768-dim) features.

## 📂 Project Structure

```
Auto-Fusion/
├── src/
│   ├── search_runner.py   # Main entry point
│   ├── controller.py      # RL Agent
│   ├── generator.py       # LLM Interface
│   ├── evaluator.py       # Proxy Task Trainer
│   ├── dataset.py         # Feature Loader
│   └── adapter.py         # Dimension Projection
├── tools/
│   ├── download_scienceqa.py
│   ├── extract_features.py
│   └── make_dummy_features.py
├── scripts/
│   └── verify_pipeline.py # Integration Test
└── data/                  # Data storage
```

## 🛡️ Security Note
The `Evaluator` executes generated code. While AST validation is implemented to block malicious imports (`os`, `sys`), run this system in a sandboxed environment for maximum safety.
