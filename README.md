# Auto-Fusion: Neural Architecture Search for Multimodal Fusion

Auto-Fusion is an autonomous system designed to discover optimal multimodal fusion architectures. By leveraging Reinforcement Learning (RL) and Large Language Models (LLM), it automatically generates, evaluates, and evolves PyTorch fusion modules. The system is optimized for efficiency, using proxy tasks and Apple Silicon (MPS) acceleration to enable rapid experimentation.

## 🌟 Highlights

- **Closed-Loop Evolution**: Seamlessly integrates an RL Controller, LLM Generator, and Proxy Evaluator.
- **Autonomous Coding**: Generates valid, executable PyTorch code for complex fusion mechanisms (e.g., Cross-Attention, Gating).
- **Efficient Evaluation**: Uses pre-computed features (DETR + T5) and a lightweight proxy task (ScienceQA) to evaluate architectures in seconds.
- **Apple Silicon Native**: Built-in support for MPS (Metal Performance Shaders) acceleration.

## 📂 Project Structure

The project is organized into core modules and experiment tracks:

### 1. [Auto-Fusion Core](./auto-fusion/README.md) (`/auto-fusion`)
The heart of the system. Contains the intelligent agents responsible for the search process.
- **Controller**: RL Agent (A2C) guiding the search direction.
- **Generator**: LLM Interface (Gemini/GPT) generating PyTorch code.
- **Evaluator**: Proxy task trainer for rapid feedback.
- **Bridge**: Prompt engine translating RL actions to natural language instructions.

### 2. [Track 1: Experiment Report](./track1/README.md) (`/track1`)
Contains artifacts and results from the first major experimental run.
- **Best Model**: The discovered Cross-Attention architecture (Gen-7).
- **Baseline**: Standard concatenation baseline for comparison.
- **Analysis**: Training logs, loss curves, and reproduction scripts.

## � Quick Start

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Starryyu77/Auto-Fusion.git
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

### Data Preparation

To run real experiments, prepare the ScienceQA dataset:

```bash
# 1. Download Raw Data
python tools/download_scienceqa.py

# 2. Extract Features (DETR + T5)
python tools/extract_features.py --batch_size 32
```

### Running the Search

Launch the evolutionary search using the `auto-fusion` core:

```bash
python auto-fusion/core/search_runner.py \
  --data_dir data/processed/scienceqa_features \
  --epochs 5 \
  --iterations 20
```

## 📊 Results Snapshot (Track 1)

Our automated search discovered a **Cross-Attention based Fusion** architecture that demonstrates superior learning dynamics compared to standard concatenation baselines.

| Model | Architecture | Characteristics |
| :--- | :--- | :--- |
| **Baseline** | Concat + MLP | Fast convergence, limited capacity. |
| **Auto-Fusion** | Cross-Attention | Better feature interaction, lower final loss. |

👉 *See full details in [Track 1 Report](./track1/README.md)*.

## 🛡️ Security Note

The system executes LLM-generated code. We employ AST-based static analysis to block dangerous imports (`os`, `sys`, `subprocess`). However, we strictly recommend running the search process in a sandboxed environment (Docker/Conda) for maximum safety.
