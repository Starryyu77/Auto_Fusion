# Auto-Fusion: Autonomous Neural Architecture Search (NAS) Module

This module encapsulates the core intelligence of the Auto-Fusion project. It is an autonomous system designed to "re-invent" deep learning architectures through closed-loop evolutionary search, powered by Large Language Models (LLM) and Reinforcement Learning (RL).

## 1. System Architecture

The system operates as a closed loop consisting of four key agents:

### 🧠 1. The Controller (RL Agent)
- **File**: `core/controller.py`
- **Role**: Strategic Commander.
- **Function**: Uses Reinforcement Learning (A2C) to decide the *direction* of exploration.
- **Actions**:
  - `MUTATION`: Drastically change the structure (high exploration).
  - `CROSSOVER`: Combine features of previous best models.
  - `INTENSITY`: Adjusts the "temperature" of the change based on recent rewards.

### 🌉 2. The Bridge (Prompt Engine)
- **File**: `prompt_engine/bridge.py`
- **Role**: Translator.
- **Function**: Converts the Controller's abstract numerical actions into rich, context-aware natural language prompts.
- **Key Technique**: Implements **Chain-of-Thought (CoT)** prompting to guide the LLM not just to write code, but to *reason* about why a specific architectural change might improve multimodal fusion (e.g., "Why might Cross-Attention work better here than Concatenation?").

### ✍️ 3. The Generator (LLM Interface)
- **File**: `core/generator.py`
- **Role**: Architect & Coder.
- **Function**: Interfaces with Google Gemini (or other LLMs) to generate executable PyTorch code (`class FusionModule`).
- **Features**:
  - **Self-Correction**: Includes fallback logic to retry generation if syntax errors occur.
  - **Safety**: Generates code within strict constraints (PyTorch only, specific signatures).

### 🧪 4. The Evaluator (Proxy Task)
- **File**: `core/evaluator.py`
- **Role**: Scientist.
- **Function**: Rapidly evaluates the generated architecture.
- **Method**:
  - **Proxy Task**: Instead of training a massive model end-to-end, it trains a lightweight adapter on pre-extracted features (DETR + T5).
  - **Speed**: Reduces evaluation time from hours to seconds per epoch.
  - **Metric**: Returns validation accuracy/loss as the **Reward** signal to the Controller.

## 2. The Evolutionary Process

1.  **Initialization**: The system starts with a simple seed architecture (or from scratch).
2.  **Cycle**:
    -   **Controller** observes current state -> Issues Action (e.g., "Mutate with High Intensity").
    -   **Bridge** translates this to: "Design a fusion module that uses a Gating mechanism to filter noise..."
    -   **Generator** writes the PyTorch code for `FusionModule`.
    -   **Evaluator** trains it on ScienceQA features -> Returns Reward (e.g., 0.75).
    -   **Controller** updates its policy based on the Reward.
3.  **Discovery**: Over generations, the system evolves from simple MLPs to complex Attention mechanisms.

## 3. Directory Structure

```
auto-fusion/
├── core/
│   ├── search_runner.py   # Main entry point (The Orchestrator)
│   ├── controller.py      # RL Agent logic
│   ├── generator.py       # LLM API interface
│   └── evaluator.py       # Training & Validation loop
├── prompt_engine/
│   └── bridge.py          # Prompt engineering logic
├── utils/
│   ├── dataset.py         # Data loading utilities
│   ├── adapter.py         # Dimension alignment modules
│   └── mps_patch.py       # Apple Silicon optimization
└── docs/
    └── README.md          # This file
```

## 4. How to Run the Search

To start a new evolutionary search session:

```bash
# Ensure PYTHONPATH is set
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the search runner
python auto-fusion/core/search_runner.py \
  --data_dir data/processed/scienceqa_features \
  --iterations 20 \
  --epochs 5
```

This will launch the autonomous loop. Results (best architectures) will be saved in `experiments/`.
