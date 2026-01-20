# Auto-Fusion Controller & Bridge Implementation Plan

## 1. RL Controller Implementation (`src/controller.py`)
- **Objective**: Implement `AutoFusionController` using A2C algorithm.
- **Specifications**:
  - **Input**: State tensor `[val_acc, param_cost, best_acc_history]`.
  - **Encoder**: Shared MLP to process state.
  - **Heads**:
    - `actor_op`: Categorical (MUTATION, CROSSOVER, FRESH_START).
    - `actor_int`: Continuous Sigmoid [0, 1] (Intensity).
    - `critic`: Scalar Value $V(s)$.
  - **Methods**: `forward(state)`, `select_action(state)`.

## 2. RL-to-LLM Bridge Logic (`src/bridge.py`)
- **Objective**: Translate RL actions into Natural Language Prompts.
- **Function**: `rl_to_llm_bridge(action_dict, top_k_history)`.
- **Logic**:
  - **Mutation**: Prompt for modification based on intensity (High=Structural, Low=Parameter).
  - **Crossover**: Prompt to merge two parent codes.
  - **Context**: Append "One-Shot" example of the best current architecture.

## 3. Track A Integration Guide (`docs/integration_guide.md`)
- **Objective**: Guide for surgically inserting the fusion module into MM-CoT.
- **Content**:
  - **Location**: Identify `MMCoTModel` class (typically where DETR and T5 meet).
  - **Before**: Show standard concatenation/attention.
  - **After**: Show instantiation of `UnifiedFusionAdapter` wrapping `FusionModule`.
  - **Code Snippet**: `MMCoTModel.forward` comparison.

## 4. Verification
- Create `tests/test_controller.py` to verify:
  - Input/Output shapes of the Controller.
  - Action selection logic and dictionary format.
- Create `tests/test_bridge.py` to verify:
  - Prompt generation based on different actions (Mutation vs Crossover).
  - One-shot example inclusion.

## 5. Documentation
- Update `.ai_status.md` with new modules and integration instructions.
