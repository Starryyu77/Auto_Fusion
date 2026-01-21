"""
Filename: bridge.py
Description: Translates RL Controller actions into Natural Language Prompts for the LLM.
Module: AutoFusion.Bridge
Author: Auto-Fusion Agent
"""

def rl_to_llm_bridge(action_dict, top_k_history):
    """
    Translates RL Agent's numeric output into a Natural Language Prompt.
    
    Args:
        action_dict (dict): Output from AutoFusionController.select_action()
                            Expected keys: "op_type", "intensity"
        top_k_history (list): List of best past code snippets. 
                              Each item should be a dict or object with a 'code' attribute or be the code string.
    
    Returns:
        str: A prompt to be appended to the System Meta-Prompt.
    """
    op_type = action_dict["op_type"]
    intensity = action_dict["intensity"]
    
    # Get best code (One-Shot)
    best_code = ""
    if top_k_history:
        # Assuming top_k_history[0] is the best
        # Handle if it's a dict or string
        item = top_k_history[0]
        best_code = item['code'] if isinstance(item, dict) and 'code' in item else str(item)
    
    prompt = ""
    
    # Common Context - Strict Requirements
    prompt += "### STRICT REQUIREMENTS (MUST FOLLOW OR FAIL):\n"
    prompt += "1. **Language**: Use ONLY PyTorch (`torch`, `torch.nn`). DO NOT import TensorFlow/Keras.\n"
    prompt += "2. **Class Name**: Must be `class FusionModule(nn.Module):`.\n"
    prompt += "3. **Signature**: `__init__` MUST accept `dim` (int) and `dropout` (float). Example: `def __init__(self, dim=768, dropout=0.1):`\n"
    prompt += "4. **Forward**: `forward` MUST accept `v_feat` (B, S_V, D) and `t_feat` (B, S_T, D). Assume batch_first=True.\n"
    prompt += "5. **Output**: Must return a tensor of shape (B, D). You MUST use pooling (mean/max/attn) to reduce sequence dimensions.\n"
    prompt += "6. **Dimensions**: Ensure LayerNorm/Linear applied to last dimension D. Do NOT swap dimensions (e.g. B, D, S) unless necessary and swapped back.\n"
    prompt += "7. **No Placeholders**: Do not use `...` or `pass`. Write fully working code.\n\n"

    prompt += f"Current State: The best architecture so far is provided below.\n"
    prompt += f"```python\n{best_code}\n```\n\n"
    
    # Operation Specific Logic
    if op_type == "MUTATION":
        prompt += f"Task: MUTATION\n"
        if intensity > 0.7:
            prompt += (
                f"Intensity: HIGH ({intensity:.2f})\n"
                "Instruction: Perform MAJOR structural changes to the fusion mechanism. "
                "You can introduce new modules, change the topology (e.g., parallel vs sequential), "
                "or replace core operators (e.g., Attention -> Gating). Be bold and innovative."
            )
        elif intensity > 0.3:
            prompt += (
                f"Intensity: MEDIUM ({intensity:.2f})\n"
                "Instruction: Refine the existing structure. "
                "Optimize dimensions, activation functions, or add normalization layers. "
                "Try to improve gradient flow and feature interaction."
            )
        else:
            prompt += (
                f"Intensity: LOW ({intensity:.2f})\n"
                "Instruction: Perform fine-tuning. "
                "Adjust hyperparameters like dropout rates, hidden dimensions (within limits), "
                "or attention head counts. Keep the structure stable."
            )
            
    elif op_type == "CROSSOVER":
        prompt += f"Task: CROSSOVER\n"
        # Need a second parent
        parent2_code = ""
        if len(top_k_history) > 1:
            item2 = top_k_history[1]
            parent2_code = item2['code'] if isinstance(item2, dict) and 'code' in item2 else str(item2)
        else:
            # Fallback if only 1 history exists
            parent2_code = best_code 
            
        prompt += (
            "Instruction: I will provide a second parent architecture below. "
            "Your goal is to MERGE the best features of both parents into a single hybrid module.\n"
            f"Parent 2:\n```python\n{parent2_code}\n```\n"
            "Analyze both parents and combine their strengths (e.g., Parent 1's gating with Parent 2's attention)."
        )
        
    elif op_type == "FRESH_START":
        prompt += f"Task: FRESH START\n"
        prompt += (
            f"Intensity: {intensity:.2f}\n"
            "Instruction: Discard the previous architecture (but keep the lessons in mind). "
            "Design a completely NEW FusionModule from scratch. "
            "Explore a different paradigm (e.g., if previous was Attention-based, try MLP-Mixer or tensor fusion)."
        )
        
    return prompt
