"""
Filename: search_runner.py
Description: Auto-Fusion 系统的核心调度脚本，负责闭环进化搜索。
             整合 Controller (RL), Generator (LLM), Evaluator (Proxy Task) 和 Bridge。
Usage: python src/search_runner.py --mock --iterations 5
Author: Auto-Fusion Assistant
"""

import argparse
import logging
import torch
from typing import List, Dict, Any

from src.controller import AutoFusionController
from src.generator import AutoFusionGenerator
from src.evaluator import AutoFusionEvaluator
from src.bridge import rl_to_llm_bridge
from src import mps_patch

# Apply MPS patch globally
mps_patch.apply_patch()

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import os
import time

def run_search(args):
    """
    Main Evolution Loop
    """
    # Create experiments directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiments/run_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    print(f"📂 Saving experiment results to {exp_dir}")

    print(f"🚀 Starting Auto-Fusion Search (Mock={args.mock}, Iterations={args.iterations})...")
    
    # 1. Initialization
    # RL Controller: Manages exploration vs exploitation
    # Updated: input_dim=3 (val_acc, param_cost, best_acc_history)
    controller = AutoFusionController(input_dim=3, hidden_dim=64)
    
    # Generator: Interfaces with LLM (or mock)
    # Note: Generator's mock_mode is passed in generate(), but we can init with defaults
    generator = AutoFusionGenerator(model_name="gpt-4")
    
    # Evaluator: Runs proxy tasks on MPS
    # Determine mode: if data_dir is provided, we run real evaluation (dummy_mode=False)
    # If mock flag is explicitly set, we respect it, unless data_dir overrides it for evaluator.
    # Logic: 
    # - If data_dir: dummy_mode = False (Real Data)
    # - Else: dummy_mode = args.mock (Default True)
    use_dummy_eval = not bool(args.data_dir)
    
    if args.data_dir:
        print(f"📂 Real Data Mode: Loading from {args.data_dir}")
        # When using real data, we usually want to use real generator too, unless explicitly mocked.
        # But user might want to test Real Data + Mock Generator (to save tokens).
        # So we keep generator mock_mode separate.
    else:
        print(f"🎭 Mock Data Mode: Using random tensors")

    evaluator = AutoFusionEvaluator(
        device='mps', 
        dummy_mode=use_dummy_eval, 
        proxy_epochs=args.epochs,
        dataset_path=args.data_dir
    )
    
    # History Archive: Stores successful architectures
    # Format: [{'code': str, 'reward': float, 'thought': str}]
    history_archive: List[Dict[str, Any]] = []
    
    # Initial State (Dummy: Accuracy=0.5, Complexity=0.5, Best=0.5)
    current_state = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    
    # 2. Evolution Loop
    for i in range(args.iterations):
        print(f"\n--- Generation {i+1}/{args.iterations} ---")
        
        # Step 1: Observation & Decision (RL)
        # Controller decides the high-level strategy (e.g., "Add Gate", "Increase Depth")
        # Action is a dict containing op_type, intensity etc.
        action_dict = controller.select_action(current_state)
        action_type = action_dict["op_type"]
        print(f"🧠 Controller Action: {action_type} (Intensity: {action_dict['intensity']:.2f})")
        
        # Step 2: Prompt Engineering (Bridge)
        # Retrieve Top-K examples for In-Context Learning
        top_k_examples = sorted(history_archive, key=lambda x: x['reward'], reverse=True)[:3]
        
        # Generate prompt using the Bridge
        # Updated to match src/bridge.py signature: rl_to_llm_bridge(action_dict, top_k_history)
        prompt = rl_to_llm_bridge(action_dict, top_k_examples)
        # (Optional) Print part of prompt for debug
        # print(f"📝 Prompt Preview: {prompt[:100]}...")
        
        # Step 3: Code Generation (LLM)
        # Returns {'thought': str, 'code': str}
        gen_result = generator.generate(prompt, mock_mode=args.mock)
        print(f"✍️  Generator Thought: {gen_result['thought'][:100]}...")
        
        # Step 4: Evaluation (Proxy Task)
        # Returns float reward (Accuracy - Penalty) or -1.0 on failure
        reward = -1.0
        try:
            reward = evaluator.evaluate(gen_result['code'])
            print(f"🧪 Evaluation Reward: {reward:.4f}")
        except Exception as e:
            logger.error(f"❌ Evaluation Failed: {e}")
            reward = -1.0
            
        # Step 5: Policy Update (RL)
        # Feedback the reward to update the controller
        # For now, we assume controller has no learnable update exposed or we skip it for mock run.
        # Ideally: controller.update_policy(reward, log_probs...)
        # But controller.py shown doesn't have it. We'll just log the reward feedback.
        # logger.info(f"RL Feedback: Reward={reward}")
        
        # Step 6: Archiving & State Update
        if reward > -1.0:
            history_archive.append({
                'code': gen_result['code'],
                'thought': gen_result['thought'],
                'reward': reward
            })
            # Update state based on result (Simulated logic for now)
            # State: [val_acc, param_cost, best_acc_history]
            new_acc = min(1.0, current_state[0] + 0.05)
            new_cost = current_state[1] # Keep cost same for mock
            best_acc = max(current_state[2], new_acc)
            current_state = torch.tensor([new_acc, new_cost, best_acc], dtype=torch.float32)
            
            # Save if it's the best so far
            if not history_archive or reward > max(a['reward'] for a in history_archive[:-1]):
                print(f"🌟 New Best Architecture Found! (Reward: {reward:.4f})")
                with open(os.path.join(exp_dir, "best_fusion.py"), "w") as f:
                    f.write(gen_result['code'])
                
        else:
            print("⚠️ Architecture failed, state remains unchanged.")
            
    # 3. Final Report
    print("\n✅ Search Complete.")
    print(f"Total Architectures Found: {len(history_archive)}")
    if history_archive:
        best_arch = max(history_archive, key=lambda x: x['reward'])
        print(f"🏆 Best Reward: {best_arch['reward']:.4f}")
        # print(f"Best Code:\n{best_arch['code']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-Fusion Evolutionary Search Runner")
    parser.add_argument('--iterations', type=int, default=5, help='Number of search iterations')
    parser.add_argument('--mock', action='store_true', default=False, help='Run in Mock mode (no LLM, dummy data)')
    parser.add_argument('--task', type=str, default='track_a', help='Task identifier')
    parser.add_argument('--data_dir', type=str, default=None, help='Path to real feature data (e.g. data/processed/scienceqa_features)')
    parser.add_argument('--epochs', type=int, default=1, help='Number of proxy training epochs')
    
    args = parser.parse_args()
    run_search(args)
