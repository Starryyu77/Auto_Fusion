"""
Filename: verify_pipeline.py
Description: End-to-end integration test for the Auto-Fusion pipeline.
             Generates dummy data, runs mock generator, and trains on MPS.
Usage: python scripts/verify_pipeline.py
"""

import os
import torch
import logging
from src.generator import AutoFusionGenerator
from src.evaluator import AutoFusionEvaluator
from tools.make_dummy_features import generate_features

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_pipeline():
    print("=== 🚀 Starting Auto-Fusion Pipeline Verification ===")
    
    # 0. Check MPS
    mps_available = torch.backends.mps.is_available()
    print(f"MPS Available: {mps_available}")
    if not mps_available:
        logger.warning("MPS is NOT available. Training will be slow on CPU.")
    
    # 1. Prepare Data
    data_dir = "data/processed_verify"
    logger.info(f"Step 1: Generating dummy features in {data_dir}...")
    generate_features(out_dir=data_dir, num_samples=50)
    
    # 2. Initialize Components
    logger.info("Step 2: Initializing Generator and Evaluator...")
    
    # Generator in Mock Mode (saves money)
    # mock_mode is not an init arg, it's passed to generate()
    generator = AutoFusionGenerator()
    
    # Evaluator in Real Mode (consumes data)
    # Note: dummy_mode=False enables loading from dataset_path
    evaluator = AutoFusionEvaluator(
        device='mps' if mps_available else 'cpu',
        dummy_mode=False, 
        proxy_epochs=2,
        dataset_path=data_dir
    )
    
    # 3. Generate Code
    logger.info("Step 3: Generating Fusion Architecture (Mock)...")
    prompt = "Design a fusion module with simple concatenation."
    gen_result = generator.generate(prompt, mock_mode=True)
    print(f"Code Preview:\n{gen_result['code'][:150]}...")
    
    # 4. Evaluate (Train Proxy Task)
    logger.info("Step 4: Running Proxy Training Task on MPS...")
    reward = evaluator.evaluate(gen_result['code'])
    
    # 5. Report
    if reward > 0:
        print(f"\n✅ Pipeline Verification SUCCESS! Reward: {reward:.4f}")
    else:
        print(f"\n❌ Pipeline Verification FAILED. Reward: {reward}")

if __name__ == "__main__":
    verify_pipeline()
