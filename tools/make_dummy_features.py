"""
Filename: make_dummy_features.py
Description: 生成随机的特征张量用于测试 Evaluator 和 DataLoader。
             模拟 ScienceQA 的特征维度。
Usage: python tools/make_dummy_features.py --out_dir data/processed --num_samples 100
"""

import torch
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_features(out_dir: str, num_samples: int = 100):
    os.makedirs(out_dir, exist_ok=True)
    
    splits = ['train', 'val']
    
    # Dimensions based on Project Protocol
    # Vision: ResNet/DETR usually 2048 or 1024. Adapter defaults to 2048.
    # Text: T5-Base embedding is 768.
    DIM_V = 2048
    DIM_T = 768
    SEQ_V = 49  # 7x7 patches
    SEQ_T = 64  # Text sequence length
    
    for split in splits:
        n = num_samples if split == 'train' else num_samples // 5
        logger.info(f"Generating {split} data ({n} samples)...")
        
        # 1. Vision Features
        # [N, Seq, Dim]
        vision = torch.randn(n, SEQ_V, DIM_V, dtype=torch.float32)
        torch.save(vision, os.path.join(out_dir, f"{split}_vision.pt"))
        
        # 2. Text Features
        text = torch.randn(n, SEQ_T, DIM_T, dtype=torch.float32)
        torch.save(text, os.path.join(out_dir, f"{split}_text.pt"))
        
        # 3. Labels
        # Regression or Classification target? 
        # For proxy task, let's assume we predict a target vector matching Text Dim (like next token prediction proxy)
        # or a simple class label. Evaluator.py uses MSELoss on output vs target.
        # In evaluator.py dummy run: target = torch.randn(..., dim)
        # So we should save a target tensor of [N, Seq_T, DIM_T] for reconstruction/feature regression task.
        labels = torch.randn(n, SEQ_T, DIM_T, dtype=torch.float32)
        torch.save(labels, os.path.join(out_dir, f"{split}_labels.pt"))
        
    logger.info(f"✅ Dummy features saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='data/processed', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of training samples')
    args = parser.parse_args()
    
    generate_features(args.out_dir, args.num_samples)
