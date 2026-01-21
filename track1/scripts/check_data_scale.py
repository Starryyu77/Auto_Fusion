"""
Filename: check_data_scale.py
Description: Verifies the scale of the dataset to prevent benchmarking on Mini-Set.
"""
from src.dataset import create_dataloaders
import os
import logging

# Suppress info logs for cleaner output
logging.getLogger("src.dataset").setLevel(logging.WARNING)

def check():
    data_dir = 'data/processed/scienceqa_features'
    print(f"🔍 Checking dataset at: {data_dir}")
    
    try:
        # Load train set
        train_loader, _ = create_dataloaders(data_dir, batch_size=32)
        n = len(train_loader.dataset)
        
        print(f"📊 Current Training Set Size: {n} samples")
        
        if n < 1000:
            print("\n" + "="*60)
            print("⚠️  WARNING: Still running on MINI-SET! (n < 1000)")
            print("="*60)
            print("Baseline's low loss (0.0185) is likely due to OVERFITTING on this tiny set.")
            print("ACTION REQUIRED: Run 'tools/extract_features.py' without limit to process full data.")
        else:
            print("\n" + "="*60)
            print("✅  DATA READY: Running on FULL-SET. (n > 1000)")
            print("="*60)
            print("The comparison between Baseline and Best Model will now be statistically significant.")
            
    except Exception as e:
        print(f"❌ Error reading dataset: {e}")

if __name__ == "__main__":
    check()
