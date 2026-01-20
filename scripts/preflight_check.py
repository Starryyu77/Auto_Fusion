"""
Filename: preflight_check.py
Description: System health check script for Auto-Fusion.
             Verifies raw data integrity, model cache availability, and feature store readiness.
Usage: python scripts/preflight_check.py
"""

import os
import glob
import logging
from transformers import T5EncoderModel, DetrModel

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'

def log_status(message, status="INFO"):
    if status == "PASS":
        print(f"{Colors.GREEN}✅ {message}{Colors.RESET}")
    elif status == "FAIL":
        print(f"{Colors.RED}❌ {message}{Colors.RESET}")
    elif status == "WARN":
        print(f"{Colors.YELLOW}⚠️  {message}{Colors.RESET}")
    else:
        print(f"ℹ️  {message}")

def check_raw_data():
    print("\n🔍 [1/3] Checking Raw ScienceQA Data...")
    raw_dir = "data/raw/ScienceQA"
    prob_path = os.path.join(raw_dir, "problems.json")
    img_dir = os.path.join(raw_dir, "images")
    
    # 1. Check problems.json
    if not os.path.exists(prob_path):
        log_status("problems.json missing!", "FAIL")
    else:
        size_mb = os.path.getsize(prob_path) / (1024 * 1024)
        if size_mb > 10:
            log_status(f"problems.json found ({size_mb:.2f} MB)", "PASS")
        else:
            log_status(f"problems.json too small ({size_mb:.2f} MB). Expected > 10MB.", "WARN")
            
    # 2. Check Images
    if not os.path.exists(img_dir):
        log_status("images/ directory missing!", "FAIL")
    else:
        # Use fast scandir instead of glob for large directories
        count = sum(1 for _ in os.scandir(img_dir))
        if count > 21000:
            log_status(f"Found {count} images (Complete Set)", "PASS")
        elif count > 0:
            log_status(f"Found {count} images (Partial/Mini Set)", "WARN")
        else:
            log_status("No images found!", "FAIL")

def check_models():
    print("\n🔍 [2/3] Checking Local Model Cache...")
    models = [
        ('google/flan-t5-base', T5EncoderModel), 
        ('facebook/detr-resnet-50', DetrModel)
    ]
    
    for model_name, model_class in models:
        try:
            # Force check local cache
            _ = model_class.from_pretrained(model_name, local_files_only=True)
            log_status(f"{model_name}: Found in local cache.", "PASS")
        except Exception:
            log_status(f"{model_name}: Not found locally. Internet required for first run.", "WARN")

def check_features():
    print("\n🔍 [3/3] Checking Processed Features...")
    feat_dir = "data/processed/scienceqa_features"
    
    if not os.path.exists(feat_dir):
        log_status(f"Feature directory {feat_dir} does not exist.", "FAIL")
        return

    # Calculate total size
    total_size = 0
    pt_files = glob.glob(os.path.join(feat_dir, "*.pt"))
    
    if not pt_files:
        log_status("No .pt feature files found.", "FAIL")
        return
        
    for f in pt_files:
        total_size += os.path.getsize(f)
    
    size_gb = total_size / (1024 * 1024 * 1024)
    
    if size_gb > 1.0:
        log_status(f"Feature store size: {size_gb:.2f} GB (Ready for Full Search)", "PASS")
    else:
        log_status(f"Feature store size: {size_gb:.2f} GB (Likely Mini-Set)", "WARN")
        print(f"   (Found files: {[os.path.basename(f) for f in pt_files]})")

if __name__ == "__main__":
    print("=== Auto-Fusion Preflight Check ===")
    check_raw_data()
    check_models()
    check_features()
    print("\n=== Check Complete ===")
