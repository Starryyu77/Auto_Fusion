"""
Filename: download_scienceqa.py
Description: Automatically downloads ScienceQA dataset from Hugging Face 
             and restructures it for the Auto-Fusion project.
Usage: python tools/download_scienceqa.py
"""

import os
import json
import logging
from datasets import load_dataset
from PIL import Image
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_data_structure():
    """
    Downloads ScienceQA and organizes into:
    data/raw/ScienceQA/
      ├── problems.json
      ├── pid_splits.json
      └── images/
    """
    ROOT_DIR = "data/raw/ScienceQA"
    IMG_DIR = os.path.join(ROOT_DIR, "images")
    
    os.makedirs(IMG_DIR, exist_ok=True)
    
    logger.info("🚀 Downloading ScienceQA from Hugging Face (derek-thomas/ScienceQA)...")
    try:
        # Load dataset (this might take a while)
        dataset = load_dataset("derek-thomas/ScienceQA", split="train+validation+test")
        logger.info(f"✅ Downloaded {len(dataset)} samples.")
    except Exception as e:
        logger.error(f"❌ Failed to download dataset: {e}")
        return

    problems = {}
    splits = {"train": [], "val": [], "test": []}
    
    logger.info("📦 Processing and saving files...")
    
    for idx, item in enumerate(dataset):
        # Generate a unique ID if not present, usually ScienceQA has explicit logic
        # HF dataset might index them differently. We'll use index as string ID for simplicity
        # or try to preserve original ID if available. 
        # The HF version usually has 'image', 'question', 'choices', 'answer', 'hint', etc.
        # It doesn't always strictly preserve the original 'pid'.
        # We will create our own PID mapping.
        
        pid = str(idx)
        
        # Determine split (HF 'derek-thomas/ScienceQA' merges them or has config)
        # Actually, let's load specific splits to be precise.
        pass # We will re-loop below with explicit splits if possible, or random split here.
    
    # Reloading with explicit splits to ensure correctness
    ds_train = load_dataset("derek-thomas/ScienceQA", split="train")
    ds_val = load_dataset("derek-thomas/ScienceQA", split="validation")
    ds_test = load_dataset("derek-thomas/ScienceQA", split="test")
    
    all_splits = [('train', ds_train), ('val', ds_val), ('test', ds_test)]
    
    global_idx = 0
    
    for split_name, ds in all_splits:
        logger.info(f"Processing {split_name} split ({len(ds)} samples)...")
        for item in ds:
            pid = str(global_idx)
            splits[split_name].append(pid)
            
            # Save Image
            image = item.get('image')
            img_filename = None
            if image:
                img_filename = f"{pid}.jpg"
                # Convert to RGB to avoid mode issues
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image.save(os.path.join(IMG_DIR, img_filename))
            
            # Construct Problem Entry
            problem = {
                "question": item['question'],
                "choices": item['choices'],
                "answer": item['answer'],
                "hint": item['hint'],
                "image": img_filename,
                "task": item.get('task'),
                "grade": item.get('grade'),
                "subject": item.get('subject'),
                "topic": item.get('topic'),
                "category": item.get('category'),
                "skill": item.get('skill'),
                "lecture": item.get('lecture'),
                "solution": item.get('solution')
            }
            problems[pid] = problem
            global_idx += 1
            
    # Save JSONs
    with open(os.path.join(ROOT_DIR, "problems.json"), "w") as f:
        json.dump(problems, f, indent=2)
        
    with open(os.path.join(ROOT_DIR, "pid_splits.json"), "w") as f:
        json.dump(splits, f, indent=2)
        
    logger.info(f"✅ Data setup complete at {ROOT_DIR}")
    logger.info(f"Total Problems: {len(problems)}")
    logger.info(f"Images saved in: {IMG_DIR}")

if __name__ == "__main__":
    setup_data_structure()
