"""
Filename: verify_mps_baseline.py
Description: Verifies PyTorch MPS acceleration and runs a dummy inference on T5.
Author: Auto-Fusion Assistant
"""

import torch
import psutil
import os
import sys
from transformers import AutoTokenizer, T5ForConditionalGeneration

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def verify_setup():
    print("=== MPS Verification & Baseline Check ===")
    
    # 1. Device Detection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✅ Success: MPS acceleration is available.")
    else:
        device = torch.device("cpu")
        print(f"⚠️ Warning: MPS not available. Using CPU.")
    
    print(f"Using Device: {device}")

    # 2. Model Loading
    model_name = "google/t5-v1_1-base"
    print(f"\nLoading model: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        model.to(device)
        print("✅ Model loaded and moved to device.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

    # 3. Mock Inference
    print("\nRunning Mock Inference...")
    input_text = "translate English to German: The cat is on the table."
    
    # Text Input
    inputs = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    
    # Dummy Image Input
    # Note: Even if T5-base is text-only, we create this to verify MPS tensor handling for floats.
    dummy_image = torch.randn(1, 3, 224, 224, dtype=torch.float32).to(device)
    print(f"Dummy Image Tensor created on {dummy_image.device}")

    # Forward Pass
    try:
        outputs = model.generate(inputs)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Input: {input_text}")
        print(f"Output: {decoded}")
        print("✅ Forward pass successful.")
    except Exception as e:
        print(f"❌ Error during inference: {e}")

    # 4. Memory Check
    mem_usage = get_memory_usage()
    print(f"\nApprox. Memory Usage: {mem_usage:.2f} MB")
    
if __name__ == "__main__":
    verify_setup()
