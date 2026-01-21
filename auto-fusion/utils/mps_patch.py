"""
Filename: mps_patch.py
Description: Monkey patch to redirect CUDA calls to MPS for Apple Silicon.
Usage: Import this module at the very top of your training script.
       from src import mps_patch
       mps_patch.apply_patch()
Author: Auto-Fusion Assistant
"""

import torch
import logging

logger = logging.getLogger(__name__)

def apply_patch():
    """
    Applies monkey patches to torch.cuda to redirect calls to torch.backends.mps.
    """
    if not torch.backends.mps.is_available():
        logger.warning("MPS is not available. Patching to CPU instead.")
        target_device = "cpu"
    else:
        target_device = "mps"
        
    print(f"⚡ Applying MPS Monkey Patch (Target: {target_device}) ⚡")

    # 1. Patch torch.cuda.is_available
    # We force it to True so scripts don't bail out immediately.
    torch.cuda.is_available = lambda: True

    # 2. Patch .cuda() methods
    def custom_cuda(self, device=None, non_blocking=False):
        # Ignore device index, force to target_device
        return self.to(target_device, non_blocking=non_blocking)

    torch.Tensor.cuda = custom_cuda
    torch.nn.Module.cuda = custom_cuda
    
    # 3. Patch torch.cuda.device_count
    torch.cuda.device_count = lambda: 1
    
    # 4. Patch torch.cuda.current_device
    torch.cuda.current_device = lambda: 0
    
    # 5. Patch torch.cuda.get_device_name
    torch.cuda.get_device_name = lambda x=None: "Apple M-Series GPU"

    print("✅ torch.cuda.is_available() -> True")
    print(f"✅ .cuda() calls -> .to('{target_device}')")
