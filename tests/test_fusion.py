import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.fusion_module import FusionModule

def test_fusion_module():
    print("Testing FusionModule...")
    
    # Parameters
    batch_size = 4
    seq_v = 49
    seq_t = 128
    dim = 512
    
    # Initialize model
    model = FusionModule(dim=dim, dropout=0.1)
    print("Model initialized successfully.")
    
    # Create dummy inputs
    v_feat = torch.randn(batch_size, seq_v, dim)
    t_feat = torch.randn(batch_size, seq_t, dim)
    
    # Forward pass
    output = model(v_feat, t_feat)
    
    # Check output shape
    expected_shape = (batch_size, seq_t, dim)
    print(f"Output shape: {output.shape}")
    
    assert output.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {output.shape}"
    print("Shape verification passed!")
    
    print("All tests passed.")

if __name__ == "__main__":
    test_fusion_module()
