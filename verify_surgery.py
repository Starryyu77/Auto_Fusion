"""
Filename: verify_surgery.py
Description: Verifies the 'Model Surgery' by applying the UnifiedFusionAdapter 
             to a Mock MM-CoT model on MPS.
Author: Auto-Fusion Assistant
"""

# 1. Apply MPS Patch immediately
from src import mps_patch
mps_patch.apply_patch()

import torch
import torch.nn as nn
from src.adapter import UnifiedFusionAdapter

# --- MOCK CLASSES (Simulating MM-CoT/T5) ---
class MockConfig:
    d_model = 768
    dropout_rate = 0.1

class MockMMCoT(nn.Module):
    """
    Simulates the original MM-CoT model with hardcoded fusion.
    """
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        # Simulate T5 components
        self.shared = nn.Embedding(32000, config.d_model)
        self.encoder = nn.Linear(config.d_model, config.d_model) # Dummy encoder
        
    def forward(self, input_ids, image_features):
        # ORIGINAL LOGIC (Simulated):
        # Usually it projects image_features then concatenates.
        # We simulate a naive projection here for the "Before" state.
        
        # Text Embeddings
        inputs_embeds = self.shared(input_ids)
        
        # Hardcoded Projection (Simulating what we want to replace)
        # Assume original model had some hardcoded linear layer or similar
        # For this mock, let's say it just expected projected features or did a simple cat
        # But to show the surgery, we'll replace the *fusion point*.
        
        print(">> [Original] Forward called (Pre-Surgery logic)")
        # In real MM-CoT, there might be a projection here, or it expects pre-projected.
        # Let's assume we want to inject our adapter *before* this concatenation.
        
        combined = torch.cat([image_features, inputs_embeds], dim=1)
        return self.encoder(combined)

# --- SURGERY TOOLS ---
def apply_surgery(model_class):
    print("\n🔪 Performing Model Surgery on", model_class.__name__)
    
    # 1. Capture original init
    original_init = model_class.__init__
    
    def new_init(self, config, *args, **kwargs):
        original_init(self, config, *args, **kwargs)
        # Inject Adapter
        print("   💉 Injecting UnifiedFusionAdapter...")
        # Assuming visual dim 2048 for ResNet/DETR, T5 dim 768
        self.auto_fusion_adapter = UnifiedFusionAdapter(vis_dim=2048, text_dim=768)
        self.auto_fusion_adapter.to(self.shared.weight.device) # Sync device
        
    # 2. Define new forward
    def new_forward(self, input_ids, image_features):
        print(">> [Patched] Forward called (New Logic)")
        
        # Get Text Embeds
        inputs_embeds = self.shared(input_ids)
        
        # 1. ADAPT VISION
        # image_features: [B, Seq_V, 2048] -> [B, Seq_V, 768]
        adapted_vision = self.auto_fusion_adapter(image_features, inputs_embeds)
        
        # 2. FUSE (Concatenate)
        combined_embeds = torch.cat([adapted_vision, inputs_embeds], dim=1)
        
        # Continue with encoder
        return self.encoder(combined_embeds)

    # 3. Apply Patches
    model_class.__init__ = new_init
    model_class.forward = new_forward
    print("✅ Surgery Complete.")

# --- VERIFICATION MAIN ---
def run_verification():
    print("=== Verifying Surgery on MPS ===")
    device = torch.device("mps")
    
    # 1. Apply Surgery to the Class
    apply_surgery(MockMMCoT)
    
    # 2. Instantiate Model
    config = MockConfig()
    model = MockMMCoT(config).to(device)
    
    # 3. Verify Attribute
    if hasattr(model, 'auto_fusion_adapter'):
        print("✅ Attribute 'auto_fusion_adapter' found.")
    else:
        print("❌ FAILED: Adapter not found.")
        return

    # 4. Mock Data (MPS)
    batch_size = 2
    seq_len_text = 10
    seq_len_vis = 49 # e.g. 7x7 patches
    vis_dim = 2048
    
    input_ids = torch.randint(0, 32000, (batch_size, seq_len_text)).to(device)
    # Important: Create visual features in native dim (2048) to test projection
    image_features = torch.randn(batch_size, seq_len_vis, vis_dim).to(device)
    
    print(f"Input Shapes: Text {input_ids.shape}, Vision {image_features.shape}")
    
    # 5. Run Forward
    try:
        output = model(input_ids, image_features)
        print(f"Output Shape: {output.shape}")
        # Expected: [Batch, Seq_V + Seq_T, 768]
        expected_seq = seq_len_vis + seq_len_text
        if output.shape[1] == expected_seq and output.shape[2] == 768:
            print("✅ Dimensions match expected fusion.")
        else:
            print(f"⚠️ Unexpected dimensions. Expected [*, {expected_seq}, 768]")
            
    except Exception as e:
        print(f"❌ Runtime Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_verification()
