# Integration Guide: Surgical Replacement for MM-CoT (Track A)

This guide describes how to replace the standard fusion logic in MM-CoT with the **Auto-Fusion** generated module.

## 1. Unified Fusion Adapter

Since the Host Model (MM-CoT) and the Generated Module (`FusionModule`) may have different dimension expectations, we use a `UnifiedFusionAdapter` to bridge them.

**Location**: `src/adapter.py` (Suggested)

```python
import torch
import torch.nn as nn
from src.fusion_module import FusionModule

class UnifiedFusionAdapter(nn.Module):
    def __init__(self, visual_dim, text_dim, fusion_dim=512):
        super().__init__()
        # 1. Instantiate the Searchable Fusion Module
        self.fusion_module = FusionModule(dim=fusion_dim)
        
        # 2. Input Projectors (if host dims != 512)
        self.proj_v = nn.Linear(visual_dim, fusion_dim) if visual_dim != fusion_dim else nn.Identity()
        self.proj_t = nn.Linear(text_dim, fusion_dim) if text_dim != fusion_dim else nn.Identity()
        
        # 3. Output Projector (Back to Host Text Dim usually)
        self.proj_out = nn.Linear(fusion_dim, text_dim) if fusion_dim != text_dim else nn.Identity()

    def forward(self, v_feat, t_feat):
        # v_feat: [Batch, Seq_V, V_Dim]
        # t_feat: [Batch, Seq_T, T_Dim]
        
        # Project to 512
        v_512 = self.proj_v(v_feat)
        t_512 = self.proj_t(t_feat)
        
        # Apply Evolved Fusion Logic
        # Returns: [Batch, Seq_T, 512]
        fused_512 = self.fusion_module(v_512, t_512)
        
        # Project back to Host Dimension
        out = self.proj_out(fused_512)
        return out
```

## 2. Surgical Replacement in MM-CoT

**Target File**: `model/mmcot.py` (or equivalent `mm_cot.py`)
**Target Class**: `MMCoTModel` (typically inherits from `T5ForConditionalGeneration` or similar)

### Step A: Initialization (`__init__`)

Find where the visual and text encoders are defined. Add the adapter.

```python
# --- BEFORE ---
class MMCoTModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.detr = build_detr(args) # Visual Encoder (e.g., dim=256)
        self.t5 = T5ForConditionalGeneration.from_pretrained(args.backbone) # Text (dim=768)
        # Standard fusion might be just concatenation or simple attention here
        
# --- AFTER ---
from src.adapter import UnifiedFusionAdapter

class MMCoTModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.detr = build_detr(args)
        self.t5 = T5ForConditionalGeneration.from_pretrained(args.backbone)
        
        # INSERT ADAPTER
        # Assuming DETR dim=256, T5 dim=768
        self.fusion_adapter = UnifiedFusionAdapter(visual_dim=256, text_dim=768, fusion_dim=512)
```

### Step B: Forward Pass (`forward`)

Locate the `forward` method where visual features and text embeddings are available but not yet passed to the Decoder.

```python
# --- BEFORE ---
def forward(self, input_ids, image_tensors, ...):
    # 1. Encode Images
    v_feat = self.detr(image_tensors) # [B, N, 256]
    
    # 2. Encode Text
    t_feat = self.t5.encoder(input_ids).last_hidden_state # [B, M, 768]
    
    # 3. Standard Fusion (Example: Concatenation)
    # MM-CoT often concatenates along sequence dimension
    # v_feat_proj = self.linear(v_feat)
    # combined_feat = torch.cat([t_feat, v_feat_proj], dim=1)
    
    # 4. Decode
    # output = self.t5.decoder(encoder_hidden_states=combined_feat, ...)

# --- AFTER ---
def forward(self, input_ids, image_tensors, ...):
    # 1. Encode Images
    v_feat = self.detr(image_tensors) # [B, N, 256]
    
    # 2. Encode Text
    t_feat = self.t5.encoder(input_ids).last_hidden_state # [B, M, 768]
    
    # 3. SURGICAL REPLACEMENT: Apply Auto-Fusion Adapter
    # We want to ENHANCE the text features with visual context
    # output shape matches t_feat: [B, M, 768]
    fused_t_feat = self.fusion_adapter(v_feat, t_feat) 
    
    # 4. Decode
    # We pass the FUSED features as the encoder hidden states
    output = self.t5.decoder(
        input_ids=decoder_input_ids,
        encoder_hidden_states=fused_t_feat, # <--- REPLACED
        ...
    )
    return output
```
