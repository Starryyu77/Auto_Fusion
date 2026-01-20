"""
Filename: adapter.py
Description: UnifiedFusionAdapter for MM-CoT (T5-Base) integration.
Author: Auto-Fusion Assistant
"""

import torch
import torch.nn as nn

class UnifiedFusionAdapter(nn.Module):
    """
    Adapts visual features to the language model's embedding space.
    Designed for T5-Base (d_model=768) and standard visual backbones.
    Updated: Default vis_dim=256 for DETR-ResNet50 compatibility.
    """
    def __init__(self, vis_dim=256, text_dim=768, dropout=0.1):
        super().__init__()
        
        # Project visual features to text dimension
        self.vis_projector = nn.Sequential(
            nn.Linear(vis_dim, text_dim),
            nn.LayerNorm(text_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Optional: Cross-Attention or Gating could be added here.
        # For now, we provide a robust projection that allows concatenation.
        
    def forward(self, image_features, text_features=None):
        """
        Args:
            image_features: [Batch, Seq_V, Vis_Dim]
            text_features:  [Batch, Seq_T, Text_Dim] (Optional, for cross-attn context)
        
        Returns:
            adapted_vision: [Batch, Seq_V, Text_Dim]
        """
        # 1. Project Visual Features
        # Ensure input is float (MPS stability)
        if image_features.dtype != torch.float32:
            image_features = image_features.float()
            
        adapted_vision = self.vis_projector(image_features)
        
        # 2. (Future) Interaction with text_features if needed
        # if text_features is not None:
        #     ... cross attention ...
            
        return adapted_vision
