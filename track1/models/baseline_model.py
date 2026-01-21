"""
Filename: baseline_model.py
Description: A simple baseline fusion module using Concatenation + MLP.
             Used for benchmarking against the evolved architecture.
"""
import torch
import torch.nn as nn

class FusionModule(nn.Module):
    """
    Baseline Architecture: Simple Concatenation
    Method: Mean Pool -> Concat -> MLP
    """
    def __init__(self, dim=512, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.classifier = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        
    def forward(self, v_feat, t_feat):
        # v_feat: [B, S_V, D] -> [B, D]
        v_pool = v_feat.mean(dim=1)
        # t_feat: [B, S_T, D] -> [B, D]
        t_pool = t_feat.mean(dim=1)
        
        # Concat: [B, 2D]
        combined = torch.cat([v_pool, t_pool], dim=-1)
        
        # Fuse -> [B, D]
        return self.classifier(combined)
