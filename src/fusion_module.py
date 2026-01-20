"""
Filename: fusion_module.py
Description: Track A (Reasoner) 专用融合模块。采用交叉注意力与门控机制。
Module: AutoFusion.TrackA
Author: Auto-Fusion Research Architect (AFRA)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionModule(nn.Module):
    """
    AutoFusion Track A: Gated Cross-Reasoning Adapter (GCRA)
    
    Design Philosophy:
    Uses Text as Query to retrieve relevant Visual evidence via Cross-Attention,
    then integrates information using a Learnable Gate to preserve textual reasoning flow.
    Adheres to strict signature: init(dim=512, dropout=0.1), forward(v_feat, t_feat).
    """
    def __init__(self, dim=512, dropout=0.1):
        super().__init__()
        
        self.search_dim = dim
        
        # 1. Feature Adapters (Refinement)
        # Even if input is 512, we use these for feature space alignment
        self.adapter_v = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        
        self.adapter_t = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )

        # 2. Cross-Modal Reasoning Core
        # Query: Text, Key/Value: Image
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        self.norm_attn = nn.LayerNorm(dim)

        # 3. Information Distillation (Gated Mechanism)
        # Gate controls how much retrieved visual info is fused
        self.fusion_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
        # 4. Output Projection
        self.head = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, v_feat, t_feat):
        """
        Args:
            v_feat: Visual features [Batch, Seq_V, 512]
            t_feat: Text features   [Batch, Seq_T, 512]
        Returns:
            fused_feat: [Batch, Seq_T, 512]
        """
        # 1. Alignment
        v_proj = self.adapter_v(v_feat)
        t_proj = self.adapter_t(t_feat)
        
        # 2. Cross-Attention: Text queries Image
        residual = t_proj
        
        # attn_output: [Batch, Seq_T, 512]
        attn_output, _ = self.cross_attn(
            query=t_proj, 
            key=v_proj, 
            value=v_proj
        )
        
        # Residual + Norm
        attn_output = self.norm_attn(attn_output + residual)
        
        # 3. Gated Integration
        # Combine original text intent (t_proj) with retrieved visual context (attn_output)
        combined = torch.cat([t_proj, attn_output], dim=-1) # [Batch, Seq_T, 1024]
        
        gate = self.fusion_gate(combined) # [Batch, Seq_T, 512]
        
        # Soft selection
        fused_internal = gate * attn_output + (1 - gate) * t_proj
        
        # 4. Final Projection
        fused_feat = self.head(self.dropout(fused_internal))
        
        return fused_feat
