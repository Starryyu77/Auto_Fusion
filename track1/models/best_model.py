# src/best_model.py 
import torch 
import torch.nn as nn 

class FusionModule(nn.Module): 
    """ 
    Auto-Fusion Gen-7 Champion Architecture 
    Method: Text-Query-Image Cross-Attention 
    Reward: 0.8399 
    """ 
    def __init__(self, dim=512, dropout=0.1): # 注意：Adapter 默认输出通常是 512，这里设为默认值 
        super().__init__() 
        self.dim = dim 
        self.dropout = nn.Dropout(dropout) 
        
        # Cross-Attention: Query=Text, Key/Value=Image 
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True) 
        self.norm1 = nn.LayerNorm(dim) 
        
        # Feed Forward Network 
        self.ff = nn.Sequential( 
            nn.Linear(dim, dim * 4), 
            nn.GELU(), 
            nn.Linear(dim * 4, dim) 
        ) 
        self.norm2 = nn.LayerNorm(dim) 
        
    def forward(self, v_feat, t_feat): 
        # v_feat: [B, S_V, D] (Vision) 
        # t_feat: [B, S_T, D] (Text) 
        
        # 1. Attention: Text queries Vision 
        # "What in the image matches this text?" 
        attn_out, _ = self.cross_attn(query=t_feat, key=v_feat, value=v_feat) 
        
        # 2. Add & Norm (Residual Connection) 
        x = self.norm1(t_feat + self.dropout(attn_out)) 
        
        # 3. FFN with Residual 
        x = self.norm2(x + self.dropout(self.ff(x))) 
        
        # 4. Global Pooling 
        # Reduce sequence to single vector for classification 
        return x.mean(dim=1)
