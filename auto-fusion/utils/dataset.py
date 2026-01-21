"""
Filename: dataset.py
Description: 数据集加载器，用于加载预提取的 ScienceQA 特征。
             支持 Proxy Task 的快速训练。
Author: Auto-Fusion Assistant
"""

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
import logging
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)

class ProxyFeatureDataset(Dataset):
    """
    加载磁盘上的 .pt 特征文件。
    预期目录结构:
      data_dir/
        {split}_vision.pt  [N, Seq_V, Dim_V]
        {split}_text.pt    [N, Seq_T, Dim_T]
        {split}_labels.pt  [N, Dim_L] or [N]
    """
    def __init__(self, data_dir: str, split: str = 'train', load_in_memory: bool = True):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        
        # Paths
        vis_path = os.path.join(data_dir, f"{split}_vision.pt")
        text_path = os.path.join(data_dir, f"{split}_text.pt")
        label_path = os.path.join(data_dir, f"{split}_labels.pt")
        
        # Check existence
        if not all(os.path.exists(p) for p in [vis_path, text_path, label_path]):
            raise FileNotFoundError(f"Missing feature files in {data_dir} for split {split}")
            
        logger.info(f"Loading {split} data from {data_dir}...")
        
        # Load Data
        # mmap=True is useful for large files to avoid OOM, but for 'load_in_memory=True' we load fully.
        # MPS usually prefers data in RAM to transfer quickly.
        if load_in_memory:
            self.visual = torch.load(vis_path, map_location='cpu')
            self.text = torch.load(text_path, map_location='cpu')
            self.labels = torch.load(label_path, map_location='cpu')
        else:
            # mmap mode (read-only)
            self.visual = torch.load(vis_path, mmap=True, map_location='cpu')
            self.text = torch.load(text_path, mmap=True, map_location='cpu')
            self.labels = torch.load(label_path, mmap=True, map_location='cpu')
            
        # Validation
        assert len(self.visual) == len(self.text) == len(self.labels), \
            f"Size mismatch: V={len(self.visual)}, T={len(self.text)}, L={len(self.labels)}"
            
        logger.info(f"Loaded {len(self.labels)} samples.")
        logger.info(f"Shapes -> Vision: {self.visual.shape}, Text: {self.text.shape}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # Return dict, kept on CPU until DataLoader collate
        return {
            'visual': self.visual[idx],
            'text': self.text[idx],
            'label': self.labels[idx]
        }

def create_dataloaders(
    data_dir: str, 
    batch_size: int = 32, 
    num_workers: int = 0
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Helper to create Train and Val DataLoaders.
    """
    # Train Loader
    train_ds = ProxyFeatureDataset(data_dir, split='train')
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True # Optimize transfer to MPS/CUDA
    )
    
    # Val Loader (Optional check)
    val_loader = None
    if os.path.exists(os.path.join(data_dir, "val_vision.pt")):
        val_ds = ProxyFeatureDataset(data_dir, split='val')
        val_loader = DataLoader(
            val_ds, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
        
    return train_loader, val_loader
