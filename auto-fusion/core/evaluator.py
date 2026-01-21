"""
Filename: evaluator.py
Description: 核心评估模块，负责动态加载、安全检查和评估 LLM 生成的 Fusion 架构。
             支持在 MPS (M4) 上进行 Proxy Task 训练或 Dummy Run。
Author: Auto-Fusion Assistant
"""

import torch
import torch.nn as nn
import ast
import logging
import time
from typing import Dict, Any, Optional
from src import mps_patch
from src.adapter import UnifiedFusionAdapter
from src.dataset import create_dataloaders

# 初始化日志
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 应用 MPS Patch
mps_patch.apply_patch()

class AutoFusionEvaluator:
    """
    AutoFusionEvaluator 负责接收生成的 Python 代码，动态构建模型，
    并计算奖励信号 (Reward Signal)。
    """
    def __init__(
        self, 
        device: str = 'mps', 
        dummy_mode: bool = True,
        proxy_epochs: int = 1,
        dataset_path: Optional[str] = None
    ):
        """
        Args:
            device: 'mps' for Apple Silicon, 'cpu' otherwise.
            dummy_mode: If True, uses random tensors instead of real data.
            proxy_epochs: Number of epochs for the proxy training task.
            dataset_path: Path to feature dataset directory (ignored in dummy_mode).
        """
        self.device = torch.device(device) if torch.backends.mps.is_available() else torch.device("cpu")
        self.dummy_mode = dummy_mode
        self.proxy_epochs = proxy_epochs
        self.dataset_path = dataset_path
        
        logger.info(f"Initialized AutoFusionEvaluator on {self.device} (Dummy Mode: {self.dummy_mode})")

    def _validate_ast(self, code_str: str) -> bool:
        """
        使用 AST 检查代码安全性，防止执行恶意系统调用。
        不允许: import os, sys, subprocess, etc.
        """
        try:
            tree = ast.parse(code_str)
            for node in ast.walk(tree):
                # 检查 import 语句
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        if alias.name in ['os', 'sys', 'subprocess', 'shutil']:
                            logger.error(f"Security Alert: Forbidden import '{alias.name}' detected.")
                            return False
                # 检查函数调用 (简易版)
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        # e.g., os.system
                        if hasattr(node.func.value, 'id') and node.func.value.id == 'os':
                            logger.error("Security Alert: os.* call detected.")
                            return False
            return True
        except SyntaxError as e:
            logger.error(f"AST Parsing Error: {e}")
            return False

    def safe_load_module(self, code_str: str) -> Optional[nn.Module]:
        """
        动态加载生成的代码，并实例化 FusionModule。
        """
        if not self._validate_ast(code_str):
            return None

        local_scope = {}
        try:
            # 1. Execute code string to define the class
            # Pass globals() to allow access to imported modules (torch, nn) within the executed code
            exec(code_str, globals(), local_scope)
            
            # 2. Identify the class
            # Priority: Look for 'FusionModule' explicitly
            module_class = local_scope.get('FusionModule')
            
            # Fallback: Look for any nn.Module subclass
            if module_class is None:
                for name, obj in local_scope.items():
                    if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module:
                        module_class = obj
                        break
            
            if module_class is None:
                logger.error("No 'FusionModule' or nn.Module subclass found in generated code.")
                return None
                
            # 3. Instantiate
            # 假设标准初始化接口: __init__(self, dim=512, dropout=0.1)
            # 我们根据 Adapter 的需求，可能需要调整 dim 参数
            # UnifiedFusionAdapter 默认 text_dim=768
            instance = module_class(dim=768) 
            return instance.to(self.device)
            
        except Exception as e:
            logger.error(f"Runtime Error during module loading: {e}")
            return None

    def _run_dummy_proxy_task(self, model: nn.Module) -> float:
        """
        执行 Dummy 训练循环：使用随机数据验证模型的前向/反向传播。
        """
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        
        start_time = time.time()
        
        try:
            for epoch in range(self.proxy_epochs):
                # Mock Data
                batch_size = 4
                seq_len = 16
                dim = 768
                
                # Inputs: [Batch, Seq, Dim]
                # 注意：这里模拟的是 Adapter 内部的 FusionModule 接收的输入
                # 通常 Adapter 会处理投影，这里为了测试 FusionModule 本身，我们假设输入已经是投影后的形状
                # 或者我们应该测试整个 Adapter + FusionModule 的组合
                
                v_feat = torch.randn(batch_size, seq_len, dim).to(self.device)
                t_feat = torch.randn(batch_size, seq_len, dim).to(self.device)
                target = torch.randn(batch_size, seq_len, dim).to(self.device) # Dummy target
                
                optimizer.zero_grad()
                
                # Forward
                output = model(v_feat, t_feat)
                
                # Check output shape
                if output.shape != t_feat.shape:
                    logger.warning(f"Shape mismatch: {output.shape} vs {t_feat.shape}")
                    return -0.5 # Penalty for shape mismatch
                
                # Loss & Backward
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                logger.info(f"Epoch {epoch+1}/{self.proxy_epochs} | Dummy Loss: {loss.item():.4f}")
                
            # Dummy Accuracy (Simulated)
            # 如果能跑通，说明代码逻辑没问题，给一个基础分
            # 减去参数量惩罚
            param_count = sum(p.numel() for p in model.parameters())
            logger.info(f"Model Parameters: {param_count}")
            
            # Reward: Base (0.8) - Params Penalty (1e-6 * params)
            reward = 0.8 - (1e-7 * param_count)
            return max(0.0, reward) # Ensure non-negative
            
        except RuntimeError as e:
            logger.error(f"Training Error: {e}")
            return -1.0

    def _run_real_proxy_task(self, model: nn.Module) -> float:
        """
        Executes real training loop using ProxyFeatureDataset.
        """
        if not self.dataset_path:
            logger.error("Dataset path not provided for real training.")
            return -1.0
            
        logger.info(f"Loading data from {self.dataset_path}...")
        train_loader, val_loader = create_dataloaders(self.dataset_path, batch_size=32)
        
        # Wrap model with Adapter to handle dimensions
        # Visual: 256 -> 768 (Adapted for DETR), Text: 768 -> 768
        # Then fusion module processes 768 + 768 -> 768
        adapter = UnifiedFusionAdapter(vis_dim=256, text_dim=768)
        # Inject the generated fusion module into the adapter
        # Note: UnifiedFusionAdapter currently doesn't have a slot for fusion_module,
        # it just does projection. We need to construct a pipeline:
        # Input -> Adapter(Projection) -> FusionModule(Interaction) -> Output
        
        class FullModel(nn.Module):
            def __init__(self, adapter, fusion):
                super().__init__()
                self.adapter = adapter
                self.fusion = fusion
                self.head = nn.Linear(768, 768) # Mock classification head
                
            def forward(self, v, t):
                # v: [B, S_V, 2048], t: [B, S_T, 768]
                v_proj = self.adapter(v) # -> [B, S_V, 768]
                # Assuming Fusion Module takes (v, t) and returns fused features
                # Note: Fusion Module inputs must match dimensions. 
                # If generated module expects 768, v_proj is ready.
                
                # Check dimensions. v_proj is [B, S_V, 768]. t is [B, S_T, 768].
                # Ensure float type for safety
                v_proj = v_proj.float()
                t = t.float()
                
                # FusionModule usually expects same sequence length for simple concat, 
                # or uses Cross-Attention.
                # For this proxy task, we assume the FusionModule handles (v, t).
                fused = self.fusion(v_proj, t)
                
                # Simple pooling for classification
                if fused.dim() == 3:
                    fused = fused.mean(dim=1)
                return self.head(fused)

        full_model = FullModel(adapter, model).to(self.device)
        
        optimizer = torch.optim.AdamW(full_model.parameters(), lr=1e-4)
        criterion = nn.MSELoss() # Using MSE against random targets for proxy
        
        full_model.train()
        
        try:
            for epoch in range(self.proxy_epochs):
                total_loss = 0
                steps = 0
                for batch in train_loader:
                    v = batch['visual'].to(self.device)
                    t = batch['text'].to(self.device)
                    target = batch['label'].to(self.device)
                    
                    # Target dimension check
                    # If target is [B, S_T, 768], we need to match output
                    # FullModel currently pools to [B, 768].
                    # Let's adjust target or model. 
                    # Dataset labels are [B, S_T, 768]. Let's pool target too.
                    if target.dim() == 3:
                        target = target.mean(dim=1)
                        
                    optimizer.zero_grad()
                    output = full_model(v, t)
                    
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    steps += 1
                
                avg_loss = total_loss / max(1, steps)
                logger.info(f"Epoch {epoch+1}/{self.proxy_epochs} | Train Loss: {avg_loss:.4f}")

            # Validation (Simple Accuracy/Loss based reward)
            # For proxy, we use 1 / (1 + val_loss) as a simple reward
            val_loss = 0
            steps = 0
            full_model.eval()
            with torch.no_grad():
                if val_loader:
                    for batch in val_loader:
                        v = batch['visual'].to(self.device)
                        t = batch['text'].to(self.device)
                        target = batch['label'].to(self.device)
                        if target.dim() == 3:
                            target = target.mean(dim=1)
                            
                        output = full_model(v, t)
                        loss = criterion(output, target)
                        val_loss += loss.item()
                        steps += 1
            
            avg_val_loss = val_loss / max(1, steps) if steps > 0 else avg_loss
            logger.info(f"Validation Loss: {avg_val_loss:.4f}")
            
            # Calculate Reward
            param_count = sum(p.numel() for p in model.parameters())
            reward = (1.0 / (1.0 + avg_val_loss)) - (1e-7 * param_count)
            return max(0.0, reward)

        except RuntimeError as e:
            logger.error(f"Training Error: {e}")
            return -1.0

    def evaluate(self, code_str: str) -> float:
        """
        主评估入口。
        """
        logger.info("Starting Evaluation...")
        
        # 1. Load Module
        fusion_module = self.safe_load_module(code_str)
        if fusion_module is None:
            return -1.0 # Failed to load
        
        # 2. Run Task
        if self.dummy_mode:
            reward = self._run_dummy_proxy_task(fusion_module)
        else:
            if self.dataset_path:
                reward = self._run_real_proxy_task(fusion_module)
            else:
                logger.warning("No dataset path provided for real mode. Falling back to dummy.")
                reward = self._run_dummy_proxy_task(fusion_module)
            
        logger.info(f"Evaluation Complete. Reward: {reward:.4f}")
        return reward

if __name__ == "__main__":
    # Test Block
    print("=== Testing AutoFusionEvaluator ===")
    
    # 1. Create a dummy fusion code
    dummy_code = """
import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, dim=768, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(dim * 2, dim)
        self.act = nn.ReLU()
    
    def forward(self, v, t):
        # Simple concat fusion
        combined = torch.cat([v, t], dim=-1)
        return self.act(self.fc(combined))
"""
    
    evaluator = AutoFusionEvaluator(dummy_mode=True)
    score = evaluator.evaluate(dummy_code)
    print(f"Test Score: {score}")
