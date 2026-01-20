"""
Filename: test_evaluator.py
Description: Unit tests for AutoFusionEvaluator.
"""

import unittest
import torch
from src.evaluator import AutoFusionEvaluator

class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = AutoFusionEvaluator(dummy_mode=True, proxy_epochs=1)
        
    def test_valid_code(self):
        code = """
import torch
import torch.nn as nn
class SimpleFusion(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.layer = nn.Linear(dim, dim)
    def forward(self, v, t):
        return self.layer(t)
"""
        reward = self.evaluator.evaluate(code)
        self.assertTrue(reward > 0, "Valid code should return positive reward")

    def test_syntax_error(self):
        code = """
import torch
class BadCode
    def __init__(self): pass
"""
        reward = self.evaluator.evaluate(code)
        self.assertEqual(reward, -1.0, "Syntax error should return -1.0")
        
    def test_runtime_error(self):
        # Dimension mismatch code
        code = """
import torch
import torch.nn as nn
class CrashFusion(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.layer = nn.Linear(10, 10) # Wrong dim
    def forward(self, v, t):
        return self.layer(t) # Will crash on shape
"""
        reward = self.evaluator.evaluate(code)
        self.assertEqual(reward, -1.0, "Runtime error should return -1.0")

    def test_security_check(self):
        code = """
import os
class MaliciousFusion(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        os.system('echo hacked')
"""
        reward = self.evaluator.evaluate(code)
        self.assertEqual(reward, -1.0, "Malicious code should be blocked")

if __name__ == "__main__":
    unittest.main()
