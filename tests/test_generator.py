"""
Filename: test_generator.py
Description: Unit tests for AutoFusionGenerator parsing and mock logic.
"""

import unittest
from src.generator import AutoFusionGenerator

class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = AutoFusionGenerator(api_key="dummy")

    def test_mock_generation(self):
        """Test if mock mode returns valid structure and code."""
        result = self.generator.generate("prompt", mock_mode=True)
        self.assertIn('thought', result)
        self.assertIn('code', result)
        self.assertIn('class FusionModule', result['code'])
        self.assertIn('def forward', result['code'])

    def test_parse_output_standard(self):
        """Test parsing standard Thought + Code format."""
        raw = """
        Thought: Since we need cross-modal interaction, I will use Cross-Attention.
        Here is the code:
        ```python
        import torch
        class FusionModule(nn.Module):
            pass
        ```
        """
        parsed = self.generator._parse_output(raw)
        self.assertIn("Cross-Attention", parsed['thought'])
        self.assertIn("class FusionModule", parsed['code'])

    def test_parse_output_no_marker(self):
        """Test parsing when 'Thought:' marker is missing."""
        raw = """
        I decided to use a simple linear layer.
        ```
        class FusionModule(nn.Module):
            pass
        ```
        """
        parsed = self.generator._parse_output(raw)
        self.assertIn("simple linear layer", parsed['thought'])
        self.assertIn("class FusionModule", parsed['code'])

    def test_parse_output_no_code(self):
        """Test parsing when no code block is present."""
        raw = "I am sorry I cannot generate code."
        parsed = self.generator._parse_output(raw)
        self.assertEqual(parsed['code'], "")
        self.assertIn("cannot generate code", parsed['thought'])

if __name__ == "__main__":
    unittest.main()
