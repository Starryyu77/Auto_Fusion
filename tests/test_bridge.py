import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bridge import rl_to_llm_bridge

def test_bridge():
    print("Testing Bridge...")
    
    history = [
        {'code': 'def forward(self, v, t): return v + t'},
        {'code': 'def forward(self, v, t): return v * t'}
    ]
    
    # Test Mutation High
    action_mut_high = {"op_type": "MUTATION", "intensity": 0.9}
    prompt = rl_to_llm_bridge(action_mut_high, history)
    print(f"[Mutation High Prompt Start]\n{prompt[:100]}...\n")
    assert "MUTATION" in prompt
    assert "MAJOR structural changes" in prompt
    
    # Test Crossover
    action_cross = {"op_type": "CROSSOVER", "intensity": 0.5}
    prompt = rl_to_llm_bridge(action_cross, history)
    print(f"[Crossover Prompt Start]\n{prompt[:100]}...\n")
    assert "CROSSOVER" in prompt
    assert "Parent 2" in prompt
    
    print("Bridge Test Passed!")

if __name__ == "__main__":
    test_bridge()
