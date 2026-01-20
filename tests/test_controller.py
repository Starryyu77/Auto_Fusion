import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.controller import AutoFusionController

def test_controller():
    print("Testing AutoFusionController...")
    
    # Init
    model = AutoFusionController(input_dim=3, hidden_dim=16)
    
    # Dummy State: [val_acc, param_cost, best_acc_history]
    state = torch.tensor([0.75, 0.5, 0.72])
    
    # Test forward
    op_logits, intensity, value = model(state)
    print(f"Op Logits: {op_logits}")
    print(f"Intensity: {intensity}")
    print(f"Value: {value}")
    
    assert op_logits.shape == (1, 3)
    assert intensity.shape == (1, 1)
    assert value.shape == (1, 1)
    assert 0 <= intensity.item() <= 1
    
    # Test select_action
    action = model.select_action(state)
    print(f"Action Dict: {action}")
    
    assert "op_type" in action
    assert action["op_type"] in ["MUTATION", "CROSSOVER", "FRESH_START"]
    assert "intensity" in action
    assert "log_prob_op" in action
    
    print("Controller Test Passed!")

if __name__ == "__main__":
    test_controller()
