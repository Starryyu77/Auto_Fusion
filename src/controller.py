"""
Filename: controller.py
Description: RL Controller based on A2C for guiding Neural Architecture Search.
Module: AutoFusion.Controller
Author: Auto-Fusion Agent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

class AutoFusionController(nn.Module):
    """
    Advantage Actor-Critic (A2C) Controller for Auto-Fusion.
    
    Inputs:
        state: Tensor [val_acc, param_cost, best_acc_history]
    Outputs:
        op_prob: Probability distribution over operations
        intensity: Continuous value [0, 1] for mutation intensity
        value: State value estimate V(s)
    """
    def __init__(self, input_dim=3, hidden_dim=64):
        super().__init__()
        
        # Actions: 0=MUTATION, 1=CROSSOVER, 2=FRESH_START
        self.action_space = ["MUTATION", "CROSSOVER", "FRESH_START"]
        
        # Shared Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor Head 1: Operation Selection (Discrete)
        self.actor_op = nn.Linear(hidden_dim, len(self.action_space))
        
        # Actor Head 2: Intensity (Continuous)
        # We output mean and std (fixed or learned) for sampling, or just a deterministic value?
        # Protocol says: "A continuous head (Sigmoid) to output a value in [0, 1]"
        # To support RL exploration, we usually treat this as the mean of a Beta or Normal distribution, 
        # or just output the value if using a simplified approach. 
        # Given "returns... intensity" and "Sigmoid", we'll implement it as a deterministic output 
        # heavily influenced by the policy, or sample from a distribution parameterized by it.
        # For A2C, we typically need a distribution. Let's use a Normal distribution mapped to sigmoid 
        # or simply output the value and add exploration noise during training.
        # Strict adherence to prompt: "A continuous head (Sigmoid) to output a value... representing... temperature"
        self.actor_int = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Critic Head: Value Function (Scalar)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        """
        Args:
            state: [Batch, 3] or [3]
        Returns:
            op_logits: [Batch, 3]
            intensity: [Batch, 1]
            value: [Batch, 1]
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        feat = self.encoder(state)
        
        op_logits = self.actor_op(feat)
        intensity = self.actor_int(feat)
        value = self.critic(feat)
        
        return op_logits, intensity, value
    
    def select_action(self, state):
        """
        Samples an action for the environment.
        Args:
            state: [3]
        Returns:
            action_dict: {
                "op_type": str,
                "intensity": float,
                "log_prob_op": Tensor,
                "value": Tensor
            }
        """
        op_logits, intensity, value = self.forward(state)
        
        # 1. Sample Operation
        dist_op = Categorical(logits=op_logits)
        action_idx = dist_op.sample()
        op_type = self.action_space[action_idx.item()]
        
        # 2. Intensity
        # For inference/selection, we use the output directly. 
        # (In full training loop, we might add noise here)
        intensity_val = intensity.item()
        
        return {
            "op_type": op_type,
            "intensity": intensity_val,
            "log_prob_op": dist_op.log_prob(action_idx),
            "value": value,
            "raw_action_idx": action_idx.item()
        }
