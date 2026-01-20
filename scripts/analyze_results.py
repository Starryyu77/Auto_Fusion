"""
Filename: analyze_results.py
Description: Parses search.log to extract and visualize the evolutionary trajectory.
Usage: python scripts/analyze_results.py
"""

import re
import matplotlib.pyplot as plt
import os

def analyze_log(log_path="search.log"):
    if not os.path.exists(log_path):
        print(f"Log file {log_path} not found.")
        return

    rewards = []
    generations = []
    
    with open(log_path, 'r') as f:
        content = f.read()
        
    # Regex to extract rewards
    # Pattern: Evaluation Complete. Reward: 0.7691
    matches = re.finditer(r"Evaluation Complete\. Reward: (-?\d+\.\d+)", content)
    
    for i, match in enumerate(matches):
        reward = float(match.group(1))
        # Filter out error rewards (-1.0) for better visualization, or keep them to show failures
        rewards.append(reward)
        generations.append(i + 1)
        
    if not rewards:
        print("No rewards found in log.")
        return

    print(f"Found {len(rewards)} evaluations.")
    print(f"Max Reward: {max(rewards):.4f}")
    
    # Save Summary
    with open("results_summary.md", "w") as f:
        f.write("# Auto-Fusion Search Results\n\n")
        f.write(f"- **Total Generations**: {len(rewards)}\n")
        f.write(f"- **Max Reward**: {max(rewards):.4f}\n")
        f.write(f"- **Final Reward**: {rewards[-1]:.4f}\n\n")
        f.write("## Trajectory\n")
        for g, r in zip(generations, rewards):
            f.write(f"- Gen {g}: {r:.4f}\n")
            
    print("✅ Results summary saved to results_summary.md")

if __name__ == "__main__":
    analyze_log()
