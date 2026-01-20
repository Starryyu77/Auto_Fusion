"""
Filename: verify_gemini.py
Description: Tests the connection to Google Gemini API.
             Sends a test prompt and prints the response and token usage.
Usage: GOOGLE_API_KEY=... python scripts/verify_gemini.py
"""

import os
import logging
from src.generator import AutoFusionGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def verify_connection():
    print("=== ♊️ Verifying Google Gemini API Connection ===")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found in environment variables.")
        return

    try:
        # Initialize Generator in Real Mode
        # Use gemini-1.5-pro for testing if flash preview is not available or desired
        # or stick to user requested model
        model_name = "gemini-3-flash-preview" 
        print(f"Initializing Generator with model: {model_name}...")
        
        generator = AutoFusionGenerator(
            model_name=model_name,
            api_key=api_key, 
            mock_mode=False
        )
        
        # Test Prompt
        prompt = (
            "Write a simple PyTorch class 'FusionLayer' that takes visual (dim=2048) "
            "and text (dim=768) features, projects them to 512, and adds them. "
            "Include 'Thought:' before the code."
        )
        
        print(f"\n📤 Sending Test Prompt:\n{prompt}\n")
        print("⏳ Waiting for response (this may take a few seconds)...")
        
        result = generator.generate(prompt)
        
        print("\n✅ Response Received!")
        print("-" * 40)
        print(f"🧠 Thought:\n{result['thought'][:200]}...") # Truncate for display
        print("-" * 40)
        print(f"💻 Code:\n{result['code']}")
        print("-" * 40)
        
        print("\n🎉 Gemini API Connection Verified Successfully!")

    except Exception as e:
        print(f"\n❌ Verification Failed: {e}")

if __name__ == "__main__":
    verify_connection()
