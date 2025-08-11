#!/usr/bin/env python3
"""
Test script to verify environment loading for Streamlit app
"""

import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_environment_loading():
    """Test environment variable loading."""
    print("🧪 Testing Environment Loading for Streamlit App")
    print("=" * 50)
    
    # Test 1: Load .env file
    print("1. Loading .env file...")
    load_dotenv()
    
    # Test 2: Check if .env file exists
    env_file_exists = os.path.exists(".env")
    print(f"   .env file exists: {env_file_exists}")
    
    # Test 3: Check API key
    google_api_key = os.getenv("GOOGLE_API_KEY")
    api_key_available = bool(google_api_key)
    print(f"   Google API key available: {api_key_available}")
    
    if api_key_available:
        print(f"   API key (first 10 chars): {google_api_key[:10]}...")
    else:
        print("   No API key found")
    
    # Test 4: Check other environment variables
    gemini_model = os.getenv("GEMINI_MODEL", "not_set")
    print(f"   Gemini model: {gemini_model}")
    
    # Test 5: Simulate the check_environment_setup function
    def check_environment_setup():
        google_api_key = os.getenv("GOOGLE_API_KEY")
        env_file_exists = os.path.exists(".env")
        
        return {
            "api_key_available": bool(google_api_key),
            "env_file_exists": env_file_exists,
            "setup_instructions": not env_file_exists
        }
    
    env_status = check_environment_setup()
    print("\n2. Environment Status:")
    print(f"   API key available: {env_status['api_key_available']}")
    print(f"   .env file exists: {env_status['env_file_exists']}")
    print(f"   Setup instructions needed: {env_status['setup_instructions']}")
    
    # Test 6: Provide recommendations
    print("\n3. Recommendations:")
    if env_status["setup_instructions"]:
        print("   ❌ Create .env file:")
        print("      cp env.example .env")
        print("      # Then add your Google AI API key")
    elif not env_status["api_key_available"]:
        print("   ⚠️  Add API key to .env file:")
        print("      GOOGLE_API_KEY=your_actual_api_key_here")
    else:
        print("   ✅ Environment properly configured!")
    
    print("\n✅ Environment loading test completed!")

if __name__ == "__main__":
    test_environment_loading() 