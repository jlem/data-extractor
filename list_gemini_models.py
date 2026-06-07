import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load key from .env file if it exists
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ Error: No API key found in your environment or .env file.")
    print("Please set your GEMINI_API_KEY in a .env file first.")
    exit(1)

try:
    genai.configure(api_key=api_key)
    
    print("Listing available models from your Gemini API Key:")
    print("=" * 50)
    
    models = list(genai.list_models())
    if not models:
        print("No models returned. Your key might be restricted or inactive.")
    else:
        for m in models:
            # Format model details
            supported_methods = ", ".join(m.supported_generation_methods)
            print(f"- {m.name} (Methods: {supported_methods})")
            
    print("=" * 50)
except Exception as e:
    print(f"❌ Failed to query Gemini API: {str(e)}")
