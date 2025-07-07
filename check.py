import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Get API key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Check if API key exists
if not GROQ_API_KEY:
    print("❌ GROQ_API_KEY not found in environment variables")
    print("Make sure you have GROQ_API_KEY=your_key in your .env file")
else:
    print(f"✅ API Key found: {GROQ_API_KEY[:20]}...")
    
    # Test the API key
    try:
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=50,
        )
        
        # Simple test
        response = llm.invoke("Hello, just testing the API")
        print("✅ API Key is ACTIVE - Connection successful!")
        print(f"Test response: {response.content}")
        
    except Exception as e:
        print("❌ API Key is INACTIVE or Invalid")
        print(f"Error: {str(e)}")
        print("\nPossible solutions:")
        print("1. Check if your API key is correct")
        print("2. Make sure you have credits/quota left")
        print("3. Visit https://console.groq.com/ to verify your account")