import os
from dotenv import load_dotenv
from langchain_openai import OpenAI

# Load environment variables from .env file
load_dotenv()

def main():
    print("Hello from my-first-langchain!")
    
    # Check if OpenAI API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OpenAI API key not found!")
        print("Please set your OPENAI_API_KEY environment variable:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    try:
        llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.2)
        response = llm.invoke("What is the capital of France?")
        print(f"🤖 AI Response: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
