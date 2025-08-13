from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv()

# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def main():
    print("Hello from 4-context-with-embedding!")
    result = llm.invoke("What is the capital of France?")
    print("Raw result: ", result)
    print("Result: ", result.content)



if __name__ == "__main__":
    main()
