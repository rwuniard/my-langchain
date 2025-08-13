from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from enum import Enum
import os

load_dotenv()

class ModelVendor (Enum):
    OPENAI = "openai"
    GOOGLE = "google"

def load_documents(file_path):
    loader = TextLoader(file_path)
    return loader.load()

def load_generative_ai_model(model_vendor: ModelVendor):
    if model_vendor == ModelVendor.OPENAI:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    elif model_vendor == ModelVendor.GOOGLE:
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    else:
        raise ValueError(f"Unsupported model vendor: {model_vendor}")


def main():
    print("Hello from 4-context-with-embedding!")
    llm = load_generative_ai_model(ModelVendor.GOOGLE)
    # result = llm.invoke("What is the capital of France?")
    # print("Raw result: ", result)
    # print("Result: ", result.content)

    fact_doc = load_documents("facts.txt")
    print("Fact doc: ", fact_doc)



if __name__ == "__main__":
    main()
