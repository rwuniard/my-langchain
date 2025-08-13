from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from enum import Enum
import os

load_dotenv()

class ModelVendor (Enum):
    OPENAI = "openai"
    GOOGLE = "google"

def load_documents(file_path):
    loader = TextLoader(file_path)
    return loader.load_and_split(
        text_splitter=get_text_splitter()
    )

def load_generative_ai_model(model_vendor: ModelVendor):
    if model_vendor == ModelVendor.OPENAI:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    elif model_vendor == ModelVendor.GOOGLE:
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    else:
        raise ValueError(f"Unsupported model vendor: {model_vendor}")

def load_embedding_model(model_vendor: ModelVendor):
    if model_vendor == ModelVendor.OPENAI:
        return OpenAIEmbeddings()
    elif model_vendor == ModelVendor.GOOGLE:
        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"
        )
    else:
        raise ValueError(f"Unsupported model vendor: {model_vendor}")

def get_text_splitter():
    return CharacterTextSplitter(
        separator="\n",
        chunk_size=200, 
        chunk_overlap=0
    )

def main():
    print("Hello from 4-context-with-embedding!")
    llm = load_generative_ai_model(ModelVendor.GOOGLE)
    # result = llm.invoke("What is the capital of France?")
    # print("Raw result: ", result)
    # print("Result: ", result.content)

    fact_doc = load_documents("facts.txt")
    for doc in fact_doc:
        print("Doc: ", doc.page_content)
        print("--------------------------------")

    embedding_model = load_embedding_model(ModelVendor.GOOGLE)
    emb = embedding_model.embed_query("What is the capital of France?")
    print("Embedding: ", emb)


if __name__ == "__main__":
    main()
