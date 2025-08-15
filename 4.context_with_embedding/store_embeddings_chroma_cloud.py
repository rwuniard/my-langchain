from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_core.embeddings import Embeddings
from enum import Enum
import os
import chromadb

load_dotenv()

class ModelVendor (Enum):
    OPENAI = "openai"
    GOOGLE = "google"

def load_documents(file_path):
    loader = TextLoader(file_path)
    return loader.load_and_split(
        text_splitter=get_text_splitter()
    )

# def load_generative_ai_model(model_vendor: ModelVendor):
#     if model_vendor == ModelVendor.OPENAI:
#         return ChatOpenAI(model="gpt-4o-mini", temperature=0)
#     elif model_vendor == ModelVendor.GOOGLE:
#         return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
#     else:
#         raise ValueError(f"Unsupported model vendor: {model_vendor}")

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

def store_to_chroma(documents: list[Document], embedding_model: Embeddings) -> Chroma:
    # Determine collection name
    if isinstance(embedding_model, OpenAIEmbeddings):
        collection_name = "openai_collection"
    elif isinstance(embedding_model, GoogleGenerativeAIEmbeddings):
        collection_name = "google_collection"
    else:
        raise ValueError(f"Unsupported embedding model: {embedding_model}")

    # Create Chroma Cloud client
    chroma_client = chromadb.CloudClient(
        tenant=os.getenv("CHROMA_TENANT_ID"),
        database=os.getenv("CHROMA_DATABASE_NAME"),
        api_key=os.getenv("CHROMA_API_KEY")
    )
    
    # Create Chroma vectorstore with cloud client
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        client=chroma_client,
        collection_name=collection_name
    )
    return vectorstore

def main():
    print("Store embeddings to Chroma!")
    fact_doc = load_documents("facts.txt")

    # Initialize the embedding model
    embedding_model = load_embedding_model(ModelVendor.GOOGLE)

    # Store the documents to Chroma
    vectorstore = store_to_chroma(fact_doc, embedding_model)
    print("Vectorstore: ", vectorstore)



if __name__ == "__main__":
    main()
