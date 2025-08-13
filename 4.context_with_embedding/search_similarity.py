from langchain_google_genai import GoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

class ModelVendor (Enum):
    OPENAI = "openai"
    GOOGLE = "google"

# Load the embedding model
def load_embedding_model(model_vendor: ModelVendor):
    if model_vendor == ModelVendor.OPENAI:
        return OpenAIEmbeddings()
    elif model_vendor == ModelVendor.GOOGLE:
        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"
        )
    else:
        raise ValueError(f"Unsupported model vendor: {model_vendor}")


# Load the vectorstore from the persist directory based on the model vendor
def load_vectorstore(model_vendor: ModelVendor):
    if model_vendor == ModelVendor.OPENAI:
        return Chroma(
            embedding_function=load_embedding_model(model_vendor),
            persist_directory="chroma_db_openai"
        )
    elif model_vendor == ModelVendor.GOOGLE:
        return Chroma(
            embedding_function=load_embedding_model(model_vendor),
            persist_directory="chroma_db_google"
        )
    else:
        raise ValueError(f"Unsupported model vendor: {model_vendor}")
    
# Search the vectorstore for the most similar documents to the query
def search_similarity(query: str, vectorstore: Chroma):
    results = vectorstore.similarity_search(query, k=2)
    return results

def main():
    print("Search similarity!")
    vectorstore = load_vectorstore(ModelVendor.GOOGLE)
    results = search_similarity("What is interesting fact about Charlie?", vectorstore)
    for result in results:
        print("Result: ", result.page_content)
        print("--------------------------------")

if __name__ == "__main__":
    main()