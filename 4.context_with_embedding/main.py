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

# This is an example on how to use the embedding model to store the documents to the vectorstore 
# and search the vectorstore for the most similar documents to the query
# See the search_similarity.py and store_embeddings.py for a more modular approach between storing 
# and searching

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

def store_to_chroma(
        documents: list[Document], 
        embedding_model: Embeddings) -> Chroma:
    if isinstance(embedding_model, OpenAIEmbeddings):
        persist_directory = "chroma_db_openai"
    elif isinstance(embedding_model, GoogleGenerativeAIEmbeddings):
        persist_directory = "chroma_db_google"
    else:
        raise ValueError(f"Unsupported embedding model: {embedding_model}")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    return vectorstore

def main():
    print("Hello from 4-context-with-embedding!")
    llm = load_generative_ai_model(ModelVendor.GOOGLE)
    # result = llm.invoke("What is the capital of France?")
    # print("Raw result: ", result)
    # print("Result: ", result.content)

    fact_doc = load_documents("facts.txt")
    # for doc in fact_doc:
    #     print("Doc: ", doc.page_content)
    #     print("--------------------------------")

    # Initialize the embedding model
    embedding_model = load_embedding_model(ModelVendor.GOOGLE)
    emb = embedding_model.embed_query("What is the capital of France?")
    print("Embedding length: ", len(emb))
    # print("Embedding: ", emb)

    # Store the documents to Chroma
    vectorstore = store_to_chroma(fact_doc, embedding_model)
    print("Vectorstore: ", vectorstore)

    # Search the vectorstore
    results = vectorstore.similarity_search_with_score(
        "What is interesting fact about the English language?",
        k=3 # Returning top 3 results with score in descending order, so we can later use the score to rank the results
    )
    print("Results: ", results)
    for result in results:
        print("Result: ", result[1])
        print("Result: ", result[0].page_content)
        print("--------------------------------")


if __name__ == "__main__":
    main()
