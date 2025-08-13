from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
from enum import Enum

load_dotenv()

class ModelVendor (Enum):
    OPENAI = "openai"
    GOOGLE = "google"

def load_embedding_model(model_vendor: ModelVendor):
    if model_vendor == ModelVendor.OPENAI:
        return OpenAIEmbeddings()
    elif model_vendor == ModelVendor.GOOGLE:
        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"
        )
    else:
        raise ValueError(f"Unsupported model vendor: {model_vendor}")
    
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
    
def load_llm(model_vendor: ModelVendor):
    if model_vendor == ModelVendor.OPENAI:
        return OpenAI(
            model="gpt-4o-mini",
            temperature=0
        )
    elif model_vendor == ModelVendor.GOOGLE:
        return GoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0
        )
    else:
        raise ValueError(f"Unsupported model vendor: {model_vendor}")
    
# We are going to use the RetrievalQA chain to answer questions based on the context
# The RetrievalQA chain is a chain that takes a query and a vectorstore and returns an answer
# This is similar to create a ChatPromptTemplate that has
# - A system message
# - A user message
# and pass it on to the LLM
# The RetrievalQA chain will use the vectorstore to find the most relevant documents
# and pass them on to the LLM
# The LLM will then use the documents to answer the question
# The RetrievalQA chain will return the answer

def load_retrieval_qa(model_vendor: ModelVendor):
    llm = load_llm(model_vendor)
    vectorstore = load_vectorstore(model_vendor)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

def main():
    retrieval_qa = load_retrieval_qa(ModelVendor.GOOGLE)
    result = retrieval_qa.invoke("how many steps are there in Eiffel Tower?")
    print("AI answer: ", result["result"])

if __name__ == "__main__":
    main()