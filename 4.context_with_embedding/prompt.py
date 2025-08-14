from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.globals import set_debug

from redundant_filter_retriever import RedundantFilterRetriever

set_debug(True)

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
# The RetrievalQA is created as a facade to the vectorstore and the LLM
# Each vectorstore can have different functions to search the vectorstore
# So, the retriever will take an object that has get_relevant_documents function
# In our case, Chroma returns a ChromaRetriever object that has get_relevant_documents function.
# 
# The chain type can be "stuff", "map_reduce", "map_rerank", "map_rerank_then_stuff"
# The chain stuff is putting the document into the prompt template and passing it on to the LLM
# The chain map_reduce:
#   - It will be getting the n documents from RAG (k=n)
#   - then from each document it will append the result with user message and pass it to LLM to find the best answer.
#   - It will get n best answer from each document
#   - It will pass the n best answer to another prompt and passing it with user question on to the LLM,
#   - This will be the final answer.
# The chain map_rerank:
#   - Similar steps as map_reduce, but it will give a score to each document for the first iteration.
#   - The highest score document will be the final answer.
# The chain map_rerank_then_stuff is using the rerank_prompt to rerank the documents and then put the documents into the prompt template
# and passing it on to the LLM
# The refine chain:
#   - It does it in a series. 
#   - It feed the first result from vector database and feed both the result with the human question to LLM. 
#   - After it gets the result, it will feed the result from prior result in the series with the second result 
#     from the vector database to LLM again to get another result.
#   - It will keep doing it until all the result from vector database is exercised, then that will be the final result.
#   - The final result can be multiple results from the previous result in the series.

def load_retrieval_qa_chain(model_vendor: ModelVendor):
    llm = load_llm(model_vendor)
    vectorstore = load_vectorstore(model_vendor)
    # This redundant filter retriever is a retriever that will filter out duplicate documents from the vectorstore.
    # This can happen when the store_embeddings.py is run multiple times with the same source, or may be the source
    # can have multiple documents with the same content.
    redundant_filter_retriever = RedundantFilterRetriever(
        embeddings=load_embedding_model(model_vendor),
        chroma=vectorstore
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        # chain_type="map_reduce",
        chain_type="stuff",
        # chain_type="refine",
        # retriever=vectorstore.as_retriever(k=4),
        retriever=redundant_filter_retriever,
        verbose=True
    )

def main():
    retrieval_qa_chain = load_retrieval_qa_chain(ModelVendor.GOOGLE)
    
    print("RAG Question-Answering System")
    print("-" * 30)
    
    while True:
        user_question = input("\nEnter your question (or 'quit' to exit): ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not user_question:
            print("Please enter a valid question.")
            continue
            
        try:
            result = retrieval_qa_chain.invoke(user_question)
            print(f"\nAI answer: {result['result']}")
        except Exception as e:
            print(f"Error processing question: {e}")

if __name__ == "__main__":
    main()