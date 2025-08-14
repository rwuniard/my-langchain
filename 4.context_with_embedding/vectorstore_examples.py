"""
Example showing how the generic RedundantFilterRetriever works with different vectorstores.
This demonstrates the portability of the generic implementation.
"""

from redundant_filter_retriever_generic import RedundantFilterRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Example documents for testing
sample_docs = [
    Document(page_content="The Earth is the third planet from the Sun."),
    Document(page_content="Earth is the third planet in our solar system."),  # Similar to first
    Document(page_content="Mars is the fourth planet from the Sun."),
    Document(page_content="The red planet Mars follows Earth in our solar system."),  # Similar to third
    Document(page_content="Jupiter is the largest planet in the solar system."),
]

def example_with_chroma():
    """Example using Chroma vectorstore"""
    from langchain_chroma import Chroma
    
    print("=== Example with Chroma ===")
    
    # Create Chroma vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(sample_docs, embeddings)
    
    # Create generic retriever
    retriever = RedundantFilterRetriever(
        embeddings=embeddings,
        vectorstore=vectorstore  # Any VectorStore works here
    )
    
    # Test retrieval
    query = "Tell me about planets"
    results = retriever.get_relevant_documents(query)
    
    print(f"Query: {query}")
    print(f"Retrieved {len(results)} documents:")
    for i, doc in enumerate(results):
        print(f"  {i+1}. {doc.page_content}")
    print()

def example_with_faiss():
    """Example using FAISS vectorstore"""
    from langchain_community.vectorstores import FAISS
    
    print("=== Example with FAISS ===")
    
    # Create FAISS vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(sample_docs, embeddings)
    
    # Create generic retriever - same code as Chroma!
    retriever = RedundantFilterRetriever(
        embeddings=embeddings,
        vectorstore=vectorstore  # Different vectorstore, same retriever code
    )
    
    # Test retrieval
    query = "What is Mars like?"
    results = retriever.get_relevant_documents(query)
    
    print(f"Query: {query}")
    print(f"Retrieved {len(results)} documents:")
    for i, doc in enumerate(results):
        print(f"  {i+1}. {doc.page_content}")
    print()

def example_with_pinecone():
    """Example showing how it would work with Pinecone (requires setup)"""
    print("=== Example with Pinecone (Conceptual) ===")
    print("# This is how you would use it with Pinecone:")
    print("""
from langchain_pinecone import PineconeVectorStore

# Create Pinecone vectorstore (requires API key and index setup)
vectorstore = PineconeVectorStore.from_documents(
    sample_docs, 
    embeddings, 
    index_name="your-index"
)

# Create generic retriever - same code as other vectorstores!
retriever = RedundantFilterRetriever(
    embeddings=embeddings,
    vectorstore=vectorstore  # Works with Pinecone too
)

# Use exactly the same way
results = retriever.get_relevant_documents("your query")
""")
    print()

def demonstrate_portability():
    """Show the key benefit: same retriever code works everywhere"""
    print("=== Key Benefit: Portability ===")
    print("The same RedundantFilterRetriever code works with:")
    print("✅ Chroma")
    print("✅ FAISS") 
    print("✅ Pinecone")
    print("✅ Qdrant")
    print("✅ Any VectorStore that implements as_retriever() with MMR support")
    print()
    print("No need for vectorstore-specific implementations!")
    print("Just change the vectorstore, keep the same retriever code.")
    print()

if __name__ == "__main__":
    # Note: These examples require OpenAI API key for embeddings
    # You can run them individually if you have the required dependencies
    
    demonstrate_portability()
    
    # Uncomment to run actual examples (requires API keys and dependencies):
    # example_with_chroma()
    # example_with_faiss()
    example_with_pinecone()  # This is just a code example, doesn't require setup