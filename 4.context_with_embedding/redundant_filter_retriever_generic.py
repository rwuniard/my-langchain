from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever, Document
from langchain.vectorstores.base import VectorStore

class RedundantFilterRetriever(BaseRetriever):
    """
    A generic retriever that works with any VectorStore implementation.
    Uses Maximum Marginal Relevance (MMR) to filter out duplicate/redundant content.
    
    This retriever is vectorstore-agnostic and works with:
    - Chroma
    - Pinecone
    - FAISS
    - Qdrant
    - Any other VectorStore that implements as_retriever() with MMR support
    """
    
    # Accept any VectorStore implementation, not just Chroma
    embeddings: Embeddings  # Kept for backward compatibility but not directly used
    vectorstore: VectorStore

    def get_relevant_documents(self, query: str) -> list[Document]:
        """
        Retrieve relevant documents using Maximum Marginal Relevance (MMR).
        
        This approach uses the standard LangChain retriever interface which works
        across different vectorstore implementations (Chroma, Pinecone, FAISS, etc.)
        
        Args:
            query: The search query string
            
        Returns:
            List of Document objects with diverse, non-redundant content
        """
        # Use the standard MMR retriever approach that works with any vectorstore
        # This is the most portable way to get MMR functionality across implementations
        mmr_retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "lambda_mult": 0.8,  # Controls diversity vs relevance (0=max diversity, 1=max relevance)
                "k": 4  # Number of documents to return
            }
        )
        return mmr_retriever.invoke(query)
    
    async def aget_relevant_documents(self, query: str) -> list[Document]:
        """
        Async version of get_relevant_documents.
        
        Args:
            query: The search query string
            
        Returns:
            List of Document objects with diverse, non-redundant content
        """
        # Use the async retriever interface
        mmr_retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "lambda_mult": 0.8,
                "k": 4
            }
        )
        return await mmr_retriever.ainvoke(query)