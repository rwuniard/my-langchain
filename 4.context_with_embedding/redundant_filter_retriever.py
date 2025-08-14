from langchain.embeddings.base import Embeddings
from langchain_chroma import Chroma
from langchain.schema import BaseRetriever, Document

class RedundantFilterRetriever(BaseRetriever):
    # Let the embedding model be passed in, so it can use many different embedding models
    embeddings: Embeddings
    chroma: Chroma

    def get_relevant_documents(self, query: str) -> list[Document]:
        # Use the standard MMR search method that works across vectorstore implementations
        # This is more portable than using Chroma-specific max_marginal_relevance_search_by_vector
        return self.chroma.max_marginal_relevance_search(
            query=query,
            lambda_mult=0.8,
            k=4  # Number of documents to return
        )
    

    async def aget_relevant_documents(self, query: str) -> list[Document]:
        
        return []
    
