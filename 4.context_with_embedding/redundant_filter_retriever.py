from langchain.embeddings.base import Embeddings
from langchain_chroma import Chroma
from langchain.schema import BaseRetriever, Document

class RedundantFilterRetriever(BaseRetriever):
    # Let the embedding model be passed in, so it can use many different embedding models
    embeddings: Embeddings
    chroma: Chroma

    def get_relevant_documents(self, query: str) -> list[Document]:
        # Calculate embedding for the 'query' string
        emb = self.embeddings.embed_query(query)

        # Take embeddings and feed them into that 
        # max_marginal_relevance_search_by_vector
        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb,
            lambda_mult=0.8
        )
    

    async def aget_relevant_documents(self, query: str) -> list[Document]:
        
        return []
    
