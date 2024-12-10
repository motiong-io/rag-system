from motiongreactor.context_providers.base_context_provider import BaseContextProvider
import asyncio
import weaviate
from weaviate.classes.init import Auth

class WeaviateContextProvider(BaseContextProvider):
    def __init__(self,weaviate_url,weaviate_key,collection_name:str) -> None:
        self.weaviate_client = weaviate.connect_to_weaviate_cloud(
                cluster_url=weaviate_url,                       
                auth_credentials=Auth.api_key(weaviate_key),
            )
        self.collection=self.weaviate_client.collections.get(collection_name)

    def hybrid_search(self, query_vector: list[float], query:str):
        response = self.collection.query.hybrid(
            query=query,
            vector=query_vector,
            alpha=0.8,
            limit=30
        )
        print(response)
        return response

    def close(self):
        self.weaviate_client.close()

    async def provide_contexts(self, query_vector: list[float], query:str) -> list[str]:
        print("Querying weaviate...")
        tasks = [
            asyncio.to_thread(self.hybrid_search,query_vector,query),
        ]
        results = await asyncio.gather(*[task for task in tasks])
        print(results)
        return results
    


def test_weaviate_context_provider():
    import os
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_key = os.getenv("WEAVIATE_KEY")
    query = "If my future wife has the same first name as the 15th first lady of the United States' mother and her surname is the same as the second assassinated president's mother's maiden name, what is my future wife's name?"
    