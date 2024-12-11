from motiongreactor.context_providers.base_context_provider import BaseContextProvider
import asyncio
import weaviate
from weaviate.classes.init import Auth
import weaviate.classes as wvc

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
            limit=50
        )
        return [obj.properties['text_to_embed'] for obj in response.objects]

    def close(self):
        self.weaviate_client.close()

    async def provide_contexts(self, query_vector: list[float], query:str) -> list[str]:
        # print("Querying weaviate...")
        if len(query_vector) == 1 and query_vector[0] :
            query_vector = query_vector[0]  

        # print(query_vector)
        # print(query)
        tasks = [
            asyncio.to_thread(self.hybrid_search,query_vector,query),
        ]
        results = await asyncio.gather(*[task for task in tasks])
        # print(results)
        return results[0]
    
    def check(self):
        try:
            collection = self.collection
            response = collection.aggregate.over_all(
                total_count=True,
                return_metrics=wvc.query.Metrics("wordCount").integer(
                    count=True,
                    maximum=True,
                    mean=True,
                    median=True,
                    minimum=True,
                    mode=True,
                    sum_=True,
                ),
            )

            print(response.total_count)
            print(response.properties)

        finally:
            self.weaviate_client.close()
    


def test_weaviate_context_provider():
    import os
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_key = os.getenv("WEAVIATE_KEY")
    query = "If my future wife has the same first name as the 15th first lady of the United States' mother and her surname is the same as the second assassinated president's mother's maiden name, what is my future wife's name?"

def check_weaviate():
    from app.config import env
    weaviate_url = env.weaviate_url
    weaviate_key = env.weaviate_api_key
    collection_name = "GPT4ominiContextualDB"
    weaviate_context_provider = WeaviateContextProvider(weaviate_url,weaviate_key,collection_name)
    weaviate_context_provider.check()

if __name__ == "__main__":
    check_weaviate()