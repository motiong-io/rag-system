from motiongreactor.context_providers.base_context_provider import BaseContextProvider
import asyncio


from app.repo.weaviate_cloud import WeaviateClient


class WeaviateContextProvider(BaseContextProvider):

    async def weaviate_contexts(self, query:str, query_vector: list[float]) -> list[str]:
        weaviate_client = WeaviateClient("ContextualVectors")
        response = weaviate_client.hybrid_search(query, query_vector, 30)
        return response



    async def provide_contexts(self, query_vector: list[float]) -> list[str]:
        tasks = [
            asyncio.sleep(5),
            asyncio.sleep(5),
        ]
        results = await asyncio.gather(*[task for task in tasks])
        return []