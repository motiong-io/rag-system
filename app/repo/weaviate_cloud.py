from typing import List
import weaviate
from weaviate.classes.init import Auth
from app.config import env
from app.model.embeddings_model import Embeddings
from tqdm import tqdm
from app.utils.md5hash import md5hash
from weaviate.classes.query import MetadataQuery

class WeaviateClient:
    def __init__(self):
        self.url = env.weaviate_url
        self.api_key = env.weaviate_api_key
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=self.url,
            auth_credentials=Auth.api_key(self.api_key),
        )

    def is_ready(self):
        if self.client:
            return self.client.is_ready()
        return False

    def close(self):
        if self.client:
            self.client.close()

    def create_object(self, collection_name:str, properties:dict,vector:List[float]):
        if self.client:
            collection = self.client.collections.get(collection_name)
            uuid = collection.data.insert(
            properties=properties,
            vector=vector
            )
            return uuid
        else:
            raise Exception("Client not connected")
        
    def batch_import(self, collection_name:str, embeddings:Embeddings):
        if self.client:
            collection = self.client.collections.get(collection_name)
            with collection.batch.dynamic() as batch:
                for i, data_row in enumerate(tqdm(embeddings.embeddings, desc="Batch Importing")):
                    batch.add_object(
                        uuid=md5hash(data_row.properties.get('text_to_embed')),
                        properties=data_row.properties,
                        vector=data_row.vector,
                    )
        else:
            raise Exception("Client not connected")
        
    def vector_search(self, collection_name:str, query_vector:List[float], k:int):
        if self.client:
            collection = self.client.collections.get(collection_name)
            response = collection.query.near_vector(
                near_vector=query_vector,
                limit=k,
                return_metadata=MetadataQuery(distance=True)
            )
            return response
        else:
            raise Exception("Client not connected")
        

def test_bach_import():
    embeddings = Embeddings.load_json("assets/dataset/embeddings_list/417e546b48ce6e74b37c0815920013dc.json")
    client = WeaviateClient()
    client.batch_import("ContextualVectors", embeddings)
    client.close()

if __name__ == "__main__":
    test_bach_import()