from typing import List
import weaviate
from weaviate.classes.init import Auth
from app.config import env
from app.model.embeddings_model import Embeddings
from tqdm import tqdm
from app.utils.md5hash import md5hash
from weaviate.classes.query import MetadataQuery

class WeaviateClient:
    def __init__(self,collection_name:str="ContextualVectors") -> None:
        self.url = env.weaviate_url
        self.api_key = env.weaviate_api_key
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=self.url,
            auth_credentials=Auth.api_key(self.api_key),
        )
        self.collection_name = collection_name

    def is_ready(self):
        if self.client:
            return self.client.is_ready()
        return False

    def close(self):
        if self.client:
            self.client.close()

    def create_object(self, properties:dict,vector:List[float]):
        if self.client:
            collection = self.client.collections.get(self.collection_name)
            uuid = collection.data.insert(
            properties=properties,
            vector=vector
            )
            return uuid
        else:
            raise Exception("Client not connected")
        
    def batch_import(self, embeddings:Embeddings):
        if self.client:
            collection = self.client.collections.get(self.collection_name)
            with collection.batch.dynamic() as batch:
                for i, data_row in enumerate(tqdm(embeddings.embeddings, desc="Batch Importing")):
                    batch.add_object(
                        uuid=md5hash(data_row.properties.get('text_to_embed')),
                        properties=data_row.properties if isinstance(data_row.properties, dict) else data_row.properties.to_dict(),
                        vector=data_row.vector,
                    )
                    # print(collection.batch.failed_objects)
        else:
            raise Exception("Client not connected")
        
    def vector_search(self, query_vector:List[float], k:int):
        if self.client:
            collection = self.client.collections.get(self.collection_name)
            response = collection.query.near_vector(
                near_vector=query_vector,
                limit=k,
                return_metadata=MetadataQuery(distance=True)
            )
            return response
        else:
            raise Exception("Client not connected")
        
    def hybrid_search(self, query:str, query_vector:List[float], k:int):
        if self.client:
            collection = self.client.collections.get(self.collection_name)
            response = collection.query.hybrid(
                query=query,
                vector=query_vector,
                alpha=0.8,
                limit=k,
                # return_metadata=MetadataQuery(distance=True)
            )
            return response
        else:
            raise Exception("Client not connected")
        

def test_bach_import():
    embeddings = Embeddings.load_json("assets/dataset/embeddings_list/1ddadbaf23ada8730ff72097d7101243.json")
    client = WeaviateClient("ContextualVectors")
    client.batch_import(embeddings)
    client.close()

if __name__ == "__main__":
    test_bach_import()