from typing import List
import weaviate
from weaviate.classes.init import Auth
from app.config import env

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
        

        
            

if __name__ == "__main__":
    client = WeaviateClient()
    print(client.is_ready())
    client.close()