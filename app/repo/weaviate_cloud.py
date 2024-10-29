import weaviate
from weaviate.classes.init import Auth
from app.config import env

class WeaviateClient:
    def __init__(self, url, api_key):
        self.url = url
        self.api_key = api_key
        self.client = None

    def connect(self):
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

if __name__ == "__main__":
    weaviate_url = env.weaviate_url
    weaviate_api_key = env.weaviate_api_key

    client = WeaviateClient(weaviate_url, weaviate_api_key)
    client.connect()
    print(client.is_ready())
    client.close()