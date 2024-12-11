from motiongreactor.embeddings.base_embedding import BaseEmbeddingService

import voyageai
from app.config import env
from typing import List

class QueryEmbedding(BaseEmbeddingService):
    def __init__(self):
        self.embed_client = voyageai.Client(api_key=env.voyage_api_key)
        self.embed_model = "voyage-3"

    def embed_text_list(self, text_list: List[str]) -> List[List[float]]:
        vector_list = self.embed_client.embed(
            texts=text_list,
            model=self.embed_model
        ).embeddings
        return vector_list
    
    def embed_text(self, text: str) -> List[float]:
        vector = self.embed_client.embed(
            texts=[text],
            model=self.embed_model
        ).embeddings[0]
        return vector
    
    def embed(self, sentences: List[str]) -> List[List[float]]:
        return self.embed_text_list(sentences)

def test_embed_text():
    text_list = ["This is a test text A"]
    embed_creator = QueryEmbedding()
    vector_list = embed_creator.embed(text_list)
    print(len(vector_list))
    print(len(vector_list[0]))




if __name__ == "__main__":
    test_embed_text()
