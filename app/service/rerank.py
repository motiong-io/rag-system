from typing import List
from app.utils.x_client import Client

class RerankService:
    def __init__(self, client_url, model_name):
        self.client = Client(client_url)
        self.model = self.client.get_model(model_name)

    def rerank(self, corpus:List[str], query:str):
        return self.model.rerank(corpus, query)

if __name__ == "__main__":
    client_url = "http://10.4.32.1:9997"
    model_name = "bge-reranker-v2-m3"
    query = "A man is eating pasta."
    corpus = [
        "A man is eating food.",
        "A man is eating a piece of bread.",
        "The girl is carrying a baby.",
        "A man is riding a horse.",
        "A woman is playing violin."
    ]

    rerank_service = RerankService(client_url, model_name)
    print(rerank_service.rerank(corpus, query))
