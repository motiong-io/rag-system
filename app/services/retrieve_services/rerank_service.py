
from motiongreactor.rerankers.base_reranker import BaseReranker

from infinity_client import Client
from infinity_client.models import RerankInput,ReRankResult
from infinity_client.api.default import classify, embeddings, embeddings_image, rerank
from infinity_client.types import Response
from typing import List


class InfinityRerankService(BaseReranker):
    def __init__(self, client_url, model_name):
        self.client_url = client_url
        self.model_name = model_name
        self.client = Client(base_url=self.client_url)

    def rerank(self, corpus: List[str], query: str, top_n: int = None) -> list[str]:
        with Client(base_url=self.client_url) as client:
            reranked_corpus = rerank.sync(
                client=client,
                body=RerankInput.from_dict(
                    {
                        "query": query,
                        "documents": corpus,
                        "return_documents": True,
                        "model": self.model_name,
                        "top_n": top_n,
                    }
                ),
            )
            return [context.document for context in reranked_corpus.results]



import cohere
from typing import Literal

class CohereRerankService(BaseReranker):

    def __init__(self, cohere_api_key,rerank_model:Literal['rerank-english-v3.0']):
        self.client = cohere.Client(cohere_api_key)
        self.rerank_model = rerank_model

    def rerank(self, contexts:List[str], query:str,top_n:int=None) -> List[str]:
        response = self.client.rerank(
            model=self.rerank_model,
            query=query,
            documents=contexts,
            top_n=top_n,
            return_documents=True
        )
        # final_results = []
        # for r in response.results:
        #     original_result = semantic_results[r.index]
        #     final_results.append(original_result['chunk'])
        # return final_results
        return [item.document.text for item in response.results]
    



def test_infinity_rerank_service():
    client_url = "http://10.2.3.50:7997"
    model="mixedbread-ai/mxbai-rerank-xsmall-v1"
    query = "A man is eating pasta."
    corpus = [
        "A man is eating food.",
        "A man is eating a piece of bread.",
        "The girl is carrying a baby.",
        "A man is riding a horse.",
        "A woman is playing violin."
    ]
    rerank_service = InfinityRerankService(client_url, model)
    reranked_corpus = rerank_service.rerank(corpus, query)
    print(reranked_corpus)


def test_cohere_rerank_service():
    from app.config import env
    rerank_service = CohereRerankService(env.cohere_api_key,"rerank-english-v3.0")
    query = "A man is eating pasta."
    corpus = [
        "A man is eating food.",
        "A man is eating a piece of bread.",
        "The girl is carrying a baby.",
        "A man is riding a horse.",
        "A woman is playing violin."
    ]
    reranked_corpus = rerank_service.rerank(corpus, query)
    print(reranked_corpus)


if __name__ == "__main__":
    test_infinity_rerank_service()
    test_cohere_rerank_service()
