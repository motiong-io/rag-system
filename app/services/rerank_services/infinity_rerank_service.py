from infinity_client import Client
from infinity_client.models import RerankInput,ReRankResult
from infinity_client.api.default import classify, embeddings, embeddings_image, rerank
from infinity_client.types import Response
from typing import List

from app.services.rerank_services.base_rerank_service import BaseRerankService


class InfinityRerankService(BaseRerankService):
    def __init__(self, client_url, model_name):
        self.client_url = client_url
        self.model_name = model_name
        self.client = Client(base_url=self.client_url)

    def rerank(self, corpus:List[str], query:str,top_n:int=None) -> ReRankResult:
        with self.client as client:
            reranked_corpus = rerank.sync(client=client, body=RerankInput.from_dict({
            "query": query,
            "documents": corpus,
            "return_documents": True,
            "model": self.model_name,
            "top_n": top_n
            }))
            return reranked_corpus

# def only_rerank(query,semantic_results:list, k: int) -> List[Dict[str, Any]]:
#     co = cohere.Client(os.getenv("COHERE_API_KEY"))
    
#     # Extract documents for reranking, using the contextualized content
#     documents = [chunk_to_content(res) for res in semantic_results]

#     response = co.rerank(
#         model="rerank-english-v3.0",
#         query=query,
#         documents=documents,
#         top_n=k
#     )
#     time.sleep(0.1)
    
#     final_results = []
#     for r in response.results:
#         original_result = semantic_results[r.index]
#         final_results.append({
#             "chunk": original_result['chunk'],
#             "score": r.relevance_score
#         })
    
#     return final_results
def test_infinity_rerank_service():
    client_url = "http://10.1.17.3:7997"
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
    print(reranked_corpus.results)

if __name__ == "__main__":
    test_infinity_rerank_service()
