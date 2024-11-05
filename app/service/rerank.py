from typing import List, Dict, Any, Callable
from app.utils.x_client import Client

class RerankService:
    def __init__(self, client_url, model_name):
        self.client = Client(client_url)
        self.model = self.client.get_model(model_name)

    def rerank(self, corpus:List[str], query:str):
        return self.model.rerank(corpus, query)

import cohere
import os
import time

def chunk_to_content(chunk: Dict[str, Any]) -> str:
    
    original_content = chunk['chunk']['original_content']
    contextualized_content = chunk['chunk']['contextualized_content']
    return f"{original_content}\n\nContext: {contextualized_content}" 

def retrieve_rerank(query: str, db, k: int) -> List[Dict[str, Any]]:
    co = cohere.Client(os.getenv("COHERE_API_KEY"))
    
    # Retrieve more results than we normally would
    semantic_results = db.search(query, k=k*10)
    
    # Extract documents for reranking, using the contextualized content
    documents = [chunk_to_content(res['chunk']) for res in semantic_results]

    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=documents,
        top_n=k
    )
    time.sleep(0.1)
    
    final_results = []
    for r in response.results:
        original_result = semantic_results[r.index]
        final_results.append({
            "chunk": original_result['metadata'],
            "score": r.relevance_score
        })
    
    return final_results

def only_rerank(query,semantic_results:list, k: int) -> List[Dict[str, Any]]:
    co = cohere.Client(os.getenv("COHERE_API_KEY"))
    
    # Extract documents for reranking, using the contextualized content
    documents = [chunk_to_content(res) for res in semantic_results]

    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=documents,
        top_n=k
    )
    time.sleep(0.1)
    
    final_results = []
    for r in response.results:
        original_result = semantic_results[r.index]
        final_results.append({
            "chunk": original_result['chunk'],
            "score": r.relevance_score
        })
    
    return final_results



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
