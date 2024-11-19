from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, BulkIndexError
from typing import List, Dict, Any

from app.config import env
from app.model.document_model import Document,Chunk


class ElasticSearchClient:
    def __init__(self, index_name: str):
        self.es_client = Elasticsearch(env.elastic_search_url,api_key=env.elastic_search_api_key)
        self.index_name = index_name
        self.get_index()

    def get_index(self):
        index_settings = {
            "settings": {
                "analysis": {"analyzer": {"default": {"type": "english"}}},
                "similarity": {"default": {"type": "BM25"}},
                "index.queries.cache.enabled": False  # Disable query cache
            },
            "mappings": {
                "properties": {
                    "content": {"type": "text", "analyzer": "english"},
                    "contextualized_content": {"type": "text", "analyzer": "english"},
                    "doc_id": {"type": "keyword", "index": False},
                    "chunk_id": {"type": "keyword", "index": False},
                    "original_index": {"type": "integer", "index": False},
                }
            },
        }
        if not self.es_client.indices.exists(index=self.index_name):
            self.es_client.indices.create(index=self.index_name, body=index_settings)
            print(f"Created new index: {self.index_name}")
        else:
            print(f"Found existed index: {self.index_name}")
    
    def index_document(self, document: Document):
        actions = [
            {
                "_index": self.index_name,
                "_id":chunk.uuid,
                "_source": {
                    "content":chunk.content,
                    "contextualized_content":chunk.contextualized_text,
                    "doc_id":document.doc_id,
                    "chunk_id":chunk.chunk_id,
                    "original_index":chunk.original_index
                }
            }
            for chunk in document.chunks
        ]
        try:
            success, _ = bulk(self.es_client, actions)
        except BulkIndexError as e:
            print(e.errors)  # 打印或记录详细的错误信息
        self.es_client.indices.refresh(index=self.index_name)
        print(f"Indexed {len(actions)} chunks")
        return success
    
    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        self.es_client.indices.refresh(index=self.index_name)  # Force refresh before each search
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["content", "contextualized_content"],
                }
            },
            "size": k,
        }
        response = self.es_client.search(index=self.index_name, body=search_body)
        return [
            {
                "doc_id": hit["_source"]["doc_id"],
                "original_index": hit["_source"]["original_index"],
                "content": hit["_source"]["content"],
                "contextualized_content": hit["_source"]["contextualized_content"],
                "score": hit["_score"],
            }
            for hit in response["hits"]["hits"]
        ]


if __name__ == "__main__":
    pass