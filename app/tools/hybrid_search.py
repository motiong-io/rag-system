from app.repo.contextual_vector_db import ContextualVectorDB
from app.services.old_service.bm25 import create_elasticsearch_bm25_index,retrieve_advanced
from app.services.old_service.rerank import only_rerank

def hybrid_search(query):
    db = ContextualVectorDB("q19_contextual_db")
    db.load_db()
    es_bm25=create_elasticsearch_bm25_index(db)
    final_results, semantic_count, bm25_count,raw_conbined=retrieve_advanced(query, db, es_bm25, k=30)
    final_results_rerank=only_rerank(query,final_results, k=15)
    return final_results_rerank
