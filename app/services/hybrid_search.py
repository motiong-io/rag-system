


def retrieve_advanced(query: str, db: ContextualVectorDB, es_bm25: ElasticsearchBM25, k: int, semantic_weight: float = 0.8, bm25_weight: float = 0.2):
    num_chunks_to_recall = 50

    # Semantic search
    semantic_results = db.search(query, k=num_chunks_to_recall)
    ranked_chunk_ids = [(result['metadata']['doc_id'], result['metadata']['original_index']) for result in semantic_results]

    # BM25 search using Elasticsearch
    bm25_results = es_bm25.search(query, k=num_chunks_to_recall)
    ranked_bm25_chunk_ids = [(result['doc_id'], result['original_index']) for result in bm25_results]
    # print(ranked_bm25_chunk_ids)
    raw_conbined_results = ranked_chunk_ids + ranked_bm25_chunk_ids

    # Combine results
    chunk_ids = list(set(ranked_chunk_ids + ranked_bm25_chunk_ids))
    chunk_id_to_score = {}

    # Initial scoring with weights
    for chunk_id in chunk_ids:
        score = 0
        if chunk_id in ranked_chunk_ids:
            index = ranked_chunk_ids.index(chunk_id)
            score += semantic_weight * (1 / (index + 1))  # Weighted 1/n scoring for semantic
        if chunk_id in ranked_bm25_chunk_ids:
            index = ranked_bm25_chunk_ids.index(chunk_id)
            score += bm25_weight * (1 / (index + 1))  # Weighted 1/n scoring for BM25
        chunk_id_to_score[chunk_id] = score

    # Sort chunk IDs by their scores in descending order
    sorted_chunk_ids = sorted(
        chunk_id_to_score.keys(), key=lambda x: (chunk_id_to_score[x], x[0], x[1]), reverse=True
    )

    # Assign new scores based on the sorted order
    for index, chunk_id in enumerate(sorted_chunk_ids):
        chunk_id_to_score[chunk_id] = 1 / (index + 1)

    # Prepare the final results
    final_results = []
    semantic_count = 0
    bm25_count = 0
    for chunk_id in sorted_chunk_ids[:k]:
        # print(chunk_id)   #ERROR
        chunk_metadata = next(chunk for chunk in db.metadata if chunk['doc_id'] == chunk_id[0] and chunk['original_index'] == chunk_id[1])
        # print(chunk_metadata)
        is_from_semantic = chunk_id in ranked_chunk_ids
        is_from_bm25 = chunk_id in ranked_bm25_chunk_ids
        final_results.append({
            'chunk': chunk_metadata,
            'score': chunk_id_to_score[chunk_id],
            'from_semantic': is_from_semantic,
            'from_bm25': is_from_bm25
        })
        
        if is_from_semantic and not is_from_bm25:
            semantic_count += 1
        elif is_from_bm25 and not is_from_semantic:
            bm25_count += 1
        else:  # it's in both
            semantic_count += 0.5
            bm25_count += 0.5

    return final_results, semantic_count, bm25_count,raw_conbined_results