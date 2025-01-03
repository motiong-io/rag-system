# Evaluation

> Note this part do not include the indexing process.

- Target:
    - Build a system to optimize the performance of the RAG system, leveraging the mixed integer nonlinear programming (MINLP) models.
    - Stage 1: Minimize the loss of answering complex questions.
    - Stage 2: Find the trade-off between efficiency (time-consuming) and quality (accuracy).

- Methodology:
    - Use [MINLP Formulation](https://pyomo.readthedocs.io/en/6.8.0/contributed_packages/mindtpy.html#minlp-formulation) to model the system and analyze the influence of each parameter.

# /demo
- Fixed parameters:
    - RAG LLM: llama3.3-70B
    - Dataset:
        - Source: https://huggingface.co/datasets/google/frames-benchmark

- Independent variables to investigate:
    - Number of chunks retrieved by hybrid search: `int N_s in range(3, 301)`
    - Number of chunks picked after reranking `int N_r in range(1, 11)`
    - Hybrid search alpha  `float alpha in [0.0, 1.0]`
    - LLM temperature `float T in [0.0, 1.0]`
    - LLM frequency penalty `float P_f in [-2.0, 2.0]`
    - Multi-step or not (simple RAG) `bool M True or False`
    - Contextual embedding or not `bool C True or False`
        - `C = True` : ChunksVectorDB: Embedding the first 30 items from the source.
        - `C = False`: Llama3_3LocalContextualDB: Contextual embedding of the first 30 items from the source.

- Dependent variables:
    - Loss: 1 - accuracy `float loss in [0.0, 1.0]`
    - Time: duration from asking to final answer `float time in (0, inf)`
