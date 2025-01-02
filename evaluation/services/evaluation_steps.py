from motiongreactor.llm.basic_llm import BasicLLM

from app.services.retrieve_services.context_provider import WeaviateContextProvider
from app.services.retrieve_services.query_embedding import QueryEmbedding
from app.services.retrieve_services.rerank_service import CohereRerankService

from app.config import env


class EVA:
    def __init__(self,
                 number_chunks_limit:int,
                 number_chunks_rerank:int,
                 temperature_LLM:float,
                 penalty_frequency_LLM:float,
                 if_multi_step_RAG:bool,
                 if_contextual_embedding:bool) -> None:

        # Parameters for the optimization
        self.number_chunks_limit = number_chunks_limit
        self.number_chunks_rerank = number_chunks_rerank
        self.temperature_LLM = temperature_LLM
        self.penalty_frequency_LLM = penalty_frequency_LLM
        self.if_multi_step_RAG = if_multi_step_RAG
        self.if_contextual_embedding = if_contextual_embedding

        # Parameters for Orchestrator
        self.llm = BasicLLM(base_url=env.nvidia_local_base_url,
                api_key="abc",
                model="llama3_3")
        
        self.context_provider = WeaviateContextProvider(weaviate_url=env.weaviate_url,
                                            weaviate_key=env.weaviate_api_key,
                                            collection_name="Llama3_3LocalContextualDB" if self.if_contextual_embedding else "ChunksVectorDB",
                                            chunks_N=self.number_chunks_limit)
        self.query_embedding = QueryEmbedding()
        self.reranker = CohereRerankService(env.cohere_api_key,'rerank-english-v3.0')

        if self.if_multi_step_RAG:
            from motiongreactor.orchestrators.multi_step_rag import MultiStepRagOrchestrator
            self.reactor = MultiStepRagOrchestrator(llm=self.llm,
                                    context_providers=[self.context_provider],
                                    embedding=self.query_embedding,
                                    reranker=self.reranker)
        else:
            from motiongreactor.orchestrators.simple_rag import SimpleRagOrchestrator
            self.reactor = SimpleRagOrchestrator(llm=self.llm,
                                context_providers=[self.context_provider],
                                embedding=self.query_embedding,
                                reranker=self.reranker)


    def run(self, question:str):
        # Step 1: Retrieve
        # Step 2: Rerank
        # Step 3: Generate
        # Step 4: Evaluate
        # Step 5: Return
        pass
    