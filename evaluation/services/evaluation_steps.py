
from app.model.message_for_llm import MessageForLLM

from app.services.retrieve_services.context_provider import WeaviateContextProvider
from app.services.retrieve_services.query_embedding import QueryEmbedding
from app.services.retrieve_services.rerank_service import CohereRerankService

from app.llm.get_llm import get_llm

from app.config import env

from evaluation.services.check_answer import check_answer


class EVA:
    """
    calculate
    
    """
    def __init__(self,
                 number_chunks_limit:int,
                 hybrid_search_alpha:float,
                 number_chunks_rerank:int,
                 temperature_LLM:float,
                 penalty_frequency_LLM:float,
                 if_multi_step_RAG:bool,
                 if_contextual_embedding:bool) -> None:

        # Parameters for the optimization
        self.number_chunks_limit = number_chunks_limit
        self.hybrid_search_alpha = hybrid_search_alpha
        self.number_chunks_rerank = number_chunks_rerank
        self.temperature_LLM = temperature_LLM
        self.penalty_frequency_LLM = penalty_frequency_LLM
        self.if_multi_step_RAG = if_multi_step_RAG
        self.if_contextual_embedding = if_contextual_embedding

        # Parameters for Orchestrator
        ## LLM
        self.llm = get_llm(llm_name='llama3_3')
        
        ## Context Provider
        self.context_provider = WeaviateContextProvider(weaviate_url=env.weaviate_url,
                                            weaviate_key=env.weaviate_api_key,
                                            collection_name="Llama3_3LocalContextualDB" if self.if_contextual_embedding else "ChunksVectorDB",
                                            chunks_retrieved=self.number_chunks_limit, hybrid_search_alpha=self.hybrid_search_alpha
                                            ) 

        ## Query Embedding
        self.query_embedding = QueryEmbedding()
        self.reranker = CohereRerankService(env.cohere_api_key,'rerank-english-v3.0',output_chunks_number=self.number_chunks_rerank)

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

    def __react(self,question:str):
        if self.if_multi_step_RAG:
            temperature_setting = [0, self.temperature_LLM,0,0]
            frequency_penalty_setting = [self.penalty_frequency_LLM]*4
        else:
            temperature_setting = self.temperature_LLM
            frequency_penalty_setting = self.penalty_frequency_LLM

        conversation = [
            MessageForLLM(
                role="user",
                content=question,
            )
        ]
        full_answer = ""
        for x in self.reactor.react(
            conversation=conversation, product_related_prompt=None, llm_temperature=temperature_setting, frequency_penalty=frequency_penalty_setting
        ):
            full_answer += x
        return full_answer

    def run(self, question:str,ground_truth:str):
        generated_answer = self.__react(question)
        check_result = check_answer(question, ground_truth, generated_answer)
        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Generated Answer: {generated_answer}")
        print(f"Check Result: {check_result}")
        
def test_EVA():
    eva = EVA(number_chunks_limit=100,
              hybrid_search_alpha=0.5,
              number_chunks_rerank=10,
              temperature_LLM=0.1,
              penalty_frequency_LLM=0.5,
              if_multi_step_RAG=True,
              if_contextual_embedding=False)
    
    question = "If my future wife has the same first name as the 15th first lady of the United States' mother and her surname is the same as the second assassinated president's mother's maiden name, what is my future wife's name? "
    ground_truth = "Jane Ballou"
    eva.run(question,ground_truth)
    eva.context_provider.close()


if __name__ == "__main__":
    test_EVA()