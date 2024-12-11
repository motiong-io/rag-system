
from typing import Iterable
from motiongreactor.llm.basic_llm import BasicLLM

from aiwrappifymodels.message import MessageForLLM

from app.services.retrieve_services.context_provider import WeaviateContextProvider
from app.services.retrieve_services.query_embedding import QueryEmbedding
from app.services.retrieve_services.rerank_service import CohereRerankService

from motiongreactor.orchestrators.multi_step_rag import MultiStepRagOrchestrator
# from motiongreactor.orchestrators.simple_rag import SimpleRagOrchestrator
from app.config import env


class RagService:
    def __init__(self) -> None:
        self.llm = BasicLLM(base_url=env.nvidia_base_url,
                            api_key=env.nvidia_api_key,
                            model="nvidia/llama-3.1-nemotron-70b-instruct"
                            )
        self.context_provider = WeaviateContextProvider(weaviate_url=env.weaviate_url,
                                                        weaviate_key=env.weaviate_api_key,
                                                        collection_name="NemotronContextualDB"
                                                        )

        self.query_embedding = QueryEmbedding()
        self.reranker = CohereRerankService(env.cohere_api_key,'rerank-english-v3.0')

        self.reactor = MultiStepRagOrchestrator(llm=self.llm,
                                                context_providers=[self.context_provider],
                                                embedding=self.query_embedding,
                                                reranker=self.reranker
                                                )
    
    def react(
        self, conversation: list[MessageForLLM], product_related_prompt: str = None
    ) -> Iterable[str]:
        return self.reactor.react(conversation, product_related_prompt)

    def close(self):
        self.context_provider.close()
    


def test_rag_service():
    rag_service = RagService()
    conversation = [
        MessageForLLM(
            role="user",
            content="If my future wife has the same first name as the 15th first lady of the United States' mother and her surname is the same as the second assassinated president's mother's maiden name, what is my future wife's name?",
        ),
        # MessageForLLM(
        #     role="assistant",
        #     content="The capital of France is Paris.",
        # ),
        # MessageForLLM(
        #     role="user",
        #     content="What is the capital of Germany?",
        # ),
    ]
    full_answer = ""
    for x in rag_service.react(
        conversation=conversation, product_related_prompt=None
    ):
        full_answer += x

    # for m in conversation:
    #     print(f"{m.role}: {m.content}")
    # print("===FULL ANSWER GENERATED===:")
    # print(f"ChatRoleEnum.assistant: {full_answer}")

    rag_service.close()


if __name__ == "__main__":
    test_rag_service()