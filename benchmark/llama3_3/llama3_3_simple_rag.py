
import json
from typing import Iterable
from aiwrappifymodels.message import MessageForLLM
import pandas as pd
import sys
import io
import time

from motiongreactor.llm.basic_llm import BasicLLM
from motiongreactor.orchestrators.simple_rag import SimpleRagOrchestrator

from app.services.retrieve_services.context_provider import WeaviateContextProvider
from app.services.retrieve_services.query_embedding import QueryEmbedding
from app.services.retrieve_services.rerank_service import CohereRerankService

from app.config import env

from app.utils.dual_output import DualOutput

llm = BasicLLM(base_url=env.nvidia_local_base_url,
                api_key="abc",
                model="llama3_3")


context_provider = WeaviateContextProvider(weaviate_url=env.weaviate_url,
                                            weaviate_key=env.weaviate_api_key,
                                            collection_name="ChunksVectorDB")

query_embedding = QueryEmbedding()
reranker = CohereRerankService(env.cohere_api_key,'rerank-english-v3.0')

# define the RAG orchestrator
reactor = SimpleRagOrchestrator(llm=llm,
                                context_providers=[context_provider],
                                embedding=query_embedding,
                                reranker=reranker)

def test_reactor(question:str):
    conversation = [
        MessageForLLM(
            role="user",
            content=question,
        )
    ]
    full_answer = ""
    for x in reactor.react(
        conversation=conversation, product_related_prompt=None
    ):
        full_answer += x
    return full_answer


def record_result(index:int,question:str, answer:str, result:str,log:str):

    record={
        "Index":index,
        "Question":question,
        "Answer":answer,
        "Result":result,
        "log":log
    }

    with open('benchmark/results.json', 'a', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=4)
        f.write("\n\n")
    # print(record)

df = pd.read_csv("hf://datasets/google/frames-benchmark/test.tsv", sep="\t")

df_first_30 = df.head(30)

try:
    captured_output = io.StringIO()
    original_stdout = sys.stdout 
    sys.stdout = DualOutput(original_stdout, captured_output)

    for index, row in df_first_30.iterrows():
        if index not in [28,29]:
            continue
        print(f"==================== {index} ====================")
        question = row['Prompt']
        answer = row['Answer']
        result = test_reactor(question)
        log = captured_output.getvalue()
        # log = None

        record_result(index,question,answer,result,log)
        time.sleep(1)
        captured_output.truncate(0)
        captured_output.seek(0)

except Exception as e:
    print(e)

finally:
    sys.stdout = original_stdout
    context_provider.close()