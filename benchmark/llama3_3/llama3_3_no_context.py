
import json
from aiwrappifymodels.message import MessageForLLM
import pandas as pd


from motiongreactor.llm.basic_llm import BasicLLM


from app.config import env


# initialize the RAG services
llm = BasicLLM(base_url=env.nvidia_local_base_url,
                api_key="abc",
                model="llama3_3")


def test_reactor(question:str):
    conversation = [
        MessageForLLM(
            role="user",
            content=question,
        )
    ]
    llm_answer = llm.full_response(conversation, sys_prompt="You are an intelligent assistant specialized in solving multi-hop questions.",temperature=0)
    return llm_answer


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

for index, row in df_first_30.iterrows():
    question = row['Prompt']
    answer = row['Answer']
    result = test_reactor(question)

    record_result(index,question,answer,result,log=None)

