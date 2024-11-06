# Set up API key and do the necessary imports
from agentjo import *
from app.config import env
from app.repo.contextual_vector_db import ContextualVectorDB
from app.service.bm25 import create_elasticsearch_bm25_index,retrieve_advanced
from app.service.rerank import only_rerank
from functools import partial
from typing import Callable

def llm(system_prompt: str, user_prompt: str) -> str:
    "Local llama3.1 70B model"
    from openai import OpenAI
    client = OpenAI(base_url='http://10.1.3.6:8001/v1', api_key='api_key')
    response = client.chat.completions.create(
        model='/data/xinference_llm/.cache/modelscope/hub/LLM-Research/Meta-Llama-3___1-70B-Instruct-AWQ-INT4',
        temperature = 0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content


def split_query(query)->list:
    split_query = strict_json(
        system_prompt='You are an intelligent assistant specialized in solving multi-hop questions. Split the query into smaller sub-queries, ensuring each sub-question is specific and answerable with only one enetity.',
        user_prompt=query,
        output_format={'Sub-queries': 'Array of sub-queries'},
        llm=llm
    )
    print(split_query['Sub-queries'][0])
    return split_query['Sub-queries']

def refine_query(subquery_list:list,observation):
    refined_query = strict_json(
        system_prompt=f"""
            You are an intelligent assistant specialized in solving multi-hop questions.
            Your task is to refine the subquery list with the observed answer of sub questions.
            Your observations are:
            {str(observation)}
            Follow these steps to refine the subqueries:
            1. Use the answer you got from one subqustion to replace the key words in the other subquestion.
            2. Remove the subquestion that already answered.
            3. If the last subquestion is answered, set the end flag to True in the output.
            4. If you have the knowledge to some part of the question, you can replace the key concept with your knowledge.
        """,
        user_prompt=subquery_list,
        output_format={'Sub-queries': 'Array of refined sub-queries',
                       'End': 'Boolean flag to indicate if all sub-questions are answered'},
        llm=llm
    )
    return refined_query['Sub-queries'], refined_query['End']



def summarize_answer(query,observed_answers):
    '''Summarize the answers of sub-questions'''
    prompt=f"""
        Question: {query},
        Observed Sub-question Answers: {observed_answers}
    """

    summary=strict_json(
        system_prompt='You are an intelligent assistant specialized in solving multi-hop questions. Summarize the sub-question answers and give your final answer to the original question.',
        user_prompt=prompt,
        output_format={'Answer': 'String'},
        llm=llm
    )
    return summary['Answer']


def hybrid_search(query: str,db_name:str):
    '''Search Context based on query using a hybrid of BM25 and Semantic Search'''
    db = ContextualVectorDB(db_name)
    db.load_db()
    es_bm25=create_elasticsearch_bm25_index(db)
    final_results, semantic_count, bm25_count,raw_conbined=retrieve_advanced(query, db, es_bm25, k=30)
    final_results_rerank=only_rerank(query,final_results, k=2)

    return final_results_rerank

def rag(query,db_name):
    """Search context and answer query using RAG"""
    context=hybrid_search(query,db_name)
    prompt=f"""
        Question: {query},
        Context: {context}
    """

    rag=strict_json(
        system_prompt='You are an intelligent assistant specialized in solving multi-hop questions. Use the context to answer the query.',
        user_prompt=prompt,
        output_format={'Answer': 'String'},
        llm=llm
    )
    return rag['Answer']


def partial_init_rag(db_name:str):
    return partial(rag,db_name=db_name)

class HybridRagService:
    def __init__(self,rag_function:Callable) -> None:
        self.rag_function = rag_function

    def run(self,query:str):
        #init agent
        my_agent = Agent('Helpful assistant', "Agent to search context", llm = llm)
        my_agent.assign_functions(function_list = [self.rag_function])
        my_agent.status()
        my_agent.reset()

        #split query
        sub_query = split_query(query)
        print(sub_query)

        end_flag = False
        output = []
        while not end_flag:
            print(sub_query[0])
            answer = my_agent.run(sub_query[0])
            print(answer)
            output.append({sub_query[0]:answer[-1]})
            #refine query
            refined_query,end_flag = refine_query(sub_query,output)
            print(refined_query)
            print(end_flag)
            sub_query = refined_query
        
        print(output)
        final_answer = summarize_answer(query,output)
        print("Final Answer:",final_answer)
        return final_answer


def main():
    query="As of August 4, 2024, in what state was the first secretary of the latest United States federal executive department born?"

    def rag_function(query:str):
        db_name="q19_contextual_db"
        result = rag(query,db_name)
        return result
    
    hybrid_rag = HybridRagService(rag_function)
    hybrid_rag.run(query)

if __name__ == "__main__":
    main()





