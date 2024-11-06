import json
from datasets import load_dataset
import ast
from typing import List
import os
from app.service.doc_factory.wikipedia_factory import wikilink_to_docs_json,tranverse_folder_to_chunks_json,combine_chunks_file
from app.repo.contextual_vector_db import ContextualVectorDB,fast_create_by_json
from app.service.hybrid_rag import HybridRagService,rag,llm

def get_doc_list(row_num:int):
    ds = load_dataset("google/frames-benchmark")
    question=ds['test'][row_num]['Prompt']
    answer=ds['test'][row_num]['Answer']
    references_docs=ast.literal_eval(ds['test'][row_num]['wiki_links'])
    return question,answer,references_docs

def load_documents(docs_link_list:List[str],db_name:str) -> None:
    db_path=f"./data/{db_name}"
    docs_dir=os.path.join(db_path,"docs")
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
    else:
        user_input = input("Directory already exists. Do you want to continue? (y/n): ")
        if user_input.lower() != 'y':
            return
    chunks_dir=os.path.join(db_path,"chunks")
    if not os.path.exists(chunks_dir):
        os.makedirs(chunks_dir)
    # save the documents to a json file
    for i,docs_link in enumerate(docs_link_list):
        print(f"Downloading document {i+1}/{len(docs_link_list)}")
        wikilink_to_docs_json(docs_link,docs_dir)

    # split the documents into chunks
    tranverse_folder_to_chunks_json(docs_dir,chunks_dir)


def index_documents(db_name:str) -> None:
    raw_json_base=combine_chunks_file(db_name)
    db = fast_create_by_json(db_name,raw_json_base)
    print(db.db_path)

def hybrid_rag_run(query:str,db_name:str):
    def rag_function(query:str):
        result = rag(query,db_name)
        return result
    
    hybrid_rag = HybridRagService(rag_function)
    gen_answer=hybrid_rag.run(query)
    return gen_answer

# def record_results(ground_truth:str,generated_result:str):
#     print(f"Ground Truth: {ground_truth}")
#     print(f"Result: {generated_result}")



def main():
    i=1
    q,a,docs=get_doc_list(i)
    db_name=f"q{i}_contextual_db"
    load_documents(docs,db_name)
    # index_documents(db_name)
    # gen_answer=hybrid_rag_run(q,db_name)
    # print(f"Question: {q}")
    # print(f"Answer: {a}")
    # print(f"Generated Answer: {gen_answer}")

if __name__ == "__main__":
    main()