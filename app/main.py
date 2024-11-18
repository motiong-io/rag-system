
from datasets import load_dataset
import ast

from app.repo.document_json_db import DocumentJsonDB


def get_QAD(row_num:int):
    ds = load_dataset("google/frames-benchmark")
    question=ds['test'][row_num]['Prompt']
    answer=ds['test'][row_num]['Answer']
    references_docs=ast.literal_eval(ds['test'][row_num]['wiki_links'])
    return question,answer,references_docs

def load_docs():
    docs_db=DocumentJsonDB("assets/dataset/document_json")
    print(docs_db.list())


def main():
    for i in range (1,2):
        q,a,docs=get_QAD(i)
        print(f"Question: {q}")
        print(f"Answer: {a}")
        print(f"Docs: {docs}")


    
if __name__ == "__main__":
    main()
