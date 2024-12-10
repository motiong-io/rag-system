from datasets import load_dataset
import ast

from app.services.index_services.index import KnowledgeIndexService


dataset=load_dataset("google/frames-benchmark")
# collection_name="GPT4ominiContextualDB"
collection_name = "NemotronContextualDB"

def get_QAD(row_num:int,ds=dataset):
    question=ds['test'][row_num]['Prompt']
    answer=ds['test'][row_num]['Answer']
    references_docs=ast.literal_eval(ds['test'][row_num]['wiki_links'])
    return question,answer,references_docs

def index():
    kis = KnowledgeIndexService(collection_name=collection_name)
    for i in range (1,10):
        q,a,docs=get_QAD(i)
        print(f"Question: {q}")
        print(f"Answer: {a}")
        print(f"Docs: {docs}")
        kis.batch_index_wikipedia_urls(docs)



    
if __name__ == "__main__":
    index()
