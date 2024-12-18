from datasets import load_dataset
import ast

from app.services.index_services.index import KnowledgeIndexService

# this script process the first 10 rows of the dataset and index the documents to the collection
dataset=load_dataset("google/frames-benchmark")
collection_name="NemotronLocalContextualDB"


def get_QAD(row_num:int,ds=dataset):
    question=ds['test'][row_num]['Prompt']
    answer=ds['test'][row_num]['Answer']
    references_docs=ast.literal_eval(ds['test'][row_num]['wiki_links'])
    return question,answer,references_docs

def index():
    kis = KnowledgeIndexService(collection_name=collection_name)
    for i in range (15,30):
        q,a,docs=get_QAD(i)
        print(f"==================== {i} ====================")
        print(f"Question: {q}")
        print(f"Answer: {a}")
        print(f"Docs: {docs}")
        kis.batch_index_wikipedia_urls(docs,model="local_nemotron")

    kis.close()

if __name__ == "__main__":
    index()
