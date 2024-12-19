
from app.repo.weaviate_cloud import WeaviateClient

def seach_doc_id_in_weaviate(doc_id):

    collection_name="GPT4ominiContextualDB"

    client = WeaviateClient(collection_name).client

    query = """
    {
    Aggregate {
        GPT4ominiContextualDB(where: {
        path: ["doc_id"]
        operator: Equal
        valueString: "%s"
        }) {
        meta {
            count
        }
        }
    }
    }
    """ % doc_id

    response = client.graphql_raw_query(query)
    print(response)
    client.close()


import pandas as pd
import ast
from app.services.docs_loader.wikipedia_loader import WikipediaLoader


df = pd.read_csv("hf://datasets/google/frames-benchmark/test.tsv", sep="\t")

df_first_30 = df.head(30)
for index, row in df_first_30.iterrows():
    print(f"==================== {index} ====================")
    links = ast.literal_eval(row['wiki_links'])
    print(links)
    print(f"{len(links)} total links")
    titles = [WikipediaLoader(link).formated_title for link in links]
    print(titles)
    for title in titles:
        seach_doc_id_in_weaviate(title)