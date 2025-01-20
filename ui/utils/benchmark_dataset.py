import pandas as pd
import math
import ast

from app.utils.md5hash import md5hash
# from ui.utils.manage_weaviate import aggregate_objects_by_property

CSV_PATH = "evaluation/dataset/row_50.csv"
df= pd.read_csv(CSV_PATH)

def load_data(page_limit:int =5):
    count = df.shape[0]
    totak_page_num = math.ceil(count/page_limit)
    page_list = list(range(totak_page_num))
    page_data_index_range = []

    for page in page_list:
        start_index = (page) * page_limit 

        end_index = min((page + 1) * page_limit-1, count-1)  
        page_data_index_range.append((start_index, end_index)) 

    return count,page_data_index_range

def get_page_data(index_range:tuple):
    df_range =  df.loc[index_range[0]:index_range[1], ["Unnamed: 0","Prompt", "Answer","wiki_links"]].to_dict(orient="records")

    data = {
        'defaultColDef': {'flex': 1},
        'columnDefs': [
            {'headerName': 'i', 'field': 'Unnamed: 0'},
            {'headerName': 'Query', 'field': 'Prompt'},
            {'headerName': 'Answer', 'field': 'Answer'},#, 'hide': True
            {'headerName': 'Wiki Links', 'field': 'wiki_links', 'hide': True}
        ],
    }
    data['rowData'] = df_range
    return data

def string_to_list(str_wiki_list:str):
    return ast.literal_eval(str_wiki_list)


def get_docs_data(wiki_links:list):

    row_data = []
    for i in range (len(wiki_links)):
        url = wiki_links[i]
        url_html = f'<a href = "{url}" target="_blank">{url}</a>'
        uuid = md5hash(url)
        # related_chunks = aggregate_objects_by_property(collection_name,'original_uuid',uuid)
        row_data.append(
            {
                'no': i+1,
                'url': url_html,
                'uuid': uuid,
                'chunk':'->'
            }
        )

    data = {
        'defaultColDef': {'flex': 1},
        'columnDefs': [
            {'headerName': 'No.', 'field': 'no', 'width': 40},
            {'headerName': 'url', 'field': 'url'},
            {'headerName': 'uuid', 'field': 'uuid'},
            {'headerName': 'chunk', 'field': 'chunk','width': 40},
        ],
        'rowData':row_data
        }
    return data
    


if __name__ == "__main__":
    count, page_data_index_range = load_data()
    print(count)
    print(page_data_index_range)
    print(get_page_data(page_data_index_range[1]))