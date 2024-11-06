import logging
from langchain_community.document_loaders import WikipediaLoader


class WikipediaFactory:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_documents(self, wiki_query: str, language: str):
        try:
            pages = WikipediaLoader(query=wiki_query.strip(), lang=language, load_all_available_meta=False, doc_content_chars_max=20000).load()
            file_name = wiki_query.strip()
            self.logger.info(f"Total Pages from Wikipedia = {len(pages)}")
            return file_name, pages
        except Exception as e:
            job_status = "Failed"
            message = "Failed To Process Wikipedia Query"
            error_message = str(e)
            file_name = wiki_query.strip()
            self.logger.error(f"Failed To Process Wikipedia Query: {file_name}")
            self.logger.exception(f'Exception Stack trace: {error_message}')
            return file_name, {
                "job_status": job_status,
                "message": message,
                "error": error_message,
                "file_name": file_name
            }

from urllib.parse import unquote


def get_wikipedia_title(url):
    # Split the URL to get the last part after "/wiki/"
    title_part = url.split('/wiki/')[-1]
    # Decode any percent-encoded characters, e.g., spaces represented as %20
    title = unquote(title_part)
    # Replace underscores with spaces if needed
    title = title.replace('_', ' ')
    return title


def link_to_json_file(wiki_link: str, language: str):
    wiki_query = get_wikipedia_title(wiki_link)
    try:
        pages = WikipediaLoader(query=wiki_query.strip(), lang=language, load_all_available_meta=False).load()
        file_name = wiki_query.strip()
        return file_name, pages
    except Exception as e:
        job_status = "Failed"
        message = "Failed To Process Wikipedia Query"
        error_message = str(e)
        file_name = wiki_query.strip()
        return file_name, {
            "job_status": job_status,
            "message": message,
            "error": error_message,
            "file_name": file_name
        }


def wikilink_to_docs_json(wiki_link:str,save_dir:str):
    print(wiki_link)
    file_name, pages = link_to_json_file(wiki_link, 'en')
    file_name = file_name.replace(" ", "_")
    combined_content = "\n".join([page.page_content for page in pages])
    title=pages[0].metadata['title']
    print(title)
    # Ensure the directory exists
    os.makedirs(f'./{save_dir}/', exist_ok=True)

    # Save the pages to a file
    with open(f'./{save_dir}/{file_name}.json', 'w') as f:
        content=Document(
            page_content=combined_content,
            metadata= {'title':pages[0].metadata['title'],'url':wiki_link}
        )
        json.dump(content.to_dict(), f)
    print(f"Saved pages to /{save_dir}/{file_name}.json")





from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
import hashlib
from app.service.doc_factory.data_model import Document
import json
import os

def split2chunks(text: str,chunk_size:int,chunk_overlap:int) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_text(text)
    return chunks


def generate_md5_hash(input_string):
    # Create an MD5 hash object
    md5_hash = hashlib.md5()
    
    # Update the hash object with the bytes of the input string
    md5_hash.update(input_string.encode('utf-8'))
    
    # Get the hexadecimal representation of the hash
    return md5_hash.hexdigest()

def docs_to_chunks_json(doc_data:Document,chunk_size:int,chunk_overlap:int):
    doc_chunks={
        "doc_id":doc_data.metadata['title'].replace(" ","_"),
        "original_uuid": generate_md5_hash(doc_data.metadata['title'].replace(" ","_")),
        "content":doc_data.page_content
    }
    chunk_list=split2chunks(doc_data.page_content,chunk_size,chunk_overlap)
    chunks=[]
    for i in range(len(chunk_list)):
        chunk_obj={
            "chunk_id":doc_chunks['doc_id']+"_chunk_"+str(i),
            "original_index": i,
            "content":chunk_list[i]
        }
        chunks.append(chunk_obj)
    doc_chunks['chunks']=chunks
    return doc_chunks


def tranverse_folder_to_chunks_json(folder_path:str,target_folder:str):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            codebase_chunks=[]
            with open(file_path) as file:
                doc_data = json.load(file)
                document = Document(page_content=doc_data['page_content'], metadata=doc_data['metadata'])
                chunked_data = docs_to_chunks_json(document, chunk_size=1000, chunk_overlap=100)
                codebase_chunks.append(chunked_data)
                # Ensure the directory exists
                os.makedirs(target_folder, exist_ok=True)

                # Save the chunked data to a file
                with open(f"{target_folder}/chunks_{file_name}", 'w') as f:
                    json.dump(codebase_chunks, f,indent=4, ensure_ascii=False)
                print(f"Saved chunks to {target_folder}/chunks_{file_name}")
    

def combine_chunks_file(db_name:str):
    chunks_folder_path=f"./data/{db_name}/chunks"
    combined_chunks=[]
    for file_name in os.listdir(chunks_folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(chunks_folder_path, file_name)
            with open(file_path) as file:
                doc_data = json.load(file)
                combined_chunks.extend(doc_data)
    save_path=f"./data/{db_name}/combined_doc_chunk.json"
    with open(save_path, 'w') as f:
        json.dump(combined_chunks, f,indent=4, ensure_ascii=False)

    print(f"Combined {len(combined_chunks)} chunks")
    return save_path




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    factory = WikipediaFactory()
    query = "Python programming language"
    language = "en"
    file_name, result = factory.get_documents(query, language)
    if isinstance(result, dict) and result.get("job_status") == "Failed":
        print(f"Error: {result['message']}")
        print(f"Details: {result['error']}")
    else:
        print(f"Successfully retrieved {len(result)} pages for query '{query}'")
        print(result[0].page_content)