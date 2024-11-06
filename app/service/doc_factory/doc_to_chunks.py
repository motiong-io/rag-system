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


def tranverse_folder(folder_path:str,target_folder:str):
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