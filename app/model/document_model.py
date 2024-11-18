import json
from app.utils.md5hash import md5hash
from typing import List

class Chunk:
    def __init__(self,chunk_id,original_index,content,contextualized_text=None):
        self.chunk_id = chunk_id
        self.original_index = original_index
        self.content = content
        self.contextualized_text=contextualized_text

    def to_dict(self):
        return {
            "chunk_id": self.chunk_id,
            "original_index": self.original_index,
            "content": self.content,
            "contextualized_text": self.contextualized_text
        }
    
    def update_contextualized_text(self,contextualized_text):
        self.contextualized_text=contextualized_text


class Document:
    def __init__(self, doc_id:str, original_uuid:str, content:str,chunks:List[Chunk]=None):
        self.doc_id = doc_id
        self.original_uuid = original_uuid
        self.content = content
        self.chunks = chunks

    def to_dict(self):
        if self.chunks is None:
            return {
                'doc_id': self.doc_id,
                'original_uuid': self.original_uuid,
                'content': self.content
            }
        else:
            return {
                'doc_id': self.doc_id,
                'original_uuid': self.original_uuid,
                'content': self.content,
                'chunks': [chunk.to_dict() for chunk in self.chunks] 
            }

    def update_chunks(self,chunks:List[Chunk]):
        self.chunks = chunks

    def save_json(self, filename: str):
        with open(filename, "w",encoding='utf-8') as f:
            json.dump(self.to_dict(), f)
        print(f"Document saved to {filename}")

    @classmethod
    def load_json(cls, filename: str):
        with open(filename, "r", encoding='utf-8') as f:
            data = json.load(f)
        chunks = [Chunk(**chunk) for chunk in data.get('chunks', [])]
        return cls(
            doc_id=data['doc_id'],
            original_uuid=data['original_uuid'],
            content=data['content'],
            chunks=chunks
        )



class DocumentJSON:
    """Will be removed in the future"""
    def __init__(self, page_content: str, title: str, url: str):
        self.page_content = page_content
        self.metadata = {
            "uuid": md5hash(url),
            "title": title,
            "url": url
        }

    def to_dict(self):
        return {
            "page_content": self.page_content,
            "metadata": self.metadata,
        }
    
    def save_json(self, filename: str):
        with open(filename, "w",encoding='utf-8') as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def from_dict(cls, data):
        return cls(
            page_content=data.get("page_content"),
            title=data.get("metadata")['title'],
            url=data.get("metadata")['url']
        )

