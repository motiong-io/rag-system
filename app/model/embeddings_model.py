import json
from typing import List

class EmbeddingObjProperties:
    """properties of Embedding object (for metadata specification)"""
    def __init__(self, doc_id:str,original_uuid:str,chunk_id:str,original_index:int,original_content:str,contextualized_text:str) -> None:
        self.doc_id = doc_id
        self.original_uuid = original_uuid
        self.chunk_id = chunk_id
        self.original_index = original_index
        self.original_content = original_content
        self.contextualized_text = contextualized_text
        self.text_to_embed = original_content + "\n\n" + contextualized_text

    def to_dict(self):
        return {
            "doc_id": self.doc_id,
            "original_uuid": self.original_uuid,
            "chunk_id": self.chunk_id,
            "original_index": self.original_index,
            "original_content": self.original_content,
            "contextualized_text": self.contextualized_text,
            "text_to_embed": self.text_to_embed
        }


class EmbeddingObj:
    """A single embedding object with vector and properties"""
    def __init__(self, vector:float, properties:EmbeddingObjProperties) -> None:
        self.vector = vector
        self.properties = properties

    def to_dict(self):
        return {
            "vector": self.vector,
            "properties": self.properties.to_dict(),
        }


class Embeddings:
    """A list of EmbeddingObj for batch import to the database"""
    def __init__(self, embeddings: List[EmbeddingObj] = None) -> None:
        self.embeddings = embeddings if embeddings is not None else []
    
    def add_embedding(self, embedding:EmbeddingObj):
        self.embeddings.append(embedding)
    
    def to_dict(self)->dict:
        if len(self.embeddings) == 0:
            return []
        else:
            return [embedding.to_dict() for embedding in self.embeddings]
    
    def save_json(self, filename: str):
        with open(filename,encoding='utf-8', mode="w") as f:
            json.dump(self.to_dict(), f)
            print(f"Embeddings saved to {filename}")

    @classmethod
    def load_json(cls, filename: str):
        with open(filename, encoding='utf-8', mode="r") as f:
            data = json.load(f)
        embeddings = [EmbeddingObj(**embedding) for embedding in data]
        return cls(embeddings=embeddings)


    