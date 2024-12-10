import voyageai
from app.config import env
from app.model.embeddings_model import EmbeddingObj,EmbeddingObjProperties,Embeddings
from app.model.document_model import Document
from typing import List

class EmbeddingCreator:
    def __init__(self):
        self.embed_client = voyageai.Client(api_key=env.voyage_api_key)
        self.embed_model = "voyage-3"

    def create_embeddings_for_document(self, document: Document) -> Embeddings:
        embeddings = Embeddings()
        
        chunks = document.chunks
        properties_list = [
            EmbeddingObjProperties(
                doc_id=document.doc_id,
                original_uuid=document.original_uuid,
                chunk_id=chunk.chunk_id,
                original_index=chunk.original_index,
                original_content=chunk.content,
                contextualized_text=chunk.contextualized_text
            )
            for chunk in chunks
        ]
        
        texts_to_embed = [prop.text_to_embed for prop in properties_list]
        vectors = self.embed_text_list(texts_to_embed)
        
        for properties, vector in zip(properties_list, vectors):
            embedding_obj = EmbeddingObj(vector=vector, properties=properties)
            embeddings.add_embedding(embedding_obj)
        
        return embeddings

    def embed_text_list(self, text_list: List[str]) -> List[List[float]]:
        vector_list = self.embed_client.embed(
            texts=text_list,
            model=self.embed_model
        ).embeddings
        return vector_list
    
    def embed_text(self, text: str) -> List[float]:
        vector = self.embed_client.embed(
            texts=[text],
            model=self.embed_model
        ).embeddings[0]
        return vector
    

def test_embed_text():
    text_list = ["This is a test text A"]
    embed_creator = EmbeddingCreator()
    vector_list = embed_creator.embed_text_list(text_list)
    print(len(vector_list))

def test_create_embeddings_for_document():
    json_path = "assets/dataset/document_json/417e546b48ce6e74b37c0815920013dc.json"
    document = Document.load_json(json_path)
    embeddings= EmbeddingCreator().create_embeddings_for_document(document)
    embeddings.save_json("assets/dataset/embeddings_list/417e546b48ce6e74b37c0815920013dc.json")


if __name__ == "__main__":
    # test_embed_text()
    test_create_embeddings_for_document()