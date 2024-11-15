import voyageai
from app.config import env
from app.model.embeddings_model import EmbeddingObj,EmbeddingObjProperties,Embeddings
from app.model.document_model import Document,Chunk
from typing import List

class EmbeddingCreator:
    def __init__(self):
        self.embed_client = voyageai.Client(api_key=env.voyage_api_key)
        self.embed_model = "voyage-3"

    def create_embedding_for_chunk(self, chunk:Chunk,doc_id:str,doc_uuid:str) -> Embeddings:
        text_to_embed = chunk.content + "\n\n" + chunk.contextualized_text
        embedding_properties = EmbeddingObjProperties(
            doc_id=doc_id,
            original_uuid=doc_uuid,
            chunk_id=chunk.chunk_id,
            original_index=chunk.original_index,
            original_content=chunk.content,
            contextualized_text=chunk.contextualized_text
        )
        vector = self.embed_client.embed(
            text=text_to_embed,
        )

    def embed_text_list(self, text_list:List[str]) -> List[List[float]]:
        vector = self.embed_client.embed(
            texts= text_list,
            model= self.embed_model
        ).embeddings
        return vector
    

def test_embed_text():
    text_list = ["This is a test text A","This is a test text B"]
    embed_creator = EmbeddingCreator()
    vector_list = embed_creator.embed_text_list(text_list)
    print(len(vector_list))

if __name__ == "__main__":
    test_embed_text()