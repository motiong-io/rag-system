from app.model.document_model import Document
from app.services.wikipedia_loader import WikipediaLoader
from app.services.document_splitter import DocumentSplitter
from app.services.chunk_contextualizer import AsyncChunkContextualizer
from app.services.embedding_creator import EmbeddingCreator
from app.repo.weaviate_cloud import WeaviateClient
from app.repo.elastic_cloud import ElasticSearchClient
import asyncio

class KnowledgeIndexService:
    def __init__(self,save_markdown:bool=True,save_document:bool=True,save_embeddings:bool=True) -> None:
        self.markdown_dir = "assets/dataset/markdown_files" if save_markdown else None
        self.document_dir="assets/dataset/document_json" if save_document else None
        self.embeddings_dir="assets/dataset/embeddings_list" if save_embeddings else None
        self.weaviate_client = WeaviateClient("ContextualVectors")
        self.elastic_client = ElasticSearchClient("contextual_chunks")
    
    def index_from_wikipedia_url(self, wikipedia_url:str):
        # Load the wikipedia page to document
        wikipedia_loader = WikipediaLoader(wikipedia_url)
        if self.markdown_dir:
            wikipedia_loader.save_markdown(f"{self.markdown_dir}/{wikipedia_loader.uuid}.md") # Save the markdown file
        document_no_chunks = wikipedia_loader.to_document()
        # Split the document into chunks
        document_with_chunks_no_contextualized_text = DocumentSplitter().split_document(document_no_chunks)
        # Contextualize the chunks
        document = asyncio.run(AsyncChunkContextualizer().contextualize_document(document_with_chunks_no_contextualized_text))
        # Save the document to a file
        if self.document_dir:
            document.save_json(f"{self.document_dir}/{document.original_uuid}.json")
        # Create embeddings for the document
        embeddings= EmbeddingCreator().create_embeddings_for_document(document)
        if self.embeddings_dir:
            embeddings.save_json(f"{self.embeddings_dir}/{document.original_uuid}.json")
        # import embeddings to weaviate
        self.weaviate_client.batch_import(embeddings)
        # Index the document to elastic search
        success=self.elastic_client.index_document(document)
        return success
    
    def close(self):
        self.weaviate_client.close()
    






def test_document_from_wikipedia_url():
    # url="https://en.wikipedia.org/wiki/List_of_tallest_buildings_in_New_York_City"
    url="https://en.wikipedia.org/wiki/Jane_Eyre"
    service = KnowledgeIndexService()
    try:
        service.index_from_wikipedia_url(url)
    except Exception as e:
        print(e)
    service.close()

if __name__ == "__main__":
    test_document_from_wikipedia_url()