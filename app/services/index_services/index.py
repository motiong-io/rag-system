from app.model.document_model import Document
from app.model.embeddings_model import Embeddings
from app.services.docs_loader.wikipedia_loader import WikipediaLoader
from app.services.index_services.document_splitter import DocumentSplitter
from app.services.index_services.chunk_contextualizer import AsyncChunkContextualizer #, ChunkContextualizer
from app.services.index_services.embedding_creator import EmbeddingCreator
from app.repo.weaviate_cloud import WeaviateClient
# from app.repo.elastic_cloud import ElasticSearchClient
import asyncio
import os
from typing import Literal
import tracemalloc



class KnowledgeIndexService:
    def __init__(self,save_markdown:bool=True,save_document:bool=True,save_embeddings:bool=True,collection_name:str="ContextualVectors") -> None:
        self.markdown_dir = "assets/markdown_files" if save_markdown else None
        self.document_dir="assets/dataset/document_json" if save_document else None
        self.embeddings_dir="assets/dataset/embeddings_list" if save_embeddings else None
        self.weaviate_client = WeaviateClient(collection_name)
        # self.elastic_client = ElasticSearchClient("contextual_chunks")
        if self.markdown_dir:
            os.makedirs(self.markdown_dir,exist_ok=True)
        if self.document_dir:
            os.makedirs(self.document_dir,exist_ok=True)
        if self.embeddings_dir:
            os.makedirs(self.embeddings_dir,exist_ok=True)

    def index_from_wikipedia_url(self, wikipedia_url:str, model:Literal['gpt', 'nemotron','local_nemotron','local_llama3_3']):
        # Load the wikipedia page to document
        wikipedia_loader = WikipediaLoader(wikipedia_url)
        if self.markdown_dir:
            wikipedia_loader.save_markdown(f"{self.markdown_dir}/{wikipedia_loader.uuid}.md") # Save the markdown file
        document_no_chunks = wikipedia_loader.to_document()
        # Split the document into chunks
        document_with_chunks_no_contextualized_text = DocumentSplitter().split_document(document_no_chunks)

        if not os.path.exists(f"{self.document_dir}/{wikipedia_loader.uuid}.json"):
            # Contextualize the chunks
            try:
                tracemalloc.start()
                document = asyncio.run(AsyncChunkContextualizer(model).contextualize_document(document_with_chunks_no_contextualized_text))
            finally:
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')

                print("[ Top 10 ]")
                for stat in top_stats[:10]:
                    print(stat)
            

            # document = ChunkContextualizer().contextualize_document(document_with_chunks_no_contextualized_text)
            # Save the document to a file
            if self.document_dir:
                document.save_json(f"{self.document_dir}/{document.original_uuid}.json")
        else:
            print(f"Document already exists for {wikipedia_loader.uuid}")
            document = Document.load_json(f"{self.document_dir}/{document_no_chunks.original_uuid}.json")

        if not os.path.exists(f"{self.embeddings_dir}/{wikipedia_loader.uuid}.json"):
            # Create embeddings for the document
            embeddings= EmbeddingCreator().create_embeddings_for_document(document)
            if self.embeddings_dir:
                embeddings.save_json(f"{self.embeddings_dir}/{document.original_uuid}.json")
        # import embeddings to weaviate
        else:
            print(f"Embeddings already exists for {wikipedia_loader.uuid}")
            embeddings = Embeddings.load_json(f"{self.embeddings_dir}/{document.original_uuid}.json")
        self.weaviate_client.batch_import(embeddings)
        # Index the document to elastic search
        # success=self.elastic_client.index_document(document)
        # return success
    
    def batch_index_wikipedia_urls(self, wikipedia_urls:list, model:Literal['gpt', 'nemotron','local_nemotron','local_llama3_3']):
        for url in wikipedia_urls:
            self.index_from_wikipedia_url(url,model)

    def search(self, query:str):
        query_vector = EmbeddingCreator().embed_text(query)
        response = self.weaviate_client.hybrid_search(query,query_vector,30)
        return response
        

    def close(self):
        self.weaviate_client.close()
    
    def simple_index_from_wikipedia_url(self, wikipedia_url:str, model:Literal['gpt', 'nemotron','local_nemotron','local_llama3_3']):
        # Load the wikipedia page to document
        wikipedia_loader = WikipediaLoader(wikipedia_url)
        if self.markdown_dir:
            wikipedia_loader.save_markdown(f"{self.markdown_dir}/{wikipedia_loader.uuid}.md") # Save the markdown file
        document_no_chunks = wikipedia_loader.to_document()
        # Split the document into chunks
        document_with_chunks_no_contextualized_text = DocumentSplitter().split_document(document_no_chunks)

        if not os.path.exists(f"{self.document_dir}/{wikipedia_loader.uuid}.json"):
            # Contextualize the chunks
            print("Use fake contextualizer")
            document = asyncio.run(AsyncChunkContextualizer(model).fake_contextualize_document(document_with_chunks_no_contextualized_text))
            

            # document = ChunkContextualizer().contextualize_document(document_with_chunks_no_contextualized_text)
            # Save the document to a file
            if self.document_dir:
                document.save_json(f"{self.document_dir}/{document.original_uuid}.json")
        else:
            print(f"Document already exists for {wikipedia_loader.uuid}")
            document = Document.load_json(f"{self.document_dir}/{document_no_chunks.original_uuid}.json")

        if not os.path.exists(f"{self.embeddings_dir}/{wikipedia_loader.uuid}.json"):
            # Create embeddings for the document
            embeddings= EmbeddingCreator().create_embeddings_for_document(document)
            if self.embeddings_dir:
                embeddings.save_json(f"{self.embeddings_dir}/{document.original_uuid}.json")
        # import embeddings to weaviate
        else:
            print(f"Embeddings already exists for {wikipedia_loader.uuid}")
            embeddings = Embeddings.load_json(f"{self.embeddings_dir}/{document.original_uuid}.json")
        self.weaviate_client.batch_import(embeddings)
        # Index the document to elastic search
        # success=self.elastic_client.index_document(document)
        # return success
    
    def batch_simple_index_wikipedia_urls(self, wikipedia_urls:list, model:Literal['gpt', 'nemotron','local_nemotron','local_llama3_3']):
        for url in wikipedia_urls:
            self.simple_index_from_wikipedia_url(url,model)

def test_document_from_wikipedia_url():
    url_list=[
        "https://en.wikipedia.org/wiki/President_of_the_United_States",
        # "https://en.wikipedia.org/wiki/James_Buchanan",
        # "https://en.wikipedia.org/wiki/Harriet_Lane",
        # "https://en.wikipedia.org/wiki/List_of_presidents_of_the_United_States_who_died_in_office",
        # "https://en.wikipedia.org/wiki/James_A._Garfield"
    ]
    query="If my future wife has the same first name as the 15th first lady of the United States' mother and her surname is the same as the second assassinated president's mother's maiden name, what is my future wife's name?"

    service = KnowledgeIndexService()
    # for url in url_list:
    #     try:
    #         service.index_from_wikipedia_url(url,"gpt")
    #     except Exception as e:
    #         print(e)

    response = service.search(query)
    # print(response.objects)
    corpus = [o.properties["text_to_embed"] for o in response.objects]
    print(corpus)

    service.close()
    return corpus

if __name__ == "__main__":
    test_document_from_wikipedia_url()