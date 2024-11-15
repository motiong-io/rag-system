from app.model.document_model import Document
from app.services.wikipedia_loader import WikipediaLoader
from app.services.document_splitter import DocumentSplitter
from app.services.chunk_contextualizer import ChunkContextualizer

class KnowledgeIndexService:
    def __init__(self,save_markdown:bool=True,save_document:bool=True,save_embeddings:bool=True) -> None:
        self.markdown_dir = "assets/dataset/markdown_files" if save_markdown else None
        self.document_dir="assets/dataset/document_json" if save_document else None
        self.embeddings_dir="assets/dataset/embeddings_list" if save_embeddings else None

    def index_from_wikipedia_url(self, wikipedia_url:str) -> Document:
        # Load the wikipedia page to document
        wikipedia_loader = WikipediaLoader(wikipedia_url)
        if self.markdown_dir:
            wikipedia_loader.save_markdown(f"{self.markdown_dir}/{wikipedia_loader.uuid}.md") # Save the markdown file
        document_no_chunks = wikipedia_loader.to_document()
        # Split the document into chunks
        document_with_chunks_no_contextualized_text = DocumentSplitter().split_document(document_no_chunks)
        # Contextualize the chunks
        document = ChunkContextualizer().contextualize_document(document_with_chunks_no_contextualized_text)
        # Save the document to a file
        if self.document_dir:
            document.save_json(f"{self.document_dir}/{document.original_uuid}.json")
        
        
        return document
    






def test_document_from_wikipedia_url():
    url="https://en.wikipedia.org/wiki/List_of_tallest_buildings_in_New_York_City"
    service = KnowledgeIndexService()
    service.index_from_wikipedia_url(url)

if __name__ == "__main__":
    test_document_from_wikipedia_url()