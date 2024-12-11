from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from app.model.document_model import Document,Chunk

class DocumentSplitter:
    def __init__(self, chunk_size: int = 1000, overlap_size: int = 100):

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
            length_function=len,
            is_separator_regex=False
        )

    def split_text(self,text: str) -> List[str]:
        """Split the text_str into chunks_list."""
        return self.spliter.split_text(text)
    
    def split_document(self, document: Document) -> Document:
        """Split the document into chunks."""
        chunk_content_list = self.split_text(document.content)
        chunks = []
        for i, chunk_content in enumerate(chunk_content_list):
            chunks.append(Chunk(chunk_id=document.doc_id+'_'+str(i), 
                                original_index=i, 
                                content=chunk_content))
        document.update_chunks(chunks)
        return document