from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

from app.model.document_json import DocumentJSON

class DocumentSplitter:
    def __init__(self, chunk_size: int = 1000, overlap_size: int = 100):

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
            length_function=len,
            is_separator_regex=False
        )

    def split(self, text: str) -> List[str]:
        return self.spliter.split_text(text)
    
    def split_document(self, document: DocumentJSON) -> List[str]:
        