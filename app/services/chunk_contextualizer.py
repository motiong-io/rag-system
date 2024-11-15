from langchain.schema import SystemMessage, HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from typing import Any
from app.config import env
from app.model.document_model import Document,Chunk
from tqdm import tqdm


class ChunkContextualizer:
    def __init__(self) -> None:
        self.openai_client = ChatOpenAI(model="gpt-4o-mini",
                                        api_key=env.openai_api_key,
                                        base_url="http://api-gw.motiong.net:5000/api/openai/ve/v1")

    def situate_context(self, doc: str, chunk: str) -> tuple[str, Any]:
        DOCUMENT_CONTEXT_PROMPT = """
        <document>
        {doc_content}
        </document>
        """

        CHUNK_CONTEXT_PROMPT = """
        Here is the chunk we want to situate within the whole document
        <chunk>
        {chunk_content}
        </chunk>

        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
        Answer only with the succinct context and nothing else.
        """

        def generate_prompt(doc, chunk):
            return [
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content=DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc)),
                    HumanMessage(content=CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk))
                ]
 
        prompt = generate_prompt(doc, chunk)
        response = self.openai_client.invoke(prompt)
        # print(response.content)
        return response.content, response.response_metadata
    
    def contextualize_document(self, document: Document) -> Document:
        new_chunks = []
        for chunk in tqdm(document.chunks, desc="Contextualizing chunks"):
            chunk.contextualized_text, _ = self.situate_context(document.content, chunk.content)
            new_chunks.append(chunk)
        document.chunks = new_chunks
        return document
