from typing import Any, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.repo.weaviate_cloud import WeaviateClient
from openai import OpenAI
import anthropic
from app.config import env


class KnowledgeIndexService:
    def __init__(self):
        # vector saved in weaviate
        # init weaviate client
        self.weaviate_client = WeaviateClient(url=env.weaviate_url, api_key=env.weaviate_api_key)
        self.weaviate_client.connect()
        self.weaviate_collection_name = "RAG_System"

        # use openai embedding service to get vector
        self.openai_api_key = env.openai_api_key
        self.openai_client = OpenAI(api_key=self.openai_api_key)

        # use anthropic for contextual embedding description
        self.anthropic_api_key = env.anthropic_api_key
        self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)

    def split2chunks(self, text: str,chunk_size:int,chunk_overlap:int) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = splitter.split_text(text)
        return chunks
    
    def text2vector(self, text,model_name="text-embedding-3-small"):
        response = self.openai_client.embeddings.create(input=text,model=model_name)
        return response.data[0].embedding

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

        response = self.anthropic_client.beta.prompt_caching.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            temperature=0.0,
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc),
                            "cache_control": {"type": "ephemeral"} #we will make use of prompt caching for the full documents
                        },
                        {
                            "type": "text",
                            "text": CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk),
                        },
                    ]
                },
            ],
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
        )
        return response.content[0].text, response.usage

    def index(self,title:str,long_text:str, properties:dict ,vector:List[float]):
        chunks = self.split2chunks(long_text,chunk_size=1000,chunk_overlap=100)
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            vector = self.text2vector(chunk)
            contextual, usage = self.situate_context(long_text, chunk)
            properties['title'] = title
            properties['text'] = chunk
            properties['contextual'] = contextual
            properties['usage'] = usage
            uuid = self.weaviate_client.create_object(collection_name=self.weaviate_collection_name, properties=properties, vector=vector)
            print(f"Indexed chunk {i + 1}/{total_chunks} with uuid: {uuid} --> {len(chunk)}")
        
    def close(self):
        self.weaviate_client.close()

if __name__ == "__main__":
    pass