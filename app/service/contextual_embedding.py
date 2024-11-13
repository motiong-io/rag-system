import os
import pickle
import json
import numpy as np
import voyageai
from typing import List, Dict, Any
from tqdm import tqdm
import anthropic
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
import openai
from app.repo.weaviate_cloud import WeaviateClient
from app.config import env

class ContextualEmbeddingService:
    def __init__(self, voyage_api_key=None, openai_api_key=None):
        if voyage_api_key is None:
            voyage_api_key = env.voyage_api_key
        if openai_api_key is None:
            openai_api_key = env.openai_api_key

        self.voyage_client = voyageai.Client(api_key=voyage_api_key) # Embedding service 
        self.openai_client = ChatOpenAI(model="gpt-4o-mini",api_key=openai_api_key,base_url="http://api-gw.motiong.net:5000/api/openai/ve/v1") # Contextualization service
        self.weaviate_client = WeaviateClient()
        self.weaviate_collection_name = "ContextualVectors"

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
    

    def process_chunk(self, doc, chunk):
        
        contextualized_text, usage = self.situate_context(doc['content'], chunk['content'])

        return {
            #append the context to the original text chunk
            'text_to_embed': f"{chunk['content']}\n\n{contextualized_text}",
            'metadata': {
                'doc_id': doc['doc_id'],
                'original_uuid': doc['original_uuid'],
                'chunk_id': chunk['chunk_id'],
                'original_index': chunk['original_index'],
                'original_content': chunk['content'],
                'contextualized_content': contextualized_text,
                'related_point':[]
            }
        }
    
    def embed_and_save_chunk(self, doc, chunk):
        data = self.process_chunk(doc, chunk)
        vector = self.voyage_client.embed(
            data['text_to_embed'],
            model="voyage-3"
            ).embeddings
        
        self.weaviate_client.create_object(self.weaviate_collection_name, data['metadata'], vector)
        return True

    def load_and_embed_documents(self, docs):
        for doc in docs:
            for chunk in doc['chunks']:
                self.embed_and_save_chunk(doc, chunk)
        return True




if __name__ == "__main__":
    # Load documents
    with open("data/q19_contextual_db/docs.json", "r") as f:
        docs = json.load(f)
    
    service = ContextualEmbeddingService()
    service.load_and_embed_documents(docs)
    print("Documents embedded and saved successfully")