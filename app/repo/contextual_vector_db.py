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

load_dotenv()

class ContextualVectorDB:
    def __init__(self, name: str, voyage_api_key=None, anthropic_api_key=None,openai_api_key=None):
        if voyage_api_key is None:
            voyage_api_key = os.getenv("VOYAGE_API_KEY")
        if anthropic_api_key is None:
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        
        self.voyage_client = voyageai.Client(api_key=voyage_api_key)
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
        # self.openai_client = ChatOpenAI(model="/data/xinference_llm/.cache/modelscope/hub/LLM-Research/Meta-Llama-3___1-70B-Instruct-AWQ-INT4",api_key="api_key", base_url="http://10.1.3.6:8001/v1",max_tokens=30000)
        self.openai_client = ChatOpenAI(model="gpt-4o-mini",api_key=openai_api_key)
        self.embedding_client = openai.Client(api_key=openai_api_key)
        self.name = name
        self.embeddings = []
        self.metadata = []
        self.query_cache = {}
        self.db_path = f"./data/{name}/contextual_vector_db.pkl"

        self.token_counts = {
            'input': 0,
            'output': 0,
            'cache_read': 0,
            'cache_creation': 0
        }
        self.token_lock = threading.Lock()

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

        # response = self.anthropic_client.beta.prompt_caching.messages.create(
        #     model="claude-3-haiku-20240307",
        #     max_tokens=2000,
        #     temperature=0.0,
        #     messages=[
        #         {
        #             "role": "user", 
        #             "content": [
        #                 {
        #                     "type": "text",
        #                     "text": DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc),
        #                     "cache_control": {"type": "ephemeral"} #we will make use of prompt caching for the full documents
        #                 },
        #                 {
        #                     "type": "text",
        #                     "text": CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk),
        #                 },
        #             ]
        #         },
        #     ],
        #     extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
        # )
        # return response.content[0].text, response.usage
    
        def generate_prompt(doc, chunk):
            return [
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content=DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc)),
                    HumanMessage(content=CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk))
                ]
 

        prompt = generate_prompt(doc, chunk)
    
        # 执行 API 调用
        response = self.openai_client.invoke(prompt)
        # print(response.content)
        return response.content, response.response_metadata


    def load_data(self, dataset: List[Dict[str, Any]], parallel_threads: int = 5):
        if self.embeddings and self.metadata:
            print("Vector database is already loaded. Skipping data loading.")
            return
        if os.path.exists(self.db_path):
            print("Loading vector database from disk.")
            self.load_db()
            return

        texts_to_embed = []
        metadata = []
        # total_chunks = sum(len(doc['chunks']) for doc in dataset)

        def process_chunk(doc, chunk):
            
            # print(f"Processing{chunk['chunk_id']}")
            #for each chunk, produce the context
            contextualized_text, usage = self.situate_context(doc['content'], chunk['content'])
            with self.token_lock:
                self.token_counts['input'] += 1
                self.token_counts['output'] += 1
                self.token_counts['cache_read'] += 1
                self.token_counts['cache_creation'] += 1
            
            return {
                #append the context to the original text chunk
                'text_to_embed': f"{chunk['content']}\n\n{contextualized_text}",
                'metadata': {
                    'doc_id': doc['doc_id'],
                    'original_uuid': doc['original_uuid'],
                    'chunk_id': chunk['chunk_id'],
                    'original_index': chunk['original_index'],
                    'original_content': chunk['content'],
                    'contextualized_content': contextualized_text
                }
            }

        # print(f"Processing {total_chunks} chunks with {parallel_threads} threads")
        # with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
        #     futures = []
        #     for doc in dataset:
        #         total_chunks = len(doc['chunks'])

        #         for chunk in doc['chunks']:
        #             # time.sleep(0.1) #to avoid hitting the API rate limit
        #             futures.append(executor.submit(process_chunk, doc, chunk))
                
        #     for future in tqdm(as_completed(futures), total=total_chunks, desc="Processing chunks"):
        #         result = future.result()
        #         texts_to_embed.append(result['text_to_embed'])
        #         metadata.append(result['metadata'])

        for doc in dataset:
            total_chunks = len(doc['chunks'])
            for chunk in tqdm(doc['chunks'],total=total_chunks,desc=f"Processing document {doc['doc_id']}"):
                result = process_chunk(doc, chunk)
                texts_to_embed.append(result['text_to_embed'])
                metadata.append(result['metadata'])


        self._embed_and_store(texts_to_embed, metadata)
        self.save_db()

        #logging token usage
        print(f"Contextual Vector database loaded and saved. Total chunks processed: {len(texts_to_embed)}")
        print(f"Total input tokens without caching: {self.token_counts['input']}")
        print(f"Total output tokens: {self.token_counts['output']}")
        print(f"Total input tokens written to cache: {self.token_counts['cache_creation']}")
        print(f"Total input tokens read from cache: {self.token_counts['cache_read']}")
        
        total_tokens = self.token_counts['input'] + self.token_counts['cache_read'] + self.token_counts['cache_creation']
        savings_percentage = (self.token_counts['cache_read'] / total_tokens) * 100 if total_tokens > 0 else 0
        print(f"Total input token savings from prompt caching: {savings_percentage:.2f}% of all input tokens used were read from cache.")
        print("Tokens read from cache come at a 90 percent discount!")

    # we use voyage AI here for embeddings. Read more here: https://docs.voyageai.com/docs/embeddings
    def _embed_and_store(self, texts: List[str], data: List[Dict[str, Any]]):
        batch_size = 1
        result = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            # print(f"Processing batch {i // batch_size + 1} of {len(texts) // batch_size + 1}")
            batch_embeddings = self.voyage_client.embed(
            texts[i : i + batch_size],
            model="voyage-3"
            ).embeddings
            result.extend(batch_embeddings)
        self.embeddings = result
        self.metadata = data

    # def _embed_and_store(self, texts: List[str], data: List[Dict[str, Any]]):
    #     result = []
    #     for i in range(0, len(texts)):
    #         print(f"Processing {i+1} of {len(texts)+1}")
    #         output = self.embedding_client.embeddings.create(input=texts[i],model="text-embedding-3-small")
    #         embedding=output.data[0].embedding
    #         result.append(embedding)
    #     self.embeddings = result
    #     self.metadata = data




    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
        else:
            query_embedding = self.voyage_client.embed([query], model="voyage-3").embeddings[0]
            # query_embedding = self.embedding_client.embeddings.create(input=query,model="text-embedding-3-small")

            self.query_cache[query] = query_embedding

        if not self.embeddings:
            raise ValueError("No data loaded in the vector database.")

        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:k]
        
        top_results = []
        for idx in top_indices:
            result = {
                "metadata": self.metadata[idx],
                "similarity": float(similarities[idx]),
            }
            top_results.append(result)
        return top_results

    def save_db(self):
        data = {
            "embeddings": self.embeddings,
            "metadata": self.metadata,
            "query_cache": json.dumps(self.query_cache),
        }
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as file:
            pickle.dump(data, file)

    def load_db(self):
        if not os.path.exists(self.db_path):
            raise ValueError("Vector database file not found. Use load_data to create a new database.")
        with open(self.db_path, "rb") as file:
            data = pickle.load(file)
        self.embeddings = data["embeddings"]
        self.metadata = data["metadata"]
        self.query_cache = json.loads(data["query_cache"])

    def add_data(self, new_embeddings, new_metadata):
        # Step 1: Load existing data
        try:
            self.load_db()
        except ValueError:
            # 如果数据库文件不存在，则初始化空结构
            self.embeddings = []
            self.metadata = []
            self.query_cache = {}

        # Step 2: Update the embeddings and metadata
        self.embeddings.append(new_embeddings)  # 假设是单个对象，也可以是多个对象
        self.metadata.append(new_metadata)

        # Update query_cache if needed
        # self.query_cache.update(new_cache_items)

        # Step 3: Save the updated data
        self.save_db()

def fast_create_by_json(db_name:str,raw_database_path:str):
    with open(raw_database_path, 'r') as f:
        transformed_dataset = json.load(f)
    contextual_db = ContextualVectorDB(db_name)
    contextual_db.load_data(transformed_dataset, parallel_threads=1)
    return contextual_db



def create():
    with open('scripts/chunked/codebase_chunks_1.json', 'r') as f:
        transformed_dataset = json.load(f)

    contextual_db = ContextualVectorDB("q19_contextual_db")
    contextual_db.load_data(transformed_dataset, parallel_threads=1)
    # q="As of August 4, 2024, in what state was the first secretary of the latest United States federal executive department born?"
    # results = contextual_db.search(q)
    # print(results)




def test():
    with open('scripts/chunked/test.json', 'r') as f:
        transformed_dataset = json.load(f)
    contextual_db = ContextualVectorDB("my_contextual_db_test")
    contextual_db.load_data(transformed_dataset, parallel_threads=1)
    q="Explain the primary purpose of using a DiffExecutor in a differential fuzzing setup. What advantages does it provide over using a single executor?"
    results = contextual_db.search(q)
    print(results)

if __name__ == "__main__":
    create()