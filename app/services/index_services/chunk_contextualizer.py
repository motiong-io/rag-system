from langchain.schema import SystemMessage, HumanMessage
# from langchain_openai.chat_models import ChatOpenAI
from typing import Any
from app.config import env
from app.model.document_model import Document,Chunk
from tqdm import tqdm



# class ChunkContextualizer:
#     def __init__(self) -> None:
#         # self.openai_client = ChatOpenAI(model="gpt-4o-mini",
#         #                                 api_key=env.openai_api_key,
#         #                                 base_url="http://api-gw.motiong.net:5000/api/openai/ve/v1")
#         self.openai_client = ChatOpenAI(model="nemotron-70b",
#                                         api_key="abc",
#                                         base_url="http://10.4.32.1:8001/v1")

#     def situate_context(self, doc: str, chunk: str) -> tuple[str, Any]:
#         DOCUMENT_CONTEXT_PROMPT = """
#         <document>
#         {doc_content}
#         </document>
#         """

#         CHUNK_CONTEXT_PROMPT = """
#         Here is the chunk we want to situate within the whole document
#         <chunk>
#         {chunk_content}
#         </chunk>

#         Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
#         Answer only with the succinct context and nothing else.
#         """

#         def generate_prompt(doc, chunk):
#             return [
#                     SystemMessage(content="You are a helpful assistant."),
#                     HumanMessage(content=DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc)),
#                     HumanMessage(content=CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk))
#                 ]
 
#         prompt = generate_prompt(doc, chunk)
#         response = self.openai_client.invoke(prompt)
#         # print(response.content)
#         return response.content, response.response_metadata
    
#     def contextualize_document(self, document: Document) -> Document:
#         new_chunks = []
#         for chunk in tqdm(document.chunks, desc="Contextualizing chunks"):
#             chunk.contextualized_text, _ = self.situate_context(document.content, chunk.content)
#             print(chunk.contextualized_text)
#             new_chunks.append(chunk)
#         document.chunks = new_chunks
#         return document


from openai import AsyncOpenAI
import asyncio
from tqdm.asyncio import tqdm
from typing import Literal



class AsyncChunkContextualizer:
    def __init__(self, model:Literal["gpt", "nemotron","local_nemotron"]) -> None:
        if model == "gpt":
            self.client = AsyncOpenAI(base_url=env.openai_base_url, api_key=env.openai_api_key)
            self.model = "gpt-4o-mini"
        elif model == "nemotron":
            self.client = AsyncOpenAI(base_url=env.nvidia_base_url, api_key=env.nvidia_api_key)
            self.model = "nvidia/llama-3.1-nemotron-70b-instruct"
        elif model == "local_nemotron":
            self.client = AsyncOpenAI(base_url=env.nvidia_local_base_url, api_key="abc")
            self.model = "nemotron-70b"
        else:
            raise ValueError("Invalid model")

    async def situate_context(self, doc: str, chunk: str) -> tuple[str, Any]:
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
                    {"role": "system", "content":"You are a helpful assistant."},
                    {"role": "user", "content":DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc)},
                    {"role": "user", "content":CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk)},
                ]
 
        prompt = generate_prompt(doc, chunk)
        response = await self.client.chat.completions.create(messages=prompt,model=self.model)
        # print(response)
        return response.choices[0].message.content, response.choices[0].message
    

    # async def contextualize_document(self, document: Document) -> Document:
    #     new_chunks = []
    #     for chunk in tqdm(document.chunks, desc="Contextualizing chunks"):
    #         chunk.contextualized_text, _ = await self.situate_context(document.content, chunk.content)
    #         new_chunks.append(chunk)
    #     document.chunks = new_chunks
    #     return document

    async def contextualize_document(self, document: Document) -> Document:
        tasks = []

        for chunk in document.chunks:
            task = self.situate_context(document.content, chunk.content)
            tasks.append(task)

        try:
            # 使用 tqdm 的 gather 处理任务
            results = await tqdm.gather(*tasks, desc="Contextualizing chunks")
        except Exception as e:
            # 捕获异常，取消所有未完成任务
            for task in tasks:
                task.cancel()
            raise e  # 重新抛出异常
    
        for chunk, (contextualized_text, _) in zip(document.chunks, results):
            chunk.contextualized_text = contextualized_text

        return document


# def test_contextualize_document():
#     document=Document.load_json('assets/dataset/document_json/1ddadbaf23ada8730ff72097d7101243.json')
#     contextualizer = ChunkContextualizer()
#     contextualized_document = contextualizer.contextualize_document(document)
#     contextualized_document.save_json('assets/dataset/document_json/test.json')

def test_async_contextualize_document():
    document=Document.load_json('assets/dataset/document_json/test.json')
    contextualizer = AsyncChunkContextualizer()
    contextualized_document = asyncio.run(contextualizer.contextualize_document(document))
    contextualized_document.save_json('assets/dataset/document_json/test_async.json')


if __name__ == "__main__":
    # test_contextualize_document()
    test_async_contextualize_document()