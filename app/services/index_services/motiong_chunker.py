from motiongchunker.advanced_chunker import AdvancedChunker

from motiongchunker.contextualizer.chunk_contextualizer import AsyncChunkContextualizer
from motiongchunker.chunker.charactor_chunker import RecursiveCharacterChunker
from motiongchunker.libs.language import Language



def read_file(file_path):
    with open(file_path, "r") as f:
        return f.read()


async def manual_test_advanced_chunker():
    # Read the content of the file
    text = read_file("assets/markdown_files/0a9f30899b05b79f9fe1ad492dbb5047.md")

    # Create an instance of the RecursiveCharacterChunker
    basic_chunker = RecursiveCharacterChunker.from_language(
        Language.MARKDOWN, chunk_size=1500, chunk_overlap=100, strip_whitespace=True, keep_separator='start'
    )

    # Create an instance of the ChunkContextualizer
    llm_base_url = "http://10.4.32.2:8001/v1"
    llm_api_key = "abc"
    llm_name = "llama3_3"
    basic_chunk_contexualizer = AsyncChunkContextualizer(llm_base_url, llm_api_key, llm_name)
    
    # Create an instance of the AdvancedChunker
    advanced_chunker = AdvancedChunker(basic_chunker, basic_chunk_contexualizer)

    # Split the document into chunks
    chunks, context_infos = await advanced_chunker.split_chunks(text)

    print(f"RecursiveCharacterChunker from Language: Split into {len(chunks)} chunks")
    print(context_infos)


# Run the async test
if __name__ == "__main__":
    import asyncio
    asyncio.run(manual_test_advanced_chunker())