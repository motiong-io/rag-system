# Set up API key and do the necessary imports
from agentjo import *
from app.config import env


def llm(system_prompt: str, user_prompt: str) -> str:
    ''' Here, we use OpenAI for illustration, you can change it to your own LLM '''
    # ensure your LLM imports are all within this function
    from openai import OpenAI
    
    # define your own LLM here
    client = OpenAI()
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        temperature = 0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content

# # Example External Function
# def binary_to_decimal(binary_string: str) -> int:
#     '''Converts binary_string to integer of base 10'''
#     return int(str(binary_string), 2)

from app.repo.contextual_vector_db import ContextualVectorDB
from app.service.bm25 import create_elasticsearch_bm25_index,retrieve_advanced
from app.service.rerank import only_rerank

def hybrid_search(query: str):
    '''Search Context based on query using a hybrid of BM25 and Semantic Search'''
    db = ContextualVectorDB("q19_contextual_db")
    db.load_db()
    es_bm25=create_elasticsearch_bm25_index(db)
    final_results, semantic_count, bm25_count,raw_conbined=retrieve_advanced(query, db, es_bm25, k=30)
    final_results_rerank=only_rerank(query,final_results, k=1)
    return final_results_rerank


# Create your agent by specifying name and description
system_prompt = """
You are an intelligent assistant specialized in solving multi-hop questions. When a user presents a query, follow these steps:

1. **Understand the Query**:
   - Identify the core objective of the user's question.
   - Recognize key entities or topics involved.

2. **Decompose the Query**:
   - Break down the complex question into smaller, manageable sub-questions.
   - Ensure each sub-question is specific and answerable.

3. **Sequential Search**:
   - Conduct context searches for each sub-question to gather necessary information.
   - Use the answers to inform subsequent sub-questions, progressively building towards a complete solution.

4. **Integrate Findings**:
   - Merge the answers of all sub-questions into a coherent final response.
   - Maintain transparency about information sources and reasoning.

5. **Iterate and Refine**:
   - Adjust question decomposition or search strategy if information is insufficient.
   - Use user feedback to refine the solving process.
Engage with the user to confirm your understanding and clarify as needed, ensuring the solution meets their needs.
"""

system_prompt = "Be patient and helpful.Do not forget the user's question"



my_agent = Agent('Helpful assistant', system_prompt, llm = llm)
my_agent.status()


# Assign functions
my_agent.assign_functions(function_list = [hybrid_search])




# Do the task by subtasks. This does generation to fulfil task
my_agent.reset()
query="As of August 4, 2024, in what state was the first secretary of the latest United States federal executive department born?"
    

output = my_agent.run(query)
