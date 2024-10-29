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

# Example External Function
def binary_to_decimal(binary_string: str) -> int:
    '''Converts binary_string to integer of base 10'''
    return int(str(binary_string), 2)

# Create your agent by specifying name and description
my_agent = Agent('Helpful assistant', 'You are a generalist agent', llm = llm)
my_agent.status()


# Assign functions
my_agent.assign_functions(function_list = [binary_to_decimal])


# Do the task by subtasks. This does generation to fulfil task
my_agent.reset()
output = my_agent.run('Give me 5 words rhyming with cool, and then make a 4-sentence poem using them')