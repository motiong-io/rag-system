from typing import Literal
from motiongreactor.llm.basic_llm import BasicLLM
from app.config import env


def get_llm(llm_name:Literal['llama3_3','gpt-4o-mini'])->BasicLLM:

    if llm_name == 'llama3_3':
        return BasicLLM(base_url=env.nvidia_local_base_url,api_key="abc",model="llama3_3")
    
    elif llm_name == 'gpt-4o-mini':
        return BasicLLM(base_url=env.openai_base_url,api_key=env.openai_api_key,model="gpt-4o-mini")

    else:
        raise ValueError(f"Model {llm_name} not found")
        