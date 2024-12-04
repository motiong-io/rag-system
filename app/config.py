from pydantic_settings import BaseSettings
from pydantic import ConfigDict

class Settings(BaseSettings):

    model_config = ConfigDict(env_file=".env")
    
    openai_api_key: str
    openai_base_url: str
    
    weaviate_url: str
    weaviate_api_key: str
    anthropic_api_key: str
    voyage_api_key: str
    cohere_api_key: str

    elastic_search_url: str
    elastic_search_api_key: str

    

env = Settings()