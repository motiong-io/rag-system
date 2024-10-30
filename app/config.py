from pydantic_settings import BaseSettings
from pydantic import ConfigDict

class Settings(BaseSettings):

    model_config = ConfigDict(env_file=".env")
    
    openai_api_key: str
    weaviate_url: str
    weaviate_api_key: str
    anthropic_api_key: str
    
env = Settings()