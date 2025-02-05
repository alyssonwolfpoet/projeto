from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Configurações do banco de dados
    DATABASE_URL: str = "postgresql://admin:admin@localhost:5432/llm_db"
    COLLECTION_NAME: str = "rag_documents"
    
    # Modelos
    EMBEDDING_MODEL: str = "llama3.2"
    LLM_MODEL: str = "llama3.2"
    
    # Configurações de segurança
    SECRET_KEY: str = "sua-chave-secreta"
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
