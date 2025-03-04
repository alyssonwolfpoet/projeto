from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Configurações do banco de dados
    DATABASE_URL: str = "postgresql://admin:admin@localhost:5432/llm_db"
    COLLECTION_NAME: str = "rag_documents"
    
    # Modelos
    EMBEDDING_MODEL: str = "nomic-embed-text"
    LLM_MODEL: str = "llama3.2"
    
    # Configurações de segurança
    SECRET_KEY: str = "sua-chave-secreta"
    
    # Configurações de upload
    UPLOAD_FOLDER: str = "/tmp/uploads"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10 MB
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings():
    return Settings()
