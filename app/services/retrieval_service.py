# app/services/retrieval_service.py

from sqlalchemy.orm import Session
from app.database.models import Document
from app.services.embedding_service import EmbeddingService
from app.services.memory_manager import MemoryManager
from app.utils.pdf_processor import extract_text_from_pdf

class RetrievalService:
    def __init__(self, db: Session):
        self.db = db
        self.embedding_service = EmbeddingService()
        self.memory_manager = MemoryManager()

    def store_document(self, content: str):
        """Armazena um documento no banco de dados com seu embedding"""
        embedding = self.embedding_service.generate_embedding(content)
        document = Document(content=content, embedding=str(embedding))
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        return document

    def get_document_by_id(self, document_id: int):
        """Recupera um documento pelo seu ID"""
        return self.db.query(Document).filter(Document.id == document_id).first()

    def retrieve_documents(self):
        """Recupera todos os documentos armazenados"""
        return self.db.query(Document).all()

    def process_user_input(self, user_input: str):
        """Adiciona a mensagem do usuário à memória e gera uma resposta"""
        return self.memory_manager.add_message(user_input)

    def process_pdf(self, pdf_file: bytes):
        """Processa o PDF enviado, extrai o texto e armazena com seu embedding"""
        text = extract_text_from_pdf(pdf_file)
        return self.store_document(text)
