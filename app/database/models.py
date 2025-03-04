# app/database/models.py

from sqlalchemy import Column, Integer, String
from app.database import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(String, index=True)
    embedding = Column(String)  # O embedding será armazenado como string (poderia ser JSON ou base64)
