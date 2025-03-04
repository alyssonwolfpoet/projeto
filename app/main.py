# app/main.py

from fastapi import FastAPI, Depends, HTTPException, File, UploadFile
from sqlalchemy.orm import Session
from app.database import SessionLocal, init_db
from app.services.retrieval_service import RetrievalService

# Inicializa o banco de dados
init_db()

app = FastAPI()

# Função para obter a sessão do banco de dados
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/documents/pdf/")
async def upload_pdf(pdf_file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Faz upload de um PDF, extrai o texto e armazena no banco de dados com seu embedding"""
    pdf_content = await pdf_file.read()  # Lê o conteúdo do arquivo PDF
    retrieval_service = RetrievalService(db)
    
    # Processa o PDF e armazena
    document = retrieval_service.process_pdf(pdf_content)
    
    return {"id": document.id, "content": document.content}

@app.get("/documents/{document_id}")
def get_document(document_id: int, db: Session = Depends(get_db)):
    """Recuperar um documento pelo ID"""
    retrieval_service = RetrievalService(db)
    document = retrieval_service.get_document_by_id(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"id": document.id, "content": document.content}

@app.get("/documents/")
def get_documents(db: Session = Depends(get_db)):
    """Recuperar todos os documentos armazenados"""
    retrieval_service = RetrievalService(db)
    documents = retrieval_service.retrieve_documents()
    return [{"id": doc.id, "content": doc.content} for doc in documents]

@app.post("/chat/")
def chat_with_model(user_input: str, db: Session = Depends(get_db)):
    """Interagir com o modelo e manter o contexto na memória"""
    retrieval_service = RetrievalService(db)
    response = retrieval_service.process_user_input(user_input)
    return {"response": response}
