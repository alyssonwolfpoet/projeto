import os
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from .document_loaders import detect_and_load_document, load_document_from_url
from .vector_store import create_vector_store

app = FastAPI()

# Configurações CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas as origens
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload-documents/")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Endpoint para upload de documentos"""
    try:
        documents = []
        for file in files:
            # Salva o arquivo temporariamente
            file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
            
            # Carrega o documento
            doc_chunks = detect_and_load_document(file_path)
            documents.extend(doc_chunks)
            
            # Opcional: remove o arquivo após processamento
            os.remove(file_path)
        
        # Cria o vetor store com os documentos carregados
        vector_store = create_vector_store(documents)
        
        return {
            "status": "success", 
            "message": f"{len(documents)} documentos carregados com sucesso"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-url/")
async def upload_document_from_url(url: str):
    """Endpoint para upload de documento via URL"""
    try:
        documents = load_document_from_url(url)
        vector_store = create_vector_store(documents)
        
        return {
            "status": "success", 
            "message": f"{len(documents)} documentos carregados da URL"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/")
async def generate_response(query: str):
    """Endpoint para geração de resposta com RAG"""
    try:
        # Implementação do RAG (Retrieval-Augmented Generation)
        response = perform_rag_query(query)
        
        return {
            "response": response
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Arquivo vector_store.py
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from typing import List

def create_vector_store(documents: List[Document]):
    """Cria um vetor store com os documentos"""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Cria o vetor store em memória
    vector_store = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings
    )
    
    return vector_store

def perform_rag_query(query: str, top_k: int = 4):
    """Realiza a busca e geração de resposta"""
    from langchain_ollama.chat_models import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    
    # Reutiliza o vetor store e embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma(embedding_function=embeddings)
    
    # Recupera documentos relevantes
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    
    # Configura o modelo de linguagem
    llm = ChatOllama(model="llama3")
    
    # Template do prompt
    template = """Use o seguinte contexto para responder a pergunta:
    {context}
    
    Pergunta: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Cadeia de processamento
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Gera a resposta
    response = rag_chain.invoke(query)
    
    return response

# Para rodar o servidor
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0",
