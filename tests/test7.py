from typing import List, Union
from langchain_community.document_loaders import (
    WebBaseLoader,
    CSVLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredODTLoader,
    TextLoader
)
from langchain_core.documents import Document

def load_document_from_url(url: str) -> List[Document]:
    """Carrega documentos de uma URL"""
    loader = WebBaseLoader(url)
    return loader.load()

def load_csv_file(file_path: str) -> List[Document]:
    """Carrega arquivos CSV"""
    loader = CSVLoader(file_path)
    return loader.load()

def load_pdf_file(file_path: str) -> List[Document]:
    """Carrega arquivos PDF"""
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

def load_docx_file(file_path: str) -> List[Document]:
    """Carrega arquivos DOCX"""
    loader = UnstructuredWordDocumentLoader(file_path)
    return loader.load()

def load_odt_file(file_path: str) -> List[Document]:
    """Carrega arquivos ODT"""
    loader = UnstructuredODTLoader(file_path)
    return loader.load()

def load_text_file(file_path: str) -> List[Document]:
    """Carrega arquivos de texto"""
    loader = TextLoader(file_path)
    return loader.load()

def detect_and_load_document(file_path: str) -> List[Document]:
    """Detecta e carrega diferentes tipos de documentos"""
    file_extension = file_path.split('.')[-1].lower()
    
    loaders = {
        'csv': load_csv_file,
        'pdf': load_pdf_file,
        'docx': load_docx_file,
        'doc': load_docx_file,
        'odt': load_odt_file,
        'txt': load_text_file
    }
    
    loader_func = loaders.get(file_extension)
    if loader_func:
        return loader_func(file_path)
    else:
        raise ValueError(f"Tipo de arquivo não suportado: {file_extension}")
