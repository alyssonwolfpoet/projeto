import os
from typing import Optional, List
import uuid
from langchain_core.documents import Document
from langchain.text_splitter import TextSplitter

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredODTLoader,
    WebBaseLoader
)
import pytesseract
from PIL import Image

class DocumentProcessor:
    def __init__(self, text_splitter: TextSplitter):
        self.text_splitter = text_splitter = text_splitter
        self.loaders = {
            '.txt': TextLoader,
            '.pdf': PyPDFLoader,
            '.csv': CSVLoader,
            '.docx': UnstructuredWordDocumentLoader,
            '.doc': UnstructuredWordDocumentLoader,
            '.odt': UnstructuredODTLoader
        }
        self.image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']

    def process_document(self, document: Document) -> List[Document]:
        """Processar documento em chunks"""
        return self.text_splitter.split_documents([document])

    def load_document(self, file_path: str) -> Optional[Document]:
        """Carregar documento baseado na extensão"""
        try:
            file_ext = os.path.splitext(file_path)[[1]].lower()
            
            # Processamento de imagens
            if file_ext in self.image_extensions:
                return self._process_image(file_path)
            
            # Processamento de documentos textuais
            loader_class = self.loaders.get(file_ext)
            if not loader_class:
                return None
            
            loader = loader_class(file_path)
            documents = loader.load()
            
            # Adicionar metadados
            for doc in documents:
                doc.metadata['id'] = str(uuid.uuid4())
                doc.metadata['source'] = file_path
            
            return documents[[0]] if documents else None
        
        except Exception as e:
            print(f"Erro ao carregar documento: {e}")
            return None

    def load_url(self, url: str) -> Optional[Document]:
        """Carregar conteúdo de URL"""
        try:
            loader = WebBaseLoader(url)
            documents = loader.load()
            
            # Adicionar metadados
            for doc in documents:
                doc.metadata['id'] = str(uuid.uuid4())
                doc.metadata['source'] = url
            
            return documents[[0]] if documents else None
        
        except Exception as e:
            print(f"Erro ao carregar URL: {e}")
            return None

    def _process_image(self, image_path: str) -> Optional[Document]:
        """Processar imagem usando OCR"""
        try:
            # Extrair texto da imagem
            text = pytesseract.image_to_string(Image.open(image_path))
            
            # Criar documento
            document = Document(
                page_content=text,
                metadata={
                    'id': str(uuid.uuid4()),
                    'source': image_path,
                    'type': 'image'
                }
            )
            
            return document
        
        except Exception as e:
            print(f"Erro ao processar imagem: {e}")
            return None
