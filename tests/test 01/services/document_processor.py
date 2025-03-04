import os
import uuid
import logging
from typing import Optional, List, Dict, Any

import pytesseract
from PIL import Image

from langchain_core.documents import Document
from langchain.text_splitter import TextSplitter

from config import get_settings

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, text_splitter: TextSplitter):
        self.settings = get_settings()
        self.text_splitter = text_splitter
        self.loaders = self._get_document_loaders()
        self.image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']

    def _get_document_loaders(self) -> Dict[str, Any]:
        from langchain_community.document_loaders import (
            TextLoader,
            PyPDFLoader,
            CSVLoader,
            UnstructuredWordDocumentLoader,
            UnstructuredODTLoader,
            WebBaseLoader
        )
        return {
            '.txt': TextLoader,
            '.pdf': PyPDFLoader,
            '.csv': CSVLoader,
            '.docx': UnstructuredWordDocumentLoader,
            '.doc': UnstructuredWordDocumentLoader,
            '.odt': UnstructuredODTLoader,
            'web': WebBaseLoader
        }

    def process_document(self, document: Document) -> List[Document]:
        """Processar documento em chunks"""
        try:
            processed_docs = self.text_splitter.split_documents([document])
            
            # Adicionar metadados consistentes
            for doc in processed_docs:
                if 'id' not in doc.metadata:
                    doc.metadata['id'] = str(uuid.uuid4())
            
            return processed_docs
        except Exception as e:
            logger.error(f"Erro ao processar documento: {e}")
            return []

    def load_document(self, file_path: str) -> Optional[Document]:
        """Carregar documento baseado na extensão"""
        try:
            # Validar tamanho do arquivo
            if os.path.getsize(file_path) > self.settings.MAX_FILE_SIZE:
                logger.warning(f"Arquivo {file_path} excede o tamanho máximo permitido")
                return None

            file_ext = os.path.splitext(file_path)[[1]]("https://python.langchain.com/docs/integrations/document_loaders/needle/").lower()
            
            # Processamento de imagens
            if file_ext in self.image_extensions:
                return self._process_image(file_path)
            
            # Processamento de documentos textuais
            loader_class = self.loaders.get(file_ext)
            if not loader_class:
                logger.warning(f"Formato de arquivo não suportado: {file_ext}")
                return None
            
            loader = loader_class(file_path)
            documents = loader.load()
            
            if not documents:
                logger.warning(f"Nenhum documento carregado de {file_path}")
                return None
            
            # Adicionar metadados
            document = documents[[0]]("https://python.langchain.com/docs/integrations/document_transformers/doctran_translate_document/")
            document.metadata['id'] = str(uuid.uuid4())
            document.metadata['source'] = file_path
            document.metadata['type'] = file_ext
            
            return document
        
        except Exception as e:
            logger.error(f"Erro ao carregar documento: {e}")
            return None

    def load_url(self, url: str) -> Optional[Document]:
        """Carregar conteúdo de URL"""
        try:
            loader = self.loaders['web'](url)
            documents = loader.load()
            
            if not documents:
                logger.warning(f"Nenhum conteúdo carregado da URL: {url}")
                return None
            
            # Adicionar metadados
            document = documents[[0]]("https://python.langchain.com/docs/integrations/document_transformers/doctran_translate_document/")
            document.metadata['id'] = str(uuid.uuid4())
            document.metadata['source'] = url
            document.metadata['type'] = 'web'
            
            return document
        
        except Exception as e:
            logger.error(f"Erro ao carregar URL: {e}")
            return None

    def _process_image(self, image_path: str) -> Optional[Document]:
        """Processar imagem usando OCR"""
        try:
            # Verificar dependências
            if not pytesseract or not Image:
                logger.warning("pytesseract ou PIL não instalados")
                return None

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
            logger.error(f"Erro ao processar imagem: {e}")
            return None
