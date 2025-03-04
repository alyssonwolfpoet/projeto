# app/services/rag_service.py
import logging
from typing import List, Optional

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import Settings
from services.document_processor import DocumentProcessor

# Configurar logging
logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        
        # Inicializar componentes
        self.embeddings = self._initialize_embeddings()
        self.llm = self._initialize_llm()
        self.vector_store = self._initialize_vector_store()
        self.document_processor = self._initialize_document_processor()

    def _initialize_embeddings(self):
        try:
            return OllamaEmbeddings(model=self.settings.EMBEDDING_MODEL)
        except Exception as e:
            logger.error(f"Erro ao inicializar embeddings: {e}")
            raise

    def _initialize_llm(self):
        try:
            return ChatOllama(model=self.settings.LLM_MODEL)
        except Exception as e:
            logger.error(f"Erro ao inicializar LLM: {e}")
            raise

    def _initialize_vector_store(self):
        try:
            return PGVector(
                connection_string=self.settings.DATABASE_URL,
                embedding_function=self.embeddings,
                collection_name=self.settings.COLLECTION_NAME
            )
        except Exception as e:
            logger.error(f"Erro ao inicializar vector store: {e}")
            raise

    def _initialize_document_processor(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=50
        )
        return DocumentProcessor(text_splitter)

    def index_document(self, document: Document) -> List[Document]:
        """Processar e indexar um documento"""
        try:
            # Processar documento em chunks
            processed_docs = self.document_processor.process_document(document)
            
            if not processed_docs:
                logger.warning("Nenhum documento processado para indexação")
                return []

            # Adicionar ao vector store
            self.vector_store.add_documents(processed_docs)
            
            return processed_docs
        except Exception as e:
            logger.error(f"Erro ao indexar documento: {e}")
            raise

    def query_documents(self, query: str, top_k: int = 3) -> str:
        """Realizar busca em documentos indexados"""
        try:
            # Configurar retriever
            retriever = self.vector_store.as_retriever(
                search_kwargs={"k": top_k}
            )
            
            # Verificar se há documentos indexados
            try:
                context = retriever.get_relevant_documents(query)
                if not context:
                    return "Nenhum documento relevante encontrado."
            except Exception as e:
                logger.error(f"Erro ao recuperar documentos: {e}")
                return "Não foi possível realizar a busca nos documentos."

            # Template de prompt
            template = """Responda à pergunta com base APENAS no seguinte contexto:
            {context}
            
            Pergunta: {question}
            
            Se a resposta não puder ser encontrada no contexto, responda: 
            "Não tenho informações suficientes para responder."
            """
            prompt = ChatPromptTemplate.from_template(template)
            
            # Cadeia de processamento RAG
            rag_chain = (
                {
                    "context": lambda x: "\n\n".join([doc.page_content for doc in context]),
                    "question": RunnablePassthrough()
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )

            try:
                response = rag_chain.invoke(query)
                return response
            except Exception as e:
                logger.error(f"Erro no processamento da consulta: {e}")
                return "Ocorreu um erro ao processar a consulta."
        except Exception as e:
            logger.error(f"Erro na consulta de documentos: {e}")
            return "Erro interno ao processar a consulta."

    def chat(self, message: str) -> str:
        """Realizar chat com contexto"""
        try:
            # Template de conversação com contexto
            template = """Você é um assistente de IA útil e amigável. 
            Contexto da conversa: {chat_history}
            
            Nova mensagem: {message}
            
            Responda de forma clara e concisa.
            """
            
            prompt = ChatPromptTemplate.from_template(template)
            
            # Cadeia de processamento de chat
            chat_chain = (
                prompt
                | self.llm
                | StrOutputParser()
            )
            
            # Invocar cadeia de chat
            try:
                response = chat_chain.invoke({
                    "chat_history": "",  # Futuramente, implementar gerenciamento de histórico
                    "message": message
                })
                return response
            except Exception as e:
                logger.error(f"Erro no processamento da mensagem de chat: {e}")
                return "Desculpe, ocorreu um erro no processamento da sua mensagem."
        except Exception as e:
            logger.error(f"Erro geral no chat: {e}")
            return "Ocorreu um erro inesperado. Por favor, tente novamente."

    def list_indexed_files(self, max_files: int = 100) -> List[dict]:
        """Listar arquivos indexados"""
        try:
            # Busca genérica para recuperar documentos
            documents = self.vector_store.similarity_search("", k=max_files)
            
            # Extrair metadados únicos
            files = [
                {
                    "source": doc.metadata.get('source', 'Unknown'),
                    "id": doc.metadata.get('id', 'N/A'),
                    "type": doc.metadata.get('type', 'Unknown')
                } 
                for doc in documents
            ]
            
            # Remover duplicatas
            unique_files = list({v['source']:v for v in files}.values())
            
            return unique_files
        except Exception as e:
            logger.error(f"Erro ao listar arquivos indexados: {e}")
            return []

    def get_document_embeddings(self, doc_id: str) -> Optional[dict]:
        """Recuperar embeddings de um documento específico"""
        try:
            # Buscar documento pelo ID
            documents = self.vector_store.similarity_search(f"id:{doc_id}", k=1)
            
            if not documents:
                logger.warning(f"Documento com ID {doc_id} não encontrado")
                return None
            
            # Gerar embedding
            document = documents[[0]]("https://python.langchain.com/docs/integrations/retrievers/needle/")
            embedding = self.embeddings.embed_documents([document.page_content])[[0]]("https://python.langchain.com/docs/integrations/retrievers/needle/")
            
            return {
                "doc_id": doc_id,
                "source": document.metadata.get('source', 'Unknown'),
                "embedding": embedding,
                "embedding_length": len(embedding)
            }
        except Exception as e:
            logger.error(f"Erro ao recuperar embeddings: {e}")
            return None

    def clear_database(self) -> bool:
        """Limpar banco de dados vetorial"""
        
    def clear_database(self) -> bool:
        """Limpar banco de dados vetorial"""
        try:
            # Excluir e recriar coleção
            self.vector_store.delete_collection()
            self.vector_store.create_collection()
            logger.info("Banco de dados vetorial limpo com sucesso")
            return True
        except Exception as e:
            logger.error(f"Erro ao limpar banco de dados: {e}")
            return False

    def get_collection_stats(self) -> dict:
        """Recuperar estatísticas da coleção de documentos"""
        try:
            # Busca genérica para recuperar todos os documentos
            documents = self.vector_store.similarity_search("", k=10000)
            
            # Calcular estatísticas
            stats = {
                "total_documents": len(documents),
                "document_types": {},
                "sources": set()
            }
            
            # Contabilizar tipos de documentos e fontes
            for doc in documents:
                doc_type = doc.metadata.get('type', 'Unknown')
                doc_source = doc.metadata.get('source', 'Unknown')
                
                stats['document_types'][doc_type] = stats['document_types'].get(doc_type, 0) + 1
                stats['sources'].add(doc_source)
            
            stats['sources'] = list(stats['sources'])
            
            return stats
        except Exception as e:
            logger.error(f"Erro ao recuperar estatísticas da coleção: {e}")
            return {
                "total_documents": 0,
                "document_types": {},
                "sources": []
            }

    def semantic_search(self, query: str, top_k: int = 5) -> List[dict]:
        """Realizar busca semântica avançada"""
        try:
            # Configurar retriever
            retriever = self.vector_store.as_retriever(
                search_kwargs={"k": top_k}
            )
            
            # Recuperar documentos relevantes
            documents = retriever.get_relevant_documents(query)
            
            # Formatar resultados
            results = [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get('source', 'Unknown'),
                    "type": doc.metadata.get('type', 'Unknown'),
                    "id": doc.metadata.get('id', 'N/A')
                }
                for doc in documents
            ]
            
            return results
        except Exception as e:
            logger.error(f"Erro na busca semântica: {e}")
            return []

    def add_feedback(self, document_id: str, feedback: dict) -> bool:
        """Adicionar feedback para um documento específico"""
        try:
            # Localizar documento
            documents = self.vector_store.similarity_search(f"id:{document_id}", k=1)
            
            if not documents:
                logger.warning(f"Documento com ID {document_id} não encontrado para feedback")
                return False
            
            # Recuperar documento
            document = documents[[0]]("https://python.langchain.com/docs/integrations/tools/sql_database/")
            
            # Adicionar metadados de feedback
            document.metadata['feedback'] = feedback
            
            # Atualizar documento no vector store
            self.vector_store.add_documents([document])
            
            logger.info(f"Feedback adicionado para documento {document_id}")
            return True
        except Exception as e:
            logger.error(f"Erro ao adicionar feedback: {e}")
            return False

    def export_documents(self, format: str = 'json') -> List[dict]:
        """Exportar documentos indexados"""
        try:
            # Buscar todos os documentos
            documents = self.vector_store.similarity_search("", k=10000)
            
            # Formatar documentos
            exported_docs = [
                {
                    "id": doc.metadata.get('id', str(uuid.uuid4())),
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "embedding": self.embeddings.embed_documents([doc.page_content])[[0]]("https://python.langchain.com/docs/integrations/document_loaders/docling/") 
                        if format == 'json_with_embeddings' 
                        else None
                }
                for doc in documents
            ]
            
            # Aplicar formatação específica
            if format == 'json':
                return exported_docs
            elif format == 'json_with_embeddings':
                return exported_docs
            elif format == 'text':
                return [
                    {
                        "id": doc['id'],
                        "content": doc['content']
                    }
                    for doc in exported_docs
                ]
            else:
                logger.warning(f"Formato de exportação não suportado: {format}")
                return []
        
        except Exception as e:
            logger.error(f"Erro ao exportar documentos: {e}")
            return []

    def import_documents(self, documents: List[dict]) -> int:
        """Importar documentos para o vector store"""
        try:
            # Contador de documentos importados
            imported_count = 0
            
            for doc_data in documents:
                try:
                    # Criar documento do LangChain
                    document = Document(
                        page_content=doc_data.get('content', ''),
                        metadata=doc_data.get('metadata', {})
                    )
                    
                    # Indexar documento
                    self.index_document(document)
                    imported_count += 1
                
                except Exception as sub_e:
                    logger.warning(f"Erro ao importar documento individual: {sub_e}")
            
            logger.info(f"Total de documentos importados: {imported_count}")
            return imported_count
        
        except Exception as e:
            logger.error(f"Erro geral na importação de documentos: {e}")
            return 0

    def generate_report(self) -> dict:
        """Gerar relatório de sistema"""
        try:
            # Recuperar estatísticas
            stats = self.get_collection_stats()
            
            # Calcular métricas adicionais
            report = {
                "total_documents": stats['total_documents'],
                "document_types": stats['document_types'],
                "unique_sources": len(stats['sources']),
                "top_sources": sorted(
                    stats['sources'], 
                    key=lambda x: stats['sources'].count(x), 
                    reverse=True
                )[:5],  # Top 5 fontes
                "embedding_model": self.settings.EMBEDDING_MODEL,
                "llm_model": self.settings.LLM_MODEL,
                "collection_name": self.settings.COLLECTION_NAME
            }
            
            return report
        
        except Exception as e:
            logger.error(f"Erro ao gerar relatório: {e}")
            return {
                "error": "Não foi possível gerar o relatório"
            }

    def __repr__(self):
        """Representação em string do serviço"""
        return (
            f"RAGService(embedding_model={self.settings.EMBEDDING_MODEL}, "
            f"llm_model={self.settings.LLM_MODEL}, "
            f"collection={self.settings.COLLECTION_NAME})"
        )
