import os
import asyncio
from typing import List, Dict, Any

# Importações do LangChain
from langchain_core.documents import Document
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Loaders para diferentes tipos de arquivos
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredODTLoader,
    UnstructuredDocxLoader,
    UnstructuredCSVLoader,
    TextLoader
)

# Embeddings e Vector Store
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Modelo de Chat
from langchain_openai import ChatOpenAI

# Ferramentas de processamento de documentos
from langchain_text_splitters import RecursiveCharacterTextSplitter

class MultiFormatRAG:
    def __init__(self, 
                 documents_path: str = "./documentos",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        # Configurações de ambiente
        os.environ["OPENAI_API_KEY"] = "seu_openai_api_key"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = "seu_langsmith_api_key"

        # Configurações
        self.documents_path = documents_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Inicializa componentes
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        self.vectorstore = None

    def load_documents(self) -> List[Document]:
        """Carrega documentos de múltiplos formatos"""
        all_documents = []
        loaders_map = {
            '.pdf': PyPDFLoader,
            '.odt': UnstructuredODTLoader,
            '.docx': UnstructuredDocxLoader,
            '.csv': UnstructuredCSVLoader,
            '.txt': TextLoader
        }

        for filename in os.listdir(self.documents_path):
            filepath = os.path.join(self.documents_path, filename)
            file_ext = os.path.splitext(filename)[[1]](https://docs.smith.langchain.com/evaluation/how_to_guides/evaluate_llm_application).lower()

            try:
                if file_ext in loaders_map:
                    loader = loaders_map[file_ext](filepath)
                    documents = loader.load()
                    
                    # Divide documentos em chunks
                    split_docs = self.text_splitter.split_documents(documents)
                    all_documents.extend(split_docs)
                    
                    print(f"✅ Carregado: {filename}")
                else:
                    print(f"❌ Formato não suportado: {filename}")
            
            except Exception as e:
                print(f"Erro ao carregar {filename}: {e}")

        return all_documents

    def create_vectorstore(self, documents: List[Document]):
        """Cria vector store a partir dos documentos"""
        self.vectorstore = Chroma.from_documents(
            documents, 
            self.embeddings,
            persist_directory="./chroma_db"
        )
        return self.vectorstore

    def create_rag_chain(self):
        """Cria cadeia de RAG com prompt personalizado"""
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )

        # Prompt template personalizado
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            Você é um assistente de IA especializado em análise de documentos. 
            Use APENAS as informações fornecidas no contexto para responder.
            Se não souber a resposta, admita honestamente.
            
            Contexto: {context}
            """),
            ("human", "Pergunta: {question}")
        ])

        # Cadeia RAG completa usando LCEL
        rag_chain = (
            RunnableParallel({
                "context": retriever,
                "question": RunnablePassthrough()
            })
            | rag_prompt
            | self.llm
            | StrOutputParser()
        )

        return rag_chain

    async def process_documents(self):
        """Processa documentos de forma assíncrona"""
        # Carrega documentos
        documents = self.load_documents()
        
        if not documents:
            print("❌ Nenhum documento encontrado!")
            return None

        # Cria vector store
        vectorstore = self.create_vectorstore(documents)
        
        # Cria cadeia RAG
        rag_chain = self.create_rag_chain()

        return rag_chain

    async def query_documents(self, question: str):
        """Consulta documentos com tratamento de erros"""
        try:
            # Verifica se vector store foi criado
            if not self.vectorstore:
                await self.process_documents()

            # Executa cadeia RAG
            response = await self.process_documents().astream(question)
            
            return response
        
        except Exception as e:
            print(f"❌ Erro na consulta: {e}")
            return None

# Classe de Análise Avançada
class DocumentAnalyzer:
    @staticmethod
    async def analyze_document_collection(rag_system: MultiFormatRAG):
        """Análise abrangente da coleção de documentos"""
        # Perguntas de análise
        analysis_questions = [
            "Qual é o tema principal dos documentos?",
            "Existem padrões ou insights recorrentes?",
            "Quais são as principais informações encontradas?",
            "Há alguma contradição entre os documentos?"
        ]

        # Resultados da análise
        analysis_results = {}

        for question in analysis_questions:
            print(f"\n🔍 Analisando: {question}")
            
            try:
                response = await rag_system.query_documents(question)
                analysis_results[question] = response
                
                print(f"📝 Resposta: {response}")
            
            except Exception as e:
                print(f"❌ Erro na análise: {e}")

        return analysis_results

# Função principal de execução
async def main():
    # Configurações de logging
    import logging
    logging.basicConfig(level=logging.INFO)

    # Inicializa sistema RAG
    rag_system = MultiFormatRAG(
        documents_path="./documentos",
        chunk_size=1000,
        chunk_overlap=200
    )

    # Processa documentos
    await rag_system.process_documents()

    # Inicializa analisador
    document_analyzer = DocumentAnalyzer()

    # Análise abrangente
    await document_analyzer.analyze_document_collection(rag_system)

    # Exemplo de consulta interativa
    while True:
        query = input("\n🤔 Faça uma pergunta sobre os documentos (ou 'sair' para encerrar): ")
        
        if query.lower() == 'sair':
            break
        
        response = await rag_system.query_documents(query)
        print("\n🤖 Resposta:", response)

# Execução do script
if __name__ == "__main__":
    asyncio.run(main())
