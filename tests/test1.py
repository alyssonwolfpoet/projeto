from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

# 1. Carregamento e Preparação dos Documentos
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/"
]

# Carregar documentos
loader = WebBaseLoader(urls)
docs = loader.load()

# Dividir documentos em chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# 2. Configuração de Embeddings e Vetorstore
embeddings = OllamaEmbeddings(model="llama3")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

# 3. Configuração do Modelo LLM
llm = ChatOllama(model="llama3", temperature=0.7)

# 4. Prompt Template para RAG
prompt_template = ChatPromptTemplate.from_template("""
Você é um assistente de IA útil. Com base no contexto abaixo, responda à pergunta:

Contexto: {context}

Pergunta: {question}

Resposta detalhada:
""")

# 5. Construção do RAG Chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt_template 
    | llm 
    | StrOutputParser()
)

# 6. Função para realizar consultas
def consultar_rag(pergunta):
    return rag_chain.invoke(pergunta)

# Exemplo de uso
resultado = consultar_rag("O que são agentes de IA?")
print(resultado)


import os
from typing import List
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PDFLoader, TextLoader
from langchain_core.documents import Document

class RAGSystem:
    def _init_(
        self, 
        model_name: str = "llama3", 
        embedding_model: str = "nomic-embed-text",
        chunk_size: int = 500, 
        chunk_overlap: int = 50
    ):
        # Configurações de modelos
        self.llm = ChatOllama(model=model_name, temperature=0.7)
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        
        # Configurações de splitting
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        
        # Vetorstore
        self.vectorstore = None
        
    def load_documents(self, sources: List[str]) -> List[Document]:
        """
        Carrega documentos de múltiplas fontes
        Suporta Web, PDF, Texto plano
        """
        all_docs = []
        for source in sources:
            if source.startswith('http'):
                loader = WebBaseLoader(source)
            elif source.endswith('.pdf'):
                loader = PDFLoader(source)
            elif source.endswith('.txt'):
                loader = TextLoader(source)
            else:
                raise ValueError(f"Tipo de fonte não suportado: {source}")
            
            docs = loader.load()
            all_docs.extend(docs)
        
        return all_docs
    
    def prepare_vectorstore(self, documents: List[Document]):
        """
        Prepara o vetorstore com documentos processados
        """
        # Divide os documentos em chunks
        splits = self.text_splitter.split_documents(documents)
        
        # Cria vetorstore
        self.vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=self.embeddings
        )
        
        return self.vectorstore.as_retriever()
    
    def create_rag_chain(self, system_prompt: str = None):
        """
        Cria o pipeline RAG com opções de personalização
        """
        # Prompt padrão se não fornecido
        if not system_prompt:
            system_prompt = """
            Você é um assistente de IA especializado em análise de documentos.
            Use APENAS o contexto fornecido para responder.
            Se não souber a resposta, admita honestamente.
            
            Contexto: {context}
            Pergunta: {question}
            
            Resposta detalhada e precisa:
            """
        
        # Template de prompt
        prompt = ChatPromptTemplate.from_template(system_prompt)
        
        # Cadeia RAG
        rag_chain = (
            RunnableParallel({
                "context": self.vectorstore.as_retriever(search_kwargs={"k": 3}),
                "question": RunnablePassthrough()
            })
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def advanced_query(
        self, 
        query: str, 
        return_sources: bool = False,
        max_sources: int = 3
    ):
        
    def advanced_query(
        self, 
        query: str, 
        return_sources: bool = False,
        max_sources: int = 3
    ):
        """
        Consulta avançada com opção de retornar fontes
        """
        # Recupera documentos relevantes
        retrieved_docs = self.vectorstore.as_retriever(
            search_kwargs={"k": max_sources}
        ).invoke(query)
        
        # Cria chain de RAG
        rag_chain = self.create_rag_chain()
        
        # Executa a consulta
        response = rag_chain.invoke(query)
        
        # Retorna com ou sem fontes
        if return_sources:
            return {
                "answer": response,
                "sources": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    } for doc in retrieved_docs
                ]
            }
        
        return response
    
    def evaluate_query(self, query: str):
        """
        Avaliação da qualidade da resposta
        """
        # Métricas de avaliação
        evaluation_prompt = ChatPromptTemplate.from_template("""
        Avalie a resposta para a pergunta com base nos seguintes critérios:
        
        Pergunta Original: {question}
        Resposta Gerada: {answer}
        
        Avaliação:
        1. Relevância (0-10): 
        2. Precisão (0-10):
        3. Completude (0-10):
        4. Clareza (0-10):
        
        Comentários detalhados:
        """)
        
        # Chain de avaliação
        evaluation_chain = (
            RunnableParallel({
                "question": RunnablePassthrough(),
                "answer": lambda x: self.advanced_query(x)
            })
            | evaluation_prompt
            | self.llm
            | StrOutputParser()
        )
        
        return evaluation_chain.invoke(query)
    
    def add_custom_documents(self, documents: List[Document]):
        """
        Adiciona documentos personalizados ao vetorstore existente
        """
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=documents, 
                embedding=self.embeddings
            )
        else:
            # Adiciona novos documentos ao vetorstore existente
            self.vectorstore.add_documents(documents)
        
        return self.vectorstore.as_retriever()
    
    def multi_query_retrieval(self, query: str, num_queries: int = 3):
        """
        Técnica de multi-query para recuperação robusta
        """
        multi_query_prompt = ChatPromptTemplate.from_template("""
        Gere {num_queries} variações diferentes da pergunta original 
        que possam ajudar a recuperar informações mais abrangentes:
        
        Pergunta Original: {query}
        
        Variações:
        """)
        
        # Gerador de variações de consulta
        query_generator = (
            multi_query_prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Gera variações de consulta
        query_variations = query_generator.invoke({
            "query": query,
            "num_queries": num_queries
        }).split('\n')
        
        # Recupera documentos para cada variação
        all_retrieved_docs = []
        for variation in query_variations:
            docs = self.vectorstore.as_retriever(
                search_kwargs={"k": 2}
            ).invoke(variation)
            all_retrieved_docs.extend(docs)
        
        # Remove documentos duplicados
        unique_docs = list({doc.page_content: doc for doc in all_retrieved_docs}.values())
        
        return unique_docs

# Exemplo de uso


