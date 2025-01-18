import logging
import time
from typing import List, Dict, Optional
from pathlib import Path
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

class RAGEmbeddingEngine:
    def __init__(self, index_path: str = "llamaindex_got", chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        self.index_path = index_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm = Ollama(model="llama3.2:latest")
        self.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large")
        self.load_index()

    def load_index(self):
        """Carga el índice desde el almacenamiento."""
        try:
            storage_context = StorageContext.from_defaults(persist_dir=self.index_path)
            self.index = load_index_from_storage(storage_context, embed_model=self.embed_model)
            self.query_engine = self.index.as_query_engine(llm=self.llm)
            logging.info("Índice cargado exitosamente.")
        except Exception as e:
            logging.error(f"Error al cargar el índice: {e}")
            self.query_engine = None


    def create_index(self, pdf_path: str):
        """Genera un índice a partir de un archivo PDF."""
        try:
            documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
            parser = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            index = VectorStoreIndex.from_documents(documents, embed_model=self.embed_model, text_splitter=parser)
            index.storage_context.persist(self.index_path)
            logging.info(f"Índice generado y guardado en {self.index_path}.")
        except Exception as e:
            logging.error(f"Error al generar el índice: {e}")
            raise

    def search(self, query: str, retries: int = 3, delay: float = 2.0) -> List[str]:
        """
        Busca fragmentos relevantes por proximidad semántica en el libro Juego de Tronos,
        con reintentos en caso de fallo.
        """
        logging.info(f"Running 'search' with query: {query}")
        for attempt in range(retries):
            try:
                response = self.query_engine.query(query)
                return response
            except Exception as e:
                logging.warning(f"Intento {attempt + 1} fallido: {e}")
                time.sleep(delay)
        return ["Fallo algo, prueba de nuevo"]
