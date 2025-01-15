import logging
import faiss
import numpy as np
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import pickle
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ChunkMetadata:
    """Clase para almacenar la metadata de cada chunk de texto."""
    page_number: int
    chunk_number: int
    text: str
    start_position: int
    end_position: int
    word_count: int

class RAGEmbeddingEngine:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        max_tokens: int = 500,
        chunk_overlap: int = 50,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None
    ):
        """
        Inicializa el motor de embeddings para RAG.
        
        Args:
            model_name: Nombre del modelo de Sentence Transformers
            max_tokens: Máximo número de tokens por chunk
            chunk_overlap: Número de tokens de superposición entre chunks
            index_path: Ruta opcional para cargar un índice FAISS existente
            metadata_path: Ruta opcional para cargar metadata existente
        """
        logger.info(f"Inicializando RAGEmbeddingEngine con modelo {model_name}")
        self.model = SentenceTransformer(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens
        self.chunk_overlap = chunk_overlap
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        if index_path and Path(index_path).exists():
            logger.info(f"Cargando índice FAISS desde {index_path}")
            self.index = faiss.read_index(index_path)
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            logger.info("Creando nuevo índice FAISS")
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = {}

    def split_text_into_chunks(self, text: str) -> List[Tuple[str, int, int]]:
        tokens = self.tokenizer.encode(text)
        chunks = []
        start_idx = 0

        while start_idx < len(tokens):
            end_idx = min(start_idx + self.max_tokens, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)

            if end_idx < len(tokens):
                # Buscar el último punto y retornar al final de la oración
                last_period = chunk_text.rfind('.')
                last_exclamation = chunk_text.rfind('!')
                last_question = chunk_text.rfind('?')

                # Encuentra la posición más grande entre los posibles finales de oración
                last_sentence_end = max(last_period, last_exclamation, last_question)

                if last_sentence_end != -1:
                    # Cortar el texto hasta el final de la última oración
                    chunk_text = chunk_text[:last_sentence_end + 1]
                    end_idx = start_idx + len(self.tokenizer.encode(chunk_text, add_special_tokens=False))

            # Calcular posiciones de inicio y fin en caracteres
            if end_idx > len(tokens):
                end_idx = len(tokens)

            start_pos = len(self.tokenizer.decode(tokens[:start_idx], skip_special_tokens=True))
            end_pos = start_pos + len(chunk_text)

            chunks.append((chunk_text, start_pos, end_pos))

            # Ajustar el inicio del siguiente chunk
            next_start_idx = min(end_idx, start_idx + self.max_tokens - self.chunk_overlap)
            start_idx = next_start_idx

        return chunks


    def append_page(self, page_text: str, page_number: int) -> None:
        """
        Añade una página al índice FAISS, dividiéndola en chunks si es necesario.
        
        Args:
            page_text: Texto de la página
            page_number: Número de página
        """
        try:
            # Dividir la página en chunks
            chunks = self.split_text_into_chunks(page_text)
            
            for chunk_number, (chunk_text, start_pos, end_pos) in enumerate(chunks):
                # Crear embedding para el chunk
                embedding = self.model.encode([chunk_text])[0]
                
                # Añadir al índice FAISS
                self.index.add(np.array([embedding]).astype('float32'))
                
                # Guardar metadata
                word_count = len(chunk_text.split())
                self.metadata[self.index.ntotal - 1] = ChunkMetadata(
                    page_number=page_number,
                    chunk_number=chunk_number,
                    text=chunk_text,
                    start_position=start_pos,
                    end_position=end_pos,
                    word_count=word_count
                )
            
            logger.info(f"Página {page_number} procesada en {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error al procesar página {page_number}: {str(e)}")
            raise

    def search(self, query: str, k: int = 3) -> List[Tuple[float, ChunkMetadata]]:
        """
        Busca los chunks más relevantes para una consulta.
        
        Args:
            query: Texto de la consulta
            k: Número de resultados a retornar
            
        Returns:
            Lista de tuplas (score, metadata) ordenadas por relevancia
        """
        try:
            query_embedding = self.model.encode([query])[0]
            scores, indices = self.index.search(
                np.array([query_embedding]).astype('float32'), 
                k
            )
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1:
                    results.append((float(score), self.metadata[idx]))
            
            logger.info(f"Búsqueda completada para '{query}'. {len(results)} resultados encontrados")
            return results
            
        except Exception as e:
            logger.error(f"Error en búsqueda: {str(e)}")
            raise

    def save(self, index_path: str, metadata_path: str) -> None:
        """
        Guarda el índice FAISS y la metadata en disco.
        """
        try:
            faiss.write_index(self.index, index_path)
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            logger.info(f"Índice y metadata guardados en {index_path} y {metadata_path}")
        except Exception as e:
            logger.error(f"Error al guardar índice y metadata: {str(e)}")
            raise

    def get_stats(self) -> Dict:
        """
        Retorna estadísticas del índice y chunks.
        """
        return {
            "total_chunks": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": type(self.index).__name__,
            "max_tokens_per_chunk": self.max_tokens,
            "chunk_overlap": self.chunk_overlap
        }