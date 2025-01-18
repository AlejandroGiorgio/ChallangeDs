import logging
import faiss
import numpy as np
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import pickle
from dataclasses import dataclass
from pathlib import Path
import re
import spacy

@dataclass
class ChunkMetadata:
    page_number: int
    chunk_number: int
    text: str
    start_position: int
    end_position: int
    word_count: int

class RAGEmbeddingEngine:
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        max_tokens: int = 250,
        chunk_overlap: int = 100,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None
    ):
        self.model = SentenceTransformer(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens
        self.chunk_overlap = chunk_overlap
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.model_name = model_name
        
        # Cargar spaCy para español
        try:
            self.nlp = spacy.load('es_core_news_sm')
        except:
            print("Por favor, instala el modelo de español de spaCy con: python -m spacy download es_core_news_sm")
            raise
        
        if index_path and Path(index_path).exists():
            self.index = faiss.read_index(index_path)
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = {}

    def preprocess_text(self, text: str) -> str:
        text = text.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
        text = text.replace('ñ', 'n').replace('ü', 'u')
        text = re.sub(r'[^\w\s\.\!\?\,¿¡]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def split_text_into_chunks(self, text: str) -> List[Tuple[str, int, int]]:
        text = self.preprocess_text(text)
        doc = self.nlp(text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        start_pos = 0
        
        for sent in doc.sents:
            sent_tokens = len(self.tokenizer.encode(sent.text))
            
            if sent_tokens > self.max_tokens:
                if current_chunk:
                    chunk_text = ' '.join([s.text for s in current_chunk])
                    end_pos = start_pos + len(chunk_text)
                    chunks.append((chunk_text, start_pos, end_pos))
                    start_pos = end_pos
                
                sent_text = sent.text
                while sent_text:
                    tokens = self.tokenizer.encode(sent_text)
                    if len(tokens) <= self.max_tokens:
                        chunks.append((sent_text, start_pos, start_pos + len(sent_text)))
                        start_pos += len(sent_text)
                        break
                    
                    partial_tokens = tokens[:self.max_tokens]
                    partial_text = self.tokenizer.decode(partial_tokens, skip_special_tokens=True)
                    chunks.append((partial_text, start_pos, start_pos + len(partial_text)))
                    start_pos += len(partial_text)
                    sent_text = sent_text[len(partial_text):].lstrip()
                
                current_chunk = []
                current_tokens = 0
                continue
            
            if current_tokens + sent_tokens <= self.max_tokens:
                current_chunk.append(sent)
                current_tokens += sent_tokens
            else:
                if current_chunk:
                    chunk_text = ' '.join([s.text for s in current_chunk])
                    end_pos = start_pos + len(chunk_text)
                    chunks.append((chunk_text, start_pos, end_pos))
                    overlap_sents = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk[-1:]
                    current_chunk = overlap_sents
                    current_tokens = sum(len(self.tokenizer.encode(s.text)) for s in overlap_sents)
                    start_pos = end_pos - sum(len(s.text) + 1 for s in overlap_sents)
                
                current_chunk.append(sent)
                current_tokens += sent_tokens

        if current_chunk:
            chunk_text = ' '.join([s.text for s in current_chunk])
            end_pos = start_pos + len(chunk_text)
            chunks.append((chunk_text, start_pos, end_pos))
        
        return chunks

    def search(self, query: str, k: int = 5) -> List[Tuple[float, ChunkMetadata]]:
        query = self.preprocess_text(query)
        query_embedding = self.model.encode([query])[0]
        
        distances, indices = self.index.search(np.array([query_embedding]).astype('float32'), k)
        
        results = [(distances[0][i], self.metadata[indices[0][i]]) for i in range(len(indices[0])) if indices[0][i] != -1]
        return results

    def append_page(self, page_text: str, page_number: int) -> None:
        chunks = self.split_text_into_chunks(page_text)
        
        for chunk_number, (chunk_text, start_pos, end_pos) in enumerate(chunks):
            embedding = self.model.encode([chunk_text])[0]
            self.index.add(np.array([embedding]).astype('float32'))
            
            word_count = len(chunk_text.split())
            self.metadata[self.index.ntotal - 1] = ChunkMetadata(
                page_number=page_number,
                chunk_number=chunk_number,
                text=chunk_text,
                start_position=start_pos,
                end_position=end_pos,
                word_count=word_count
            )

    def save(self, base_path: str) -> None:
        try:
            base_path = Path(base_path)
            base_path.mkdir(parents=True, exist_ok=True)
            
            faiss.write_index(self.index, str(base_path / "faiss_index.bin"))
            
            with open(base_path / "metadata.pkl", 'wb') as f:
                pickle.dump(self.metadata, f)
            
            config = {
                'model_name': self.model_name, 
                'max_tokens': self.max_tokens,
                'chunk_overlap': self.chunk_overlap,
                'dimension': self.dimension
            }
            with open(base_path / "config.pkl", 'wb') as f:
                pickle.dump(config, f)
                
            logging.info(f"Modelo RAG guardado exitosamente en {base_path}")
            
        except Exception as e:
            logging.error(f"Error al guardar el modelo RAG: {str(e)}")
            raise

    @classmethod
    def load(cls, base_path: str) -> 'RAGEmbeddingEngine':
        try:
            base_path = Path(base_path)

            with open(base_path / "config.pkl", 'rb') as f:
                config = pickle.load(f)

            instance = cls(
                model_name=config['model_name'],
                max_tokens=config['max_tokens'],
                chunk_overlap=config['chunk_overlap']
            )

            instance.index = faiss.read_index(str(base_path / "faiss_index.bin"))

            with open(base_path / "metadata.pkl", 'rb') as f:
                instance.metadata = pickle.load(f)

            logging.info(f"Modelo RAG cargado exitosamente desde {base_path}")
            return instance

        except Exception as e:
            logging.error(f"Error al cargar el modelo RAG: {str(e)}")
            raise