# Proyecto de Chatbot Experto en Juego de Tronos

Este proyecto implementa un chatbot experto en el primer libro de la serie "Juego de Tronos". Utiliza un motor de búsqueda RAG (Retrieval-Augmented Generation) para responder preguntas basadas en el contenido del libro.

## Estructura del Proyecto

### Archivos y Directorios Principales

- `deprecated_code/`: Código antiguo que ya no se utiliza.
- `llamaindex_got/`: Almacena los datos del índice utilizado por el motor RAG.
- `src/`: Contiene el código fuente principal del proyecto.
  - `expert_agent.py`: Implementa el agente experto que interactúa con el usuario.
  - `rag_llamaindex.py`: Implementa el motor RAG para la búsqueda de información.
- `test_chatbot.ipynb`: Notebook de Jupyter para probar el chatbot.
- `test_rag.ipynb`: Notebook de Jupyter para probar el motor RAG.

## Instalación

1. Clona el repositorio:
    ```sh
    git clone <URL_DEL_REPOSITORIO>
    cd <NOMBRE_DEL_REPOSITORIO>
    ```

2. Instala las dependencias:
    ```sh
    pip install -r requirements.txt
    ```

## Uso

### Inicialización del Chatbot

Para inicializar el chatbot, primero debes importar e instanciar las clases `ExpertAgent` y `RAGEmbeddingEngine`:

```python
from src.expert_agent import ExpertAgent
from src.rag_llamaindex import RAGEmbeddingEngine

rag_tool = RAGEmbeddingEngine()
expert = ExpertAgent(rag_tool=rag_tool)
```

### Realizar Preguntas

Puedes realizar preguntas al chatbot utilizando el método `chat` del `ExpertAgent`:

```python
response = expert.chat("¿Cómo se llama el lobo de Sansa?")
print(response)
```

### Ejecución en Notebooks

Puedes probar el chatbot y el motor RAG utilizando los notebooks `test_chatbot.ipynb` y `test_rag.ipynb`.