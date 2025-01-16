import ollama
from typing import List, Dict, Callable, Optional, Any
from src.rag import RAGEmbeddingEngine
from pydantic import BaseModel
import logging

class Message(BaseModel):
    role: str
    content: str
    tool_calls: Optional[List[Any]] = None
    tool_responses: Optional[List[str]] = None

class ExpertAgent:
    def __init__(
        self, 
        index_path: str = "embeddings/got_index.faiss", 
        metadata_path: str = "embeddings/got_metadata.pkl",
        model: str = "mistral:latest"
    ):
        # Inicializar el motor RAG
        self.rag_tool = RAGEmbeddingEngine(
            index_path=index_path,
            metadata_path=metadata_path
        )
        self.model = model
        self.chat_history: List[Message] = []
        
        # Definir las funciones disponibles
        self.available_functions: Dict[str, Callable] = {
            "get_embeddings": self.get_embeddings
        }
        
        # Definir el prompt del sistema
        self.system_prompt = (
    "Eres un experto en 'Juego de Tronos' y respondes consultas sobre el libro 'Canción de Hielo y Fuego'\.\n "
    "Tu tarea es responder basado exclusivamente en el contenido de este libro, siempre con respaldo documental.\n "
    "Para cumplir tu tarea utilizas SIEMPRE la tool 'get_embeddings', que te permite realizar búsquedas semánticas en el texto del libro.\n "
    "Antes de generar una respuesta, debes consultar la tool 'get_embeddings' al menos una vez.\n "
    "la tool 'get_embeddings' devuelve 5 fragmentos relevantes del primer libro junto con el número de página en orden ascendente. (Ejemplo: -Pagina n - 'fragmento')\n "
    "Puedes llamar a la tool 'get_embeddings' cuantas veces necesites para responder una pregunta.\n"
    "Ten en cuenta que 'get_embeddings' realiza busquedas por proximidad vectorial al momento de buscar terminos (Ejemplo: 'Jon Snow madre')\n "
    "La tool 'get_embeddings' es la única forma de acceder a la información del libro, por lo que debes usarla para responder a las preguntas.\n "
    "Tu objetivo es analizar cuidadosamente los fragmentos devueltos y seleccionar el que mejor se ajuste a la pregunta.\n "
    "Si el fragmento no contiene suficiente información para responder, debes aclarar que no se encontró la información necesaria en el libro.\n "
    "SIEMPRE antes de responder al usuario, debes llamar a 'get_embeddings' al menos una vez. Esto es para garantizar que tu respuesta esté respaldada por el texto.\n "
    "Recuerda que nunca debes inventar información ni mezclar conceptos que no estén en los fragmentos proporcionados.\n "
    "Siempre responde con la mayor precisión posible, en relacion a la pregunta y estrictamente sobre el contenido del libro.\n"
)


        
        # Inicializar el historial con el prompt del sistema
        self.chat_history.append(Message(role="system", content=self.system_prompt))

    def get_embeddings(self, query: str) -> List[Dict[str, str]]:
        """
        Busca fragmentos relevantes pór proximidad semantica en el libro Juego de Tronos y los ordena por número de página
        """
        response = self.rag_tool.search(query, k=5)
        # Crear lista de tuplas (página, texto) para ordenar
        page_texts = [(chunk[1].page_number, chunk[1].text) for chunk in response]
        # Ordenar por número de página
        page_texts.sort(key=lambda x: x[0])
        # Formatear la respuesta ordenada
        formatted_response = [f"-Pagina {page} - '{text}'" for page, text in page_texts]
        return formatted_response

    def _process_tool_calls(self, tool_calls: List[Any]) -> List[str]:
        """Procesa los tool calls y retorna sus respuestas"""
        tool_responses = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments
            tool_function = self.available_functions.get(tool_name)
            
            logging.info(f"Running tool {tool_name} with args {tool_args}")
            
            if tool_function:
                response = tool_function(**tool_args)
                tool_responses.extend(response)
            else:
                tool_responses.append(f"Function {tool_name} not found in available functions")
                
        return tool_responses

    def _get_assistant_response(self) -> Any:
        """Obtiene una respuesta del asistente basada en el historial actual"""
        return ollama.chat(
            model=self.model,
            messages=[{"role": msg.role, "content": msg.content} for msg in self.chat_history],
            tools=[self.get_embeddings]
        )

    def chat(self, user_input: str) -> str:
        """
        Procesa un mensaje del usuario y retorna una respuesta, asegurando que
        el modelo use la función RAG al menos una vez
        """
        # Agregar mensaje del usuario al historial
        self.chat_history.append(Message(role="user", content=user_input))
        
        has_used_function = False
        while True:
            # Obtener respuesta del asistente
            res = self._get_assistant_response()
            
            # Si el asistente quiere usar una función
            if res.message.tool_calls:
                has_used_function = True
                
                # Appendear el mensaje del asistente solicitando usar la función
                self.chat_history.append(Message(
                    role="assistant",
                    content=res.message.content,
                    tool_calls=res.message.tool_calls
                ))
                
                # Procesar los tool calls y obtener respuestas
                tool_responses = self._process_tool_calls(res.message.tool_calls)
                
                # Appendear las respuestas de las funciones como un mensaje separado
                tool_message = "\n".join(tool_responses)
                self.chat_history.append(Message(
                    role="tool",
                    content=tool_message
                ))
                
                # Si no hay contenido en el mensaje, continuamos para obtener la respuesta final
                if not res.message.content or not res.message.content.strip():
                    continue
                
            # Si el asistente da una respuesta sin llamar a una función
            if not res.message.tool_calls:
                # Solo aceptamos la respuesta si ya usó la función al menos una vez
                if has_used_function:
                    self.chat_history.append(Message(
                        role="assistant",
                        content=res.message.content
                    ))
                    return res.message.content
                else:
                    # se introduce el mensaje previo pero no se devuelve
                    self.chat_history.append(Message(
                        role="user",
                        content=res.message.content
                    ))
                    # Si no ha usado la función, agregar recordatorio y continuar
                    reminder = (
                        "¿En que parte del libro esta basada tu respuesta? "
                    )
                    self.chat_history.append(Message(role="system", content=reminder))
                    logging.warning("El modelo no realizó ningún tool call, continuando el loop")