import ollama
from typing import List, Dict, Callable, Optional, Any
from deprecated_code.rag import RAGEmbeddingEngine
from pydantic import BaseModel
import logging
from llama_index.core.base.response.schema import Response

class Message(BaseModel):
    role: str
    content: str
    tool_calls: Optional[List[Any]] = None
    tool_responses: Optional[List[str]] = None

class NoToolCallException(Exception):
    """Excepcion custom para cuando el modelo no realiza ninguna llamada a la función"""
    pass

class ExpertAgent:
    def __init__(
        self,
        rag_tool: RAGEmbeddingEngine,
        model: str = "llama3.2:latest",
    ):
        # Inicializar el motor RAG
        self.rag_tool = rag_tool
        self.model = model
        self.chat_history: List[Message] = []
        
        # Definir las funciones disponibles
        self.available_functions: Dict[str, Callable] = {
            "consult_expert": self.consult_expert
        }
        
        # Definir el prompt del sistema
        self.system_prompt =  (
"Eres un asistente conversacional. Tu única función es ser una interfaz "
"amigable entre el usuario y la tool 'consult_expert'."
"\n\n"
"Tu rol es exclusivamente hacer esta tarea respecto a la tematica de Juego de Tronos (primer libro). No respondes preguntas sobre otros temas."
"\n\n"
"Para cada pregunta del usuario relacionada a juego de tronos debes: "
"\n1. Consultar sobre preguntas de la tematica de Juego de Tronos a la tool 'consult_expert'"
"\n2. La tool 'consult_expert' devolvera una respuesta directa a la pregunta formulada y los numeros de pagina de donde se puede encontrar la información. "
"\n4. Dar una respuesta amigable que transmita la informacion proporcionada por la tool, sin agregar información adicional. "
"\n\n"
"Reglas que debes seguir: "
"\n- NUNCA respondas sobre el libro sin consultar la tool 'consult_expert'"
"\n- SIEMPRE genera tus respuestas sobre Juego de Tronos solo con la información proporcionada por la tool 'consult_expert' e indicas las paginas de donde proviene la información "
"\n- E.G. -respuesta- (REFERENCIAS PAGINA 1, 2, 3) "
"\n- Nunca mencionas la tool al usuario. "
"\n  Da una respuesta amigable que transmita la información proporcionada por la tool, sin agregar información adicional."
"\n- Si el usuario pregunta de manera relacionada a una pregunta anterior, consulta a la tool 'consult_expert' formulando una consulta completa, "
"que contemple la nueva información relevante para la nueva pregunta."
"\n- Si la pregunta no es sobre Juego de Tronos, directamente discúlpate y explica que solo puedes hablar sobre este libro."
"\n- Si la tool no encuentra información relevante, indica que no tienes información sobre ese tema."
"\n\n"
"Si te preguntan cualquier cosa que no sea sobre Juego de Tronos, responde: 'Lo siento, solo puedo responder preguntas sobre Juego de Tronos.'"
)


     # Inicializar el historial con el prompt del sistema
        self.chat_history.append(Message(role="system", content=self.system_prompt))
        logging.info("Expert Agent initialized")

    def consult_expert(self, query: str) -> Dict:
        """
        Consulta al asistente experto con una pregunta y obtiene una respuesta.
        Args:
            query (str): Pregunta para el asistente experto.
        Returns:
            Dict: Respuesta del asistente experto.
        """
        logging.info(f"Running 'consult_expert' with query: {query}")
        response: Response = self.rag_tool.search(query)

        # Crear diccionario con la respuesta y los fragmentos
        result = {
            'response': response.response,
            'sources': []
        }

        # Procesar cada nodo fuente
        for node in response.source_nodes:
            # Obtener el ID del nodo para acceder a su metadata
            node_id = node.node.id_
            node_metadata = response.metadata.get(node_id, {})

            # Crear diccionario con la información del nodo
            node_info = {
                'page': node_metadata.get('page_label', '')
            }
            result['sources'].append(node_info)

        return result

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
                # Convertir el diccionario de respuesta a string
                formatted_response = str(response)
                tool_responses.append(formatted_response)
            else:
                tool_responses.append(f"Function {tool_name} not found in available functions")

        return tool_responses

    def _get_assistant_response(self) -> Any:
        """Obtiene una respuesta del asistente basada en el historial actual"""
        return ollama.chat(
            model=self.model,
            messages=[{"role": msg.role, "content": msg.content} for msg in self.chat_history],
            tools=[self.consult_expert]
        )

    def chat(self, user_input: str) -> str:
        self.chat_history.append(Message(role="user", content=user_input))
        
        while True:
            res = self._get_assistant_response()
            
            if res.message.tool_calls:
                logging.info(f"Assistant requested tool calls: {res.message.tool_calls}")
                
                self.chat_history.append(Message(
                    role="assistant",
                    content=res.message.content,
                    tool_calls=res.message.tool_calls
                ))
                
                tool_responses = self._process_tool_calls(res.message.tool_calls)
                tool_message = "\n".join(tool_responses)
                self.chat_history.append(Message(
                    role="tool",
                    content=tool_message
                ))
                
                continue  # Seguir iterando si hay tool calls
            
            self.chat_history.append(Message(
                role="assistant",
                content=res.message.content
            ))
            return res.message.content