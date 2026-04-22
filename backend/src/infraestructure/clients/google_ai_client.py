from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from typing import Protocol  # Para abstrações, se quiser interfaces

# Opcional: Defina protocolos (interfaces) para abstrações, seguindo SOLID I/D
class EmbeddingsProtocol(Protocol):
    def embed_query(self, text: str) -> list[float]: ...

class LLMProtocol(Protocol):
    async def ainvoke(self, messages) -> str: ...

class GoogleAIClient:
    """Cliente para configurar e fornecer instâncias de Google AI (Embeddings e LLM)."""
    
    def __init__(self):
        # Configurações centralizadas aqui
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.1,
            convert_system_message_to_human=True
        )
    
    def get_embeddings(self) -> EmbeddingsProtocol:
        return self.embeddings
    
    def get_llm(self) -> LLMProtocol:
        return self.llm