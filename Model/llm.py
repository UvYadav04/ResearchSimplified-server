from langchain_groq import ChatGroq
from threading import Lock
import json

class LLM:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Prevent re-initialization (important!)
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.client = ChatGroq(
            model="llama3-8b-8192",
            temperature=0,
            streaming=True,
        )

        self._initialized = True

    def stream(self, messages):
        """
        messages: list[dict] OR LangChain messages
        returns: generator (token stream)
        """
        try:
            stream = self.client.stream(messages)

            for chunk in stream:
                if chunk.content:
                    yield chunk.content

        except Exception as e:
            raise RuntimeError(f"LLM stream failed: {str(e)}")