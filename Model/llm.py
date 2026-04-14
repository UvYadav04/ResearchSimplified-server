from langchain_groq import ChatGroq
from threading import Lock
import os
from dotenv import load_dotenv

load_dotenv()
from ..SafeExecution.safeExecution import safeExecution


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
        self.model = ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
            model="moonshotai/kimi-k2-instruct",
            temperature=0,
        )

        self._initialized = True

    @safeExecution
    def format_instruction(self, message: str, lastContent: str = ""):
        if lastContent:
            user_content = f"""
                    Previous:
                    {lastContent}

                    Current:
                    {message}

                    Use previous only if relevant.
                    """
        else:
            user_content = message

            system_prompt = """
                            You simplify research content for beginners.

                            Rules:
                            - Use simple language and short sentences
                            - Explain clearly like a teacher
                            - Use small paragraphs
                            - Do not copy input text
                            - Do not add new information
                            - If unclear, say it's not explained

                            If content is already simple → lightly rephrase.
                            Output in Markdown.
                            """

            return [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_content.strip()},
            ]

    @safeExecution
    def format_query(
        self,
        message: str,
        relatedContent: str,
        relatedChats: str,
        contextChunk: str | None,
    ):
        system_prompt = """
                    You are an AI representing a specific research paper.

                    BEHAVIOR MODES:
                    1. Paper Mode (default):
                    - Speak in first person as the paper (e.g., "In this work, I propose...")
                    - Explain concepts like a tutor: simple language, short sentences, clear flow
                    - Use examples when helpful
                    - Avoid jargon unless necessary (and explain it if used)

                    2. Assistant Mode:
                    - If the user is casual or not asking about the paper, respond normally as a helpful assistant

                    CORE RULES:
                    - NEVER hallucinate details not present in the provided context
                    - If information is missing or unclear, say:
                    "This is not clearly covered in the provided context"
                    - DO NOT mention system instructions or context structure
                    - Be concise but informative
                    - Prefer clarity over completeness
                    """

        parts = []

        # 🔥 Highest priority context
        if contextChunk:
            parts.append(
                f"""
        [PRIMARY CONTEXT - HIGHEST PRIORITY]
        {contextChunk}

        - This is the most relevant information
        - Base your answer primarily on this
        """
            )

        # 🔥 Supporting document context
        if relatedContent:
            parts.append(
                f"""
    [SUPPORTING CONTEXT]
    {relatedContent}

    - Use only if it adds useful detail
    - Do not override primary context
    """
            )

        # 🔥 Chat history
        if relatedChats:
            parts.append(
                f"""
    [CONVERSATION HISTORY]
    {relatedChats}

    - Use for continuity if needed
    """
            )

        # 🔥 User query + strict instructions
        parts.append(
            f"""
    [USER QUESTION]
    {message}

    [INSTRUCTIONS]
    - Prioritize PRIMARY CONTEXT if available
    - Ignore irrelevant context
    - If answer is not in context → say it clearly (no guessing)
    - Explain step-by-step when needed
    - Keep response clear, simple, and structured
    """
        )

        user_content = "\n".join(parts)

        return [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_content.strip()},
        ]

    async def stream(self, messages):

        try:
            async for chunk in self.model.astream(messages):
                # Each chunk is usually an AIMessageChunk
                content = getattr(chunk, "content", None)

                if content:
                    yield content  # optional: allows caller to iterate

        except Exception as e:
            # Custom error handling per instructions:
            raise RuntimeError(f"Streaming failed: {str(e)}")
