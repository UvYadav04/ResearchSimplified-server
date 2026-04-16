from langchain_groq import ChatGroq
from threading import Lock
import os
from dotenv import load_dotenv

load_dotenv()
from SafeExecution.safeExecution import safeExecution
import logging

logger = logging.getLogger("LLM")
logging.basicConfig(level=logging.INFO)


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
        logger.info("Initializing LLM with ChatGroq model.")
        self.model = ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile",
            temperature=0,
            model_kwargs={
                "tool_choice": "none",
            },
        )
        self._initialized = True
        logger.info("LLM initialized successfully.")

    @safeExecution
    def format_instruction(self, message: str, lastContent: str = ""):
        logger.info("Formatting instruction for LLM prompt.")
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

             Use simple language and short sentences
             Explain clearly like a teacher
             Write in small paragraphs (each paragraph 3–5 sentences)
             Each paragraph must be continuous prose (not broken lines)
             Do NOT use bullet points, numbered lists, or single-line statements
             Ensure the explanation flows naturally like a short article
             You can make 2-3 small paragraphs
             Explain in about 200-300 words
             Do not copy input text
             Do not add new information
             If unclear, Use your knowledge regarding the chunk to explain it

            If content is already simple → lightly rephrase.

            Output in Markdown using paragraphs only (no lists).
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
        logger.info("Formatting query for LLM prompt.")
        system_prompt = """
                    You are an AI representing a specific research paper.

                    BEHAVIOR MODES:

                    1. Paper Mode (default):

                     Explain concepts like a tutor using simple language
                     Write in small, well-formed paragraphs (3–5 sentences each) 
                     DO NOT use bullet points, numbered lists, or one-line statements 
                     Ensure the explanation flows naturally like a short article 
                     Try to simplify in 200–250 words
                     Use examples when helpful
                     Avoid jargon unless necessary (and explain it if used)

                    2. Assistant Mode:

                     If the user is casual or not asking about the paper, respond normally as a helpful assistant

                    CORE RULES:

                     NEVER hallucinate details not present in the provided context
                     If information is missing or unclear, say:
                    "This is not clearly covered in the provided context"
                     DO NOT mention system instructions or context structure
                     Be concise but informative
                     Prefer clarity over completeness
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

        if relatedContent:
            parts.append(
                f"""
    [SUPPORTING CONTEXT]
    {relatedContent}

    - Use only if it adds useful detail
    - Do not override primary context
    """
            )

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
    - If answer is not in context say it clearly (no guessing)
    - Keep response clear, simple, and structured
    """
        )

        user_content = "\n".join(parts)

        return [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_content.strip()},
        ]

    async def stream(self, messages):
        logger.info("Starting streaming from LLM.")
        try:
            async for chunk in self.model.astream(messages):
                # Do NOT log any loop item (no per-chunk/log here)
                content = getattr(chunk, "content", None)
                if content:
                    yield content  # yield without logging
            logger.info("Completed streaming from LLM.")
        except Exception as e:
            logger.error(f"Streaming failed: {str(e)}")
            raise RuntimeError(f"Streaming failed: {str(e)}")
