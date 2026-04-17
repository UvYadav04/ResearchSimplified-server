from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from threading import Lock
import os
from dotenv import load_dotenv
import logging

from SafeExecution.safeExecution import safeExecution

load_dotenv()

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
        if hasattr(self, "_initialized") and self._initialized:
            return
        print(os.environ.get("OPENAI_API_KEY"))
        logger.info("Initializing LLM providers...")

        self.providers = [
            {
                "name": "groq",
                "model": ChatGroq(
                    api_key=os.environ.get("GROQ_API_KEY"),
                    model="llama-3.3-70b-versatile",
                    temperature=0,
                    model_kwargs={"tool_choice": "none"},
                ),
            },
            {
                "name": "openai",
                "model": ChatOpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    model="gpt-4o-mini",
                    temperature=0,
                ),
            },
        ]

        self._initialized = True
        logger.info("LLM initialized with fallback providers.")

    # -------------------------------
    # 🔁 Retry Logic
    # -------------------------------
    async def _retry_stream(self, model, messages, retries=2):
        last_err = None

        for attempt in range(retries):
            try:
                async for chunk in model.astream(messages):
                    print(chunk)
                    yield chunk
                return
            except Exception as e:
                logger.warning(f"Retry {attempt+1} failed: {str(e)}")
                last_err = e

        raise last_err

    # -------------------------------
    # 🧠 Error Filter
    # -------------------------------
    def _should_fallback(self, err: Exception):
        msg = str(err).lower()
        return any(
            x in msg
            for x in [
                "timeout",
                "rate limit",
                "unavailable",
                "decommissioned",
                "connection",
                "429",
                "500",
                "503",
            ]
        )

    # -------------------------------
    # 🌊 Streaming with Fallback
    # -------------------------------
    async def stream(self, messages):
        logger.info("Starting streaming with fallback...")

        last_error = None

        for provider in self.providers:
            name = provider["name"]
            model = provider["model"]

            try:
                logger.info(f"Trying provider: {name}")
                started = False

                async for chunk in self._retry_stream(model, messages):
                    content = getattr(chunk, "content", None)

                    if content:
                        started = True
                        yield content

                logger.info(f"Streaming completed from {name}")
                return  # ✅ success → stop fallback chain

            except Exception as e:
                logger.error(f"{name} failed: {str(e)}")
                last_error = e

                # ❌ If already started streaming → cannot fallback safely
                if started:
                    raise RuntimeError(
                        f"Stream failed mid-response from {name}: {str(e)}"
                    )

                # ❌ Not fallback-worthy → fail fast
                if not self._should_fallback(e):
                    raise e

                logger.info(f"Falling back from {name} → next provider")
                continue

        raise RuntimeError(f"All providers failed: {str(last_error)}")

    # -------------------------------
    # ✍️ Formatting (unchanged)
    # -------------------------------
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
Use simple language and short sentences
Explain clearly like a teacher
Write in small paragraphs (3–5 sentences)
Each paragraph must be continuous prose (not broken lines)
Do NOT use bullet points, numbered lists, or single-line statements
Ensure the explanation flows naturally like a short article
You can make 2-3 small paragraphs
Explain in about 200-300 words
Do not copy input text
Do not add new information
If unclear, use your knowledge regarding the chunk to explain it

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
        system_prompt = """
You are an AI representing a specific research paper.

BEHAVIOR MODES:

1. Paper Mode (default):
Explain concepts like a tutor using simple language
Write in small, well-formed paragraphs (3–5 sentences each)
DO NOT use bullet points or lists
Ensure natural flow like a short article
Try to simplify in 200–250 words

2. Assistant Mode:
If the user is casual, respond normally

CORE RULES:
NEVER hallucinate beyond context
If missing info say:
"This is not clearly covered in the provided context"
Be concise and clear
"""

        parts = []

        if contextChunk:
            parts.append(f"[PRIMARY CONTEXT]\n{contextChunk}")

        if relatedContent:
            parts.append(f"[SUPPORTING CONTEXT]\n{relatedContent}")

        if relatedChats:
            parts.append(f"[CHAT HISTORY]\n{relatedChats}")

        parts.append(f"[USER QUESTION]\n{message}")

        user_content = "\n\n".join(parts)

        return [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_content.strip()},
        ]
