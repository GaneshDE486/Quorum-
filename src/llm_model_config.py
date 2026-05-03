import os
import threading
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from agents import AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled

# Disable internal tracing
set_tracing_disabled(disabled=True)

# Load .env from project root
BASE_DIR = Path(__file__).resolve().parent.parent
dotenv_path = BASE_DIR / ".env"
load_dotenv(dotenv_path)

_DEFAULT_GROQ_BASE = "https://api.groq.com/openai/v1"


def get_model_from_config(
    model: Optional[str] = None,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> OpenAIChatCompletionsModel:
    """
    Build an ``OpenAIChatCompletionsModel`` from ``.env`` (Groq OpenAI-compatible API).

    Parameters override environment variables when provided:

    * ``model`` — model id (defaults to env ``model_name``).
    * ``api_key`` — defaults to env ``GROQ_API_KEY``.
    * ``base_url`` — defaults to env ``GROQ_BASE_URL`` or Groq's OpenAI-compatible URL.

    Pass this object as ``Agent(..., model=...)``. No ``OPENAI_API_KEY`` is used here.
    """
    load_dotenv(dotenv_path)
    key = api_key if api_key is not None else os.getenv("GROQ_API_KEY")
    model_id = model if model is not None else os.getenv("model_name")
    url = base_url if base_url is not None else (os.getenv("GROQ_BASE_URL") or _DEFAULT_GROQ_BASE)
    if not key or not model_id:
        raise ValueError(
            "Missing GROQ_API_KEY or model_name. Set them in .env or pass api_key= and model= "
            "to get_model_from_config()."
        )
    client = AsyncOpenAI(api_key=key, base_url=url)
    return OpenAIChatCompletionsModel(model=model_id, openai_client=client)


class ModelSingleton:
    _instance = None
    _lock = threading.Lock()  # Thread-safe for multi-threaded apps

    @classmethod
    def get_instance(cls, model: Optional[str] = None):
        """
        Lazily return a shared ``OpenAIChatCompletionsModel`` from config.

        Optional ``model`` only applies on the first successful build (same as
        ``get_model_from_config(model=...)`` on first call).
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    try:
                        cls._instance = get_model_from_config(model=model)
                    except Exception as e:
                        print(f"[ModelSingleton] Error initializing model: {e}")
                        cls._instance = None
        return cls._instance