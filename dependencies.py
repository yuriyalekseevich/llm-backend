from fastapi import Depends
import os
import logging
from openai import OpenAI
from groq import Groq

logger = logging.getLogger(__name__)


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("openai.api_key.missing")
        raise ValueError("OPENAI_API_KEY not set in .env")
    # Do not log the key itself; only reveal length for debugging
    logger.info("openai.client.created", extra={"api_key_len": len(api_key)})
    return OpenAI(api_key=api_key)


def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("groq.api_key.missing")
        raise ValueError("GROQ_API_KEY not set in .env")
    logger.info("groq.client.created", extra={"api_key_len": len(api_key)})
    return Groq(api_key=api_key)