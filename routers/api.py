from fastapi import APIRouter, Depends, HTTPException
from openai import OpenAI
from dependencies import get_openai_client, get_groq_client
from models.models import Query, Response
from groq import Groq
import json
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=Response)
async def chat(query: Query, client: OpenAI = Depends(get_openai_client)):
    logger.info("chat.invoked", extra={"text_len": len(query.text), "temperature": query.temperature})
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Cheap, fast; scale to gpt-4o later
            messages=[{"role": "user", "content": query.text}],
            temperature=query.temperature,
        )
        output = response.choices[0].message.content
        tokens = getattr(response.usage, "total_tokens", None)  # Track costs
        logger.info("chat.success", extra={"tokens_used": tokens, "output_len": len(output)})
        return Response(output=output, tokens_used=tokens)
    except Exception as e:
        logger.exception("chat.error", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summarize", response_model=Response)
async def summarize(query: Query, client: OpenAI = Depends(get_openai_client)):
    logger.info("summarize.invoked", extra={"text_len": len(query.text), "temperature": query.temperature})
    try:
        # Custom prompt for summarization
        prompt = f"Summarize this text concisely: {query.text}"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=query.temperature,
        )
        output = response.choices[0].message.content
        tokens = getattr(response.usage, "total_tokens", None)
        logger.info("summarize.success", extra={"tokens_used": tokens, "output_len": len(output)})
        return Response(output=output, tokens_used=tokens)
    except Exception as e:
        logger.exception("summarize.error", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat_groq", response_model=Response)
async def chat_groq(query: Query, client: Groq = Depends(get_groq_client)):
    # Pretty-print the full incoming query for easy, readable console debugging
    try:
        pretty = json.dumps(query.model_dump(), indent=2, ensure_ascii=False)
    except Exception:
        pretty = str(query)

    messages_count = len(query.messages) if query.messages else (1 if query.text else 0)
    logger.info(
        "chat_groq.invoked",
        extra={
            "messages": messages_count,
            "temperature": query.temperature,
            "payload": pretty,
        },
    )

    # Also print a human-friendly, multi-line payload block to stdout
    # (Use plain print so the console shows the payload without repeating timestamps.)
    try:
        print("\n→ → → REQUEST [POST] /api/chat_groq")
        print(pretty)
    except Exception:
        print("→ → → REQUEST [POST] /api/chat_groq\n", query)

    try:
        # Build messages payload: prefer `messages`, fall back to `text`.
        if query.messages:
            messages_payload = [m.model_dump() for m in query.messages]
        elif query.text:
            messages_payload = [{"role": "user", "content": query.text}]
        else:
            raise HTTPException(status_code=400, detail="No messages or text provided")

        chat_completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",

            # 🧠 messages from frontend
            messages=messages_payload,

            # 🔐 guarded controls
            temperature=min(max(query.temperature, 0.0), 1.0),
            top_p=min(max(query.top_p, 0.1), 1.0),
            max_tokens=min(max(query.max_tokens, 64), 2048),

            # reasoning
            reasoning_effort=(
                None if query.reasoning_effort == "none" else query.reasoning_effort
            ),

            # determinism
            seed=query.seed,

            # stop tokens
            stop=query.stop,
        )

        output = chat_completion.choices[0].message.content

        tokens = getattr(chat_completion, "usage", None)
        if tokens is not None:
            tokens = getattr(tokens, "total_tokens", None)

        logger.info(
            "chat_groq.success",
            extra={"tokens_used": tokens, "output_len": len(output)},
        )

        return Response(output=output, tokens_used=tokens)

    except Exception as e:
        logger.exception("chat_groq.error", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))
