import os

from dotenv import load_dotenv

from langchain_groq import ChatGroq


load_dotenv()


GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

GROQ_MODEL = os.getenv(
    "GROQ_MODEL"
)


if not GROQ_API_KEY:

    raise RuntimeError(
        "GROQ_API_KEY is not set"
    )


if not GROQ_MODEL:

    raise RuntimeError(
        "GROQ_MODEL is not set"
    )


llm = ChatGroq(

    model=GROQ_MODEL,

    temperature=0,

    api_key=GROQ_API_KEY
)