from src.services.llm_service import llm


def classify_intent(question: str) -> str:

    prompt = f"""
You are an intent classifier for an AI Study Assistant.

Classify the user's request into exactly one category:

QA
SUMMARY
QUIZ

Definitions:

QA:
The user wants an answer to a specific question.

SUMMARY:
The user wants a summary, overview, or explanation
of a document or topic.

QUIZ:
The user wants questions, MCQs, a quiz, or practice
questions generated from their documents.

Return ONLY one word:
QA
SUMMARY
or
QUIZ

User question:
{question}

Intent:
"""

    response = llm.invoke(prompt)

    intent = response.content.strip().upper()

    if intent not in {
        "QA",
        "SUMMARY",
        "QUIZ"
    }:
        return "QA"

    return intent