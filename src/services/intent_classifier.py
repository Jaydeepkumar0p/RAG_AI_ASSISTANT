import json
import re
from typing import Dict, Any

from groq import Groq

from src.core.config import settings


# ============================================================
# GROQ CLIENT
# ============================================================

client = Groq(
    api_key=settings.GROQ_API_KEY
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "openai/gpt-oss-120b"


# ============================================================
# VALID INTENTS
# ============================================================

VALID_INTENTS = {
    "QA",
    "SUMMARY",
    "LIST",
    "COMPARISON",
    "EXTRACTION",
    "EXPLANATION",
    "SKILLS",
    "EXPERIENCE",
    "EDUCATION",
    "PROJECTS",
    "CERTIFICATIONS",
    "CONTACT",
    "DOCUMENT_INFO",
    "FOLLOW_UP",
    "GREETING",
    "OUT_OF_SCOPE",
}


# ============================================================
# SYSTEM PROMPT
# ============================================================

INTENT_SYSTEM_PROMPT = """
You are an advanced intent classifier for a personal document
RAG assistant.

The user has uploaded documents such as:

- Resume
- CV
- Certificates
- Project documents
- Notes
- PDFs
- Professional documents

Your job is to understand the user's REAL information need.

You must classify the request into exactly ONE intent.

Available intents:

QA
SUMMARY
LIST
COMPARISON
EXTRACTION
EXPLANATION
SKILLS
EXPERIENCE
EDUCATION
PROJECTS
CERTIFICATIONS
CONTACT
DOCUMENT_INFO
FOLLOW_UP
GREETING
OUT_OF_SCOPE


INTENT DEFINITIONS
==================

QA:
Questions asking for specific information from documents.

Examples:
- What technologies are in my resume?
- Where did I work?
- What is my CGPA?
- What projects did I build?

SUMMARY:
User wants a summary of a document or its contents.

Examples:
- Summarize my resume
- Give me a summary of this document
- Explain my resume briefly
- Give me the main points

LIST:
User wants a list of items.

Examples:
- List my skills
- List all my projects
- Show all technologies
- Give me all certifications

EXTRACTION:
User wants specific structured information extracted.

Examples:
- Extract my email
- Extract all company names
- Extract dates from my resume
- Extract all programming languages

COMPARISON:
User wants two or more things compared.

Examples:
- Compare my two projects
- Which project uses more technologies?
- Compare my internships

EXPLANATION:
User wants an explanation of something contained in the documents.

Examples:
- Explain my RAG project
- Explain how my AI project works
- Explain my internship experience

SKILLS:
Questions specifically about skills, technologies, tools,
programming languages, frameworks, databases or technical abilities.

Examples:
- What are my skills?
- List my technical skills
- What technologies do I know?
- What frameworks are in my resume?

EXPERIENCE:
Questions about work experience, internships or employment.

Examples:
- Tell me about my experience
- Where did I intern?
- What companies have I worked with?

EDUCATION:
Questions about education, degree, university, CGPA or academic history.

Examples:
- Where did I study?
- What is my CGPA?
- What degree am I pursuing?

PROJECTS:
Questions about projects.

Examples:
- Tell me about my projects
- What projects have I built?
- Explain my CRM project

CERTIFICATIONS:
Questions about certificates or certifications.

Examples:
- What certifications do I have?
- List my certificates

CONTACT:
Questions about contact information.

Examples:
- What is my email?
- What is my phone number?
- What is my LinkedIn?

DOCUMENT_INFO:
Questions about the document itself.

Examples:
- What document did I upload?
- How many pages does my document have?
- What files have I uploaded?

FOLLOW_UP:
A question that clearly continues the previous conversation.

Examples:
User: What projects did I build?
User: Explain the first one.

GREETING:
Simple conversational greetings.

Examples:
- Hi
- Hello
- Hey
- Good morning

OUT_OF_SCOPE:
Questions unrelated to the uploaded documents or the assistant's purpose.

Examples:
- What is the weather?
- Write me a game
- Who won the football match?


IMPORTANT RULES
===============

1. Understand the semantic meaning, not just keywords.

2. "List my skills", "what are my skills", "show my technologies",
   and "technical skills in my resume" should normally be SKILLS.

3. "Summarize my document" should be SUMMARY.

4. "Tell me everything about my resume" should be SUMMARY.

5. "Extract my email" should be EXTRACTION or CONTACT.
   Prefer CONTACT when asking specifically for contact information.

6. Follow-up questions should use FOLLOW_UP when they depend
   strongly on previous conversation context.

7. Return ONLY valid JSON.

Required JSON format:

{
    "intent": "SKILLS",
    "confidence": 0.98,
    "reason": "The user is asking for technical skills from the document."
}

The intent MUST be exactly one of the allowed intents.
"""


# ============================================================
# JSON CLEANER
# ============================================================

def _extract_json(text: str) -> Dict[str, Any]:

    if not text:
        return {
            "intent": "QA",
            "confidence": 0.0,
            "reason": "Empty classifier response"
        }

    text = text.strip()

    # Remove markdown code fences
    text = re.sub(
        r"```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```\s*",
        "",
        text
    )

    # Try direct JSON
    try:

        data = json.loads(text)

        if isinstance(data, dict):
            return data

    except json.JSONDecodeError:
        pass

    # Try finding JSON object inside response
    match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL
    )

    if match:

        try:

            data = json.loads(
                match.group(0)
            )

            if isinstance(data, dict):
                return data

        except json.JSONDecodeError:
            pass

    return {
        "intent": "QA",
        "confidence": 0.0,
        "reason": "Could not parse classifier response"
    }


# ============================================================
# NORMALIZE INTENT
# ============================================================

def _normalize_intent(
    intent: str
) -> str:

    if not intent:
        return "QA"

    intent = str(intent).strip().upper()

    # Common aliases
    aliases = {

        "QUESTION": "QA",

        "QUESTION_ANSWERING": "QA",

        "SUMMARIZATION": "SUMMARY",

        "BULLET_LIST": "LIST",

        "TECHNICAL_SKILLS": "SKILLS",

        "TECHNOLOGY": "SKILLS",

        "TECHNOLOGIES": "SKILLS",

        "WORK_EXPERIENCE": "EXPERIENCE",

        "ACADEMICS": "EDUCATION",

        "CERTIFICATE": "CERTIFICATIONS",

        "CERTIFICATES": "CERTIFICATIONS",

        "DOCUMENT": "DOCUMENT_INFO",

        "GREETING_MESSAGE": "GREETING",

    }

    intent = aliases.get(
        intent,
        intent
    )

    if intent not in VALID_INTENTS:
        return "QA"

    return intent


# ============================================================
# MAIN CLASSIFIER
# ============================================================

def classify_intent(
    question: str,
    conversation_context: str = ""
) -> Dict[str, Any]:

    if not question:

        return {
            "intent": "QA",
            "confidence": 0.0,
            "reason": "Empty question"
        }

    question = question.strip()

    # --------------------------------------------------------
    # Very cheap local handling for greetings
    # --------------------------------------------------------

    greeting_words = {
        "hi",
        "hello",
        "hey",
        "hey there",
        "good morning",
        "good afternoon",
        "good evening"
    }

    if question.lower() in greeting_words:

        return {
            "intent": "GREETING",
            "confidence": 1.0,
            "reason": "Simple greeting"
        }

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context_text = ""

    if conversation_context:

        context_text = f"""
Previous conversation context:

{conversation_context}

Use this context only to understand follow-up questions.
"""

    user_prompt = f"""
Classify this user request.

{context_text}

Current user request:

{question}

Return ONLY JSON.
"""


    # --------------------------------------------------------
    # Groq request
    # --------------------------------------------------------

    try:

        response = client.chat.completions.create(

            model=MODEL_NAME,

            messages=[

                {
                    "role": "system",
                    "content": INTENT_SYSTEM_PROMPT
                },

                {
                    "role": "user",
                    "content": user_prompt
                }

            ],

            temperature=0,

            max_tokens=300
        )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        data = _extract_json(
            content
        )

        intent = _normalize_intent(
            data.get("intent")
        )

        confidence = data.get(
            "confidence",
            0.5
        )

        try:

            confidence = float(
                confidence
            )

        except (
            TypeError,
            ValueError
        ):

            confidence = 0.5

        confidence = max(
            0.0,
            min(
                1.0,
                confidence
            )
        )

        return {

            "intent": intent,

            "confidence": confidence,

            "reason": str(
                data.get(
                    "reason",
                    ""
                )
            )

        }

    except Exception as e:

        print(
            "Intent classification error:",
            repr(e)
        )

        # Never crash the complete RAG pipeline
        return {

            "intent": "QA",

            "confidence": 0.0,

            "reason": "Intent classifier unavailable"

        }
