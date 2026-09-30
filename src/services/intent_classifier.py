import json
import re
from typing import Any, Dict

from groq import Groq

from src.core.config import settings


# ============================================================
# GROQ CLIENT
# ============================================================

client = Groq(
    api_key=settings.GROQ_API_KEY
)


# ============================================================
# INTENT CLASSIFICATION PROMPT
# ============================================================

INTENT_PROMPT = """
You are an advanced intent classifier for a document-grounded
AI RAG assistant.

The user has uploaded documents such as resumes, PDFs, reports,
notes, certificates, project documents, etc.

Your job is to determine what the user wants and whether the
answer should be retrieved from their uploaded documents.

IMPORTANT RULE:

Do NOT require the user to explicitly mention:
- resume
- document
- PDF
- uploaded file

before deciding that retrieval is required.

Questions such as:

"list my skills"
"what technologies do I know?"
"what projects have I worked on?"
"tell me about my experience"
"what certifications do I have?"

are document queries even when the user does not mention a document.

============================================================
AVAILABLE INTENTS
============================================================

DOCUMENT_QUERY
DOCUMENT_SUMMARY
DOCUMENT_COMPARISON
DOCUMENT_ANALYSIS
CONVERSATION_QUERY
CLARIFICATION
OUT_OF_SCOPE

============================================================
QUERY TYPES
============================================================

GENERAL
SKILLS
TECHNOLOGIES
PROGRAMMING_LANGUAGES
FRAMEWORKS
LIBRARIES
TOOLS
DATABASES
CLOUD
AI_ML
EXPERIENCE
INTERNSHIPS
EDUCATION
PROJECTS
CERTIFICATIONS
ACHIEVEMENTS
RESPONSIBILITIES
WORK_HISTORY
CONTACT_INFORMATION
LOCATION
TIMELINE
STATISTICS
SUMMARY
DETAILED_SUMMARY
COMPARISON
OTHER_DOCUMENT_INFORMATION

============================================================
ANSWER MODES
============================================================

SHORT_ANSWER
DETAILED_ANSWER
STRUCTURED_LIST
TABLE
TIMELINE
SUMMARY
DETAILED_SUMMARY
COMPARISON

============================================================
RULES
============================================================

1. If the question can reasonably be answered using information
   contained in the user's uploaded documents, use:

   DOCUMENT_QUERY

   or another appropriate document intent.

2. Never require words such as "resume", "document", or "PDF".

3. "list my skills" must be:

   DOCUMENT_QUERY
   SKILLS
   requires_retrieval=true

4. "what technologies do I know?" must be:

   DOCUMENT_QUERY
   TECHNOLOGIES
   requires_retrieval=true

5. "what projects have I worked on?" must be:

   DOCUMENT_QUERY
   PROJECTS
   requires_retrieval=true

6. "what certifications do I have?" must be:

   DOCUMENT_QUERY
   CERTIFICATIONS
   requires_retrieval=true

7. "tell me about my experience" must be:

   DOCUMENT_QUERY
   EXPERIENCE
   requires_retrieval=true

8. "summarize my resume" must be:

   DOCUMENT_SUMMARY
   SUMMARY
   requires_retrieval=true

9. "give me everything about my resume" must be:

   DOCUMENT_SUMMARY
   DETAILED_SUMMARY
   requires_retrieval=true
   requires_multiple_chunks=true

10. "compare my projects" must be:

    DOCUMENT_COMPARISON
    COMPARISON
    requires_retrieval=true

11. Questions asking for information from multiple parts of a
    document should set:

    requires_multiple_chunks=true

12. Questions requiring a list of information should normally use:

    STRUCTURED_LIST

13. Questions asking for a complete document summary should use:

    DETAILED_SUMMARY

14. Only use OUT_OF_SCOPE when the question genuinely cannot
    reasonably be answered from the uploaded documents and is
    not a conversation-related question.

15. Never reject a question merely because the user used a short
    or informal sentence.

16. Preserve the user's actual intent.

============================================================
QUERY REWRITING
============================================================

The rewritten query must be optimized for semantic retrieval.

Do NOT simply repeat the user's question.

Examples:

User:
"list my skills"

Rewritten:
"Extract all technical and professional skills explicitly
mentioned in the user's uploaded documents, including programming
languages, frameworks, libraries, databases, tools, cloud
technologies, AI/ML technologies and other relevant skills."

User:
"what technologies do I know?"

Rewritten:
"Extract all technologies, programming languages, frameworks,
libraries, databases, cloud platforms and AI technologies explicitly
mentioned in the user's documents."

User:
"what projects have I worked on?"

Rewritten:
"Extract all projects mentioned in the user's documents,
including project names, descriptions, technologies and
responsibilities."

User:
"what certifications do I have?"

Rewritten:
"Extract all certifications explicitly mentioned in the user's
documents."

User:
"tell me about my experience"

Rewritten:
"Extract the user's professional experience, internships,
companies, roles, responsibilities and relevant work history
mentioned in the documents."

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Use exactly this structure:

{
    "intent": "...",
    "query_type": "...",
    "requires_retrieval": true,
    "requires_multiple_chunks": false,
    "answer_mode": "...",
    "rewritten_query": "..."
}
"""


# ============================================================
# FALLBACK CLASSIFIER
# ============================================================

def fallback_intent(question: str) -> Dict[str, Any]:

    q = question.lower().strip()

    # --------------------------------------------------------
    # Skills
    # --------------------------------------------------------

    if any(word in q for word in [
        "skill",
        "skills",
        "abilities",
        "technical skills"
    ]):

        return {
            "intent": "DOCUMENT_QUERY",
            "query_type": "SKILLS",
            "requires_retrieval": True,
            "requires_multiple_chunks": True,
            "answer_mode": "STRUCTURED_LIST",
            "rewritten_query": (
                "Extract all technical and professional skills "
                "explicitly mentioned in the user's uploaded "
                "documents, including programming languages, "
                "frameworks, libraries, databases, tools, cloud "
                "technologies and AI/ML technologies."
            )
        }

    # --------------------------------------------------------
    # Technologies
    # --------------------------------------------------------

    if any(word in q for word in [
        "technology",
        "technologies",
        "tech stack",
        "tech"
    ]):

        return {
            "intent": "DOCUMENT_QUERY",
            "query_type": "TECHNOLOGIES",
            "requires_retrieval": True,
            "requires_multiple_chunks": True,
            "answer_mode": "STRUCTURED_LIST",
            "rewritten_query": (
                "Extract all technologies, programming languages, "
                "frameworks, libraries, databases, cloud platforms "
                "and AI technologies mentioned in the user's "
                "documents."
            )
        }

    # --------------------------------------------------------
    # Projects
    # --------------------------------------------------------

    if any(word in q for word in [
        "project",
        "projects",
        "built",
        "build"
    ]):

        return {
            "intent": "DOCUMENT_QUERY",
            "query_type": "PROJECTS",
            "requires_retrieval": True,
            "requires_multiple_chunks": True,
            "answer_mode": "STRUCTURED_LIST",
            "rewritten_query": (
                "Extract all projects mentioned in the user's "
                "documents, including project names, descriptions, "
                "technologies and responsibilities."
            )
        }

    # --------------------------------------------------------
    # Certifications
    # --------------------------------------------------------

    if any(word in q for word in [
        "certification",
        "certifications",
        "certificate",
        "certificates"
    ]):

        return {
            "intent": "DOCUMENT_QUERY",
            "query_type": "CERTIFICATIONS",
            "requires_retrieval": True,
            "requires_multiple_chunks": True,
            "answer_mode": "STRUCTURED_LIST",
            "rewritten_query": (
                "Extract all certifications and certificates "
                "explicitly mentioned in the user's documents."
            )
        }

    # --------------------------------------------------------
    # Education
    # --------------------------------------------------------

    if any(word in q for word in [
        "education",
        "degree",
        "college",
        "university",
        "cgpa",
        "qualification"
    ]):

        return {
            "intent": "DOCUMENT_QUERY",
            "query_type": "EDUCATION",
            "requires_retrieval": True,
            "requires_multiple_chunks": True,
            "answer_mode": "STRUCTURED_LIST",
            "rewritten_query": (
                "Extract the user's educational qualifications, "
                "degrees, institutions, academic information and "
                "other education details explicitly mentioned in "
                "the documents."
            )
        }

    # --------------------------------------------------------
    # Experience
    # --------------------------------------------------------

    if any(word in q for word in [
        "experience",
        "internship",
        "internships",
        "work experience",
        "worked"
    ]):

        return {
            "intent": "DOCUMENT_QUERY",
            "query_type": "EXPERIENCE",
            "requires_retrieval": True,
            "requires_multiple_chunks": True,
            "answer_mode": "DETAILED_ANSWER",
            "rewritten_query": (
                "Extract the user's professional experience, "
                "internships, companies, roles, responsibilities "
                "and work history from the documents."
            )
        }

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    if any(word in q for word in [
        "summarize",
        "summary",
        "overview",
        "summarise"
    ]):

        return {
            "intent": "DOCUMENT_SUMMARY",
            "query_type": "SUMMARY",
            "requires_retrieval": True,
            "requires_multiple_chunks": True,
            "answer_mode": "DETAILED_SUMMARY",
            "rewritten_query": (
                "Create a comprehensive summary of the information "
                "contained in the user's uploaded documents."
            )
        }

    # --------------------------------------------------------
    # Generic document query
    # --------------------------------------------------------

    return {
        "intent": "DOCUMENT_QUERY",
        "query_type": "GENERAL",
        "requires_retrieval": True,
        "requires_multiple_chunks": True,
        "answer_mode": "DETAILED_ANSWER",
        "rewritten_query": question
    }


# ============================================================
# MAIN INTENT CLASSIFIER
# ============================================================

def classify_intent(question: str) -> Dict[str, Any]:

    question = question.strip()

    if not question:
        return {
            "intent": "CLARIFICATION",
            "query_type": "GENERAL",
            "requires_retrieval": False,
            "requires_multiple_chunks": False,
            "answer_mode": "SHORT_ANSWER",
            "rewritten_query": ""
        }

    try:

        response = client.chat.completions.create(

            model="llama-3.3-70b-versatile",

            messages=[
                {
                    "role": "system",
                    "content": INTENT_PROMPT
                },
                {
                    "role": "user",
                    "content": question
                }
            ],

            temperature=0,

            max_tokens=500,

            response_format={
                "type": "json_object"
            }
        )

        content = response.choices[0].message.content

        result = json.loads(content)

        # ----------------------------------------------------
        # Validate / normalize
        # ----------------------------------------------------

        allowed_intents = {
            "DOCUMENT_QUERY",
            "DOCUMENT_SUMMARY",
            "DOCUMENT_COMPARISON",
            "DOCUMENT_ANALYSIS",
            "CONVERSATION_QUERY",
            "CLARIFICATION",
            "OUT_OF_SCOPE"
        }

        if result.get("intent") not in allowed_intents:
            return fallback_intent(question)

        # ----------------------------------------------------
        # Safety fallback:
        # document-like questions should retrieve
        # ----------------------------------------------------

        if result.get("intent") == "OUT_OF_SCOPE":

            document_keywords = [
                "my",
                "resume",
                "document",
                "pdf",
                "skill",
                "skills",
                "experience",
                "project",
                "projects",
                "technology",
                "technologies",
                "certification",
                "education",
                "internship",
                "work"
            ]

            if any(
                word in question.lower()
                for word in document_keywords
            ):

                return fallback_intent(question)

        # ----------------------------------------------------
        # Normalize booleans
        # ----------------------------------------------------

        result["requires_retrieval"] = bool(
            result.get(
                "requires_retrieval",
                True
            )
        )

        result["requires_multiple_chunks"] = bool(
            result.get(
                "requires_multiple_chunks",
                True
            )
        )

        # ----------------------------------------------------
        # Missing rewritten query
        # ----------------------------------------------------

        if not result.get("rewritten_query"):

            result["rewritten_query"] = question

        return result

    except Exception as e:

        print(
            "Intent classification error:",
            str(e)
        )

        # Never make the whole RAG system fail
        # because intent classification failed.

        return fallback_intent(question)
