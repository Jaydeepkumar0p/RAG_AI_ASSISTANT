import json

from groq import Groq

from src.core.config import settings


# ======================================================
# GROQ CLIENT
# ======================================================

client = Groq(
    api_key=settings.GROQ_API_KEY
)


# ======================================================
# INTENT PROMPT
# ======================================================

SYSTEM_PROMPT = """
You are an advanced intent classifier for a document
intelligence and RAG application.

The user may ask questions about uploaded PDFs,
resumes, documents, or their knowledge base.

Classify the user's request into exactly ONE intent.

AVAILABLE INTENTS:

DOCUMENT_QA
DOCUMENT_SUMMARY
DOCUMENT_DETAILED_SUMMARY
DOCUMENT_SKILLS
DOCUMENT_EXPERIENCE
DOCUMENT_EDUCATION
DOCUMENT_PROJECTS
DOCUMENT_CERTIFICATIONS
DOCUMENT_TECHNOLOGIES
DOCUMENT_CONTACT_INFO
DOCUMENT_ACHIEVEMENTS
DOCUMENT_EXTRACT
DOCUMENT_COMPARE
DOCUMENT_TIMELINE
DOCUMENT_OVERVIEW
GENERAL_CHAT
UNKNOWN


INTENT DEFINITIONS
==================

DOCUMENT_QA

Specific factual question about the document.

Examples:
- What is my CGPA?
- Where did I study?
- Where did I intern?
- What was my role?
- When did I graduate?


DOCUMENT_SUMMARY

User wants a normal summary.

Examples:
- Summarize my document
- Summarize my resume
- Give me a summary
- Give me a brief summary


DOCUMENT_DETAILED_SUMMARY

User wants a comprehensive explanation.

Examples:
- Explain my entire resume
- Give me a detailed summary
- Tell me everything important in my resume
- Explain the document section by section


DOCUMENT_SKILLS

Questions about skills.

Examples:
- List my skills
- What skills do I have?
- What technical skills are in my resume?
- Show all my skills


DOCUMENT_EXPERIENCE

Questions about work experience or internships.

Examples:
- What experience do I have?
- Where did I work?
- List my internships
- Tell me about my work experience


DOCUMENT_EDUCATION

Questions about academic information.

Examples:
- What is my degree?
- Where did I study?
- What is my CGPA?
- Tell me about my education


DOCUMENT_PROJECTS

Questions about projects.

Examples:
- List my projects
- What projects have I built?
- Explain my projects


DOCUMENT_CERTIFICATIONS

Questions about certifications.

Examples:
- What certifications do I have?
- List my certificates


DOCUMENT_TECHNOLOGIES

Questions about technologies.

Examples:
- What technologies do I know?
- List technologies from my resume
- What frameworks do I use?
- What programming languages are listed?


DOCUMENT_CONTACT_INFO

Questions about contact information.

Examples:
- What is my email?
- What is my GitHub?
- What is my LinkedIn?


DOCUMENT_ACHIEVEMENTS

Questions about achievements.

Examples:
- What are my achievements?
- List my awards
- What accomplishments are mentioned?


DOCUMENT_EXTRACT

User wants structured extraction.

Examples:
- Extract all company names
- Extract all dates
- Extract all programming languages
- Extract all technologies
- Extract all organizations


DOCUMENT_COMPARE

User wants comparison between documents,
experiences, projects, skills, etc.


DOCUMENT_TIMELINE

User wants chronological information.

Examples:
- Give me my career timeline
- Show my education timeline


DOCUMENT_OVERVIEW

Broad overview of a document.

Examples:
- Give me an overview
- What does my resume contain?
- Tell me about this document


GENERAL_CHAT

Normal conversation that does not require document retrieval.


UNKNOWN

Cannot confidently determine the intent.


IMPORTANT:

If the user refers to:

resume
CV
PDF
document
uploaded file
my file
my resume
my document
my experience
my skills
my projects

then prefer a DOCUMENT_* intent.

Return ONLY valid JSON.

Required format:

{
    "intent": "DOCUMENT_QA",
    "confidence": 0.95,
    "requires_retrieval": true,
    "requires_generation": true,
    "rewritten_query": "clean retrieval query"
}
"""


# ======================================================
# INTENT DETECTION
# ======================================================

def detect_intent(question: str) -> dict:

    try:

        response = client.chat.completions.create(

            model="openai/gpt-oss-120b",

            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": question
                }
            ],

            temperature=0,

            response_format={
                "type": "json_object"
            }
        )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        result = json.loads(content)

        intent = result.get(
            "intent",
            "DOCUMENT_QA"
        )

        confidence = float(
            result.get(
                "confidence",
                0.5
            )
        )

        rewritten_query = result.get(
            "rewritten_query",
            question
        )

        return {

            "intent": intent,

            "confidence": confidence,

            "requires_retrieval": bool(
                result.get(
                    "requires_retrieval",
                    True
                )
            ),

            "requires_generation": bool(
                result.get(
                    "requires_generation",
                    True
                )
            ),

            "rewritten_query":
                rewritten_query
        }

    except Exception as e:

        print(
            "Intent classification error:",
            repr(e)
        )

        # Safe fallback
        return {

            "intent":
                "DOCUMENT_QA",

            "confidence":
                0.0,

            "requires_retrieval":
                True,

            "requires_generation":
                True,

            "rewritten_query":
                question
        }
