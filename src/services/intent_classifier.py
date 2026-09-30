from typing import Literal

from src.services.llm_service import llm


Intent = Literal[
    "QA",
    "SUMMARY",
    "EXTRACTION",
    "LIST",
    "EXPLANATION",
    "COMPARISON",
    "QUIZ",
    "PRACTICE",
    "DOCUMENT_OVERVIEW",
    "FOLLOW_UP",
]


VALID_INTENTS = {
    "QA",
    "SUMMARY",
    "EXTRACTION",
    "LIST",
    "EXPLANATION",
    "COMPARISON",
    "QUIZ",
    "PRACTICE",
    "DOCUMENT_OVERVIEW",
    "FOLLOW_UP",
}


def _normalize_intent(value: str) -> str:

    if not value:
        return "QA"

    value = value.strip().upper()

    # Handle accidental LLM formatting
    value = value.replace("`", "")
    value = value.replace(".", "")
    value = value.split("\n")[0].strip()

    if value in VALID_INTENTS:
        return value

    return "QA"


def classify_intent(question: str) -> str:

    """
    Classify a user's request into a high-level
    RAG/document intent.

    The classifier intentionally returns ONLY
    the intent name.
    """

    if not question or not question.strip():
        return "QA"

    prompt = f"""
You are an advanced intent classifier for a
document-based AI RAG assistant.

The user may ask questions about uploaded documents,
resumes, PDFs, study material, notes, projects,
certifications, skills, experience, or any other
retrieved knowledge.

Classify the user's request into EXACTLY ONE
of the following intents.

==================================================
AVAILABLE INTENTS
==================================================

QA

Use QA when the user wants a direct factual answer
to a specific question.

Examples:
- What programming language is mentioned?
- Where did I complete my internship?
- What was my CGPA?
- When did I graduate?
- What database did I use?
- How many projects are listed?

--------------------------------------------------

SUMMARY

Use SUMMARY when the user explicitly asks to
summarize, condense, shorten, or provide the main
points of document content.

Examples:
- Summarize my resume.
- Summarize this document.
- Give me a short summary.
- Give me the main points.
- Summarize my internship experience.
- Give me a concise overview of this PDF.

--------------------------------------------------

DOCUMENT_OVERVIEW

Use DOCUMENT_OVERVIEW when the user wants to know
what a document contains generally, without necessarily
asking for a condensed summary.

Examples:
- What is this document about?
- Tell me what is inside this PDF.
- What information does my resume contain?
- Give me an overview of this document.
- What sections are present in my resume?
- Explain what this document contains.

--------------------------------------------------

EXTRACTION

Use EXTRACTION when the user asks to find, extract,
identify, or retrieve specific pieces of information
from a document.

Examples:
- Find all certifications in my resume.
- Extract my internship details.
- What skills are mentioned?
- Find all programming languages.
- List all project names and their descriptions.
- Find every mention of React.
- Extract my education details.

--------------------------------------------------

LIST

Use LIST when the main purpose is to produce a list
of multiple related items from the documents.

Examples:
- List my skills.
- List all my projects.
- List my certifications.
- Give me all technologies.
- Show all programming languages.
- Give me a list of my achievements.

If the request is primarily asking to retrieve multiple
items, prefer LIST over QA.

--------------------------------------------------

EXPLANATION

Use EXPLANATION when the user asks for a detailed
explanation, interpretation, or understanding of
something contained in the document.

Examples:
- Explain my AI project.
- Explain how my RAG system works.
- Explain my internship experience.
- Explain this project in detail.
- Explain the architecture mentioned in my resume.
- Explain my technical skills.

--------------------------------------------------

COMPARISON

Use COMPARISON when the user asks to compare two or
more things.

Examples:
- Compare my two projects.
- Which project used more technologies?
- Compare my internships.
- What are the differences between these projects?
- Compare my MERN project and AI project.

--------------------------------------------------

QUIZ

Use QUIZ when the user explicitly asks for a quiz,
MCQs, questions, or testing based on their documents.

Examples:
- Create a quiz from my resume.
- Give me 10 MCQs.
- Test me on my projects.
- Make a quiz from this PDF.
- Generate questions from my study material.

--------------------------------------------------

PRACTICE

Use PRACTICE when the user wants interview questions,
practice questions, mock interviews, preparation,
or questions designed to test their knowledge.

Examples:
- Give me interview questions based on my resume.
- Prepare me for an interview.
- Ask me questions about my projects.
- Give me Java interview questions from my skills.
- Conduct a mock interview.
- Test my knowledge of this document.

--------------------------------------------------

FOLLOW_UP

Use FOLLOW_UP when the user's question clearly
continues or refers to the previous conversation.

Examples:
- Explain that in more detail.
- Tell me more about it.
- What about the second project?
- Explain the previous answer.
- Give me more details.
- Continue.
- What else?
- Why?
- How?

Only use FOLLOW_UP when the question clearly depends
on previous conversation context.

==================================================
IMPORTANT RULES
==================================================

1. Return EXACTLY ONE intent.

2. Return ONLY the intent name.

3. Never return explanations.

4. Never return JSON.

5. Never return Markdown.

6. Never return multiple intents.

7. If the user asks to "list" multiple things,
   prefer LIST.

8. If the user asks to "find", "extract", or
   retrieve specific information, prefer EXTRACTION.

9. If the user asks to "summarize", prefer SUMMARY.

10. If the user asks "what is this document about?"
    prefer DOCUMENT_OVERVIEW.

11. If the user asks "explain", prefer EXPLANATION.

12. If the user asks to compare things,
    use COMPARISON.

13. If the user asks for MCQs or a quiz,
    use QUIZ.

14. If the user asks for interview preparation
    or practice questions, use PRACTICE.

15. If the request is a normal factual question,
    use QA.

16. Do not classify based only on individual words.
    Understand the user's overall intent.

==================================================
USER REQUEST
==================================================

{question}

==================================================
INTENT
==================================================
"""

    try:

        response = llm.invoke(prompt)

        content = getattr(
            response,
            "content",
            ""
        )

        return _normalize_intent(content)

    except Exception as e:

        print(
            f"Intent classification error: {e}"
        )

        # Safe fallback.
        return "QA"
