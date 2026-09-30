from src.services.llm_service import llm


# ============================================================
# INTENT CLASSIFICATION
# ============================================================

VALID_INTENTS = {
    "DOCUMENT_QA",
    "SUMMARY",
    "QUIZ",
    "GENERAL_QA",
}


def classify_intent(question: str) -> str:

    question = question.strip()

    if not question:
        return "GENERAL_QA"

    prompt = f"""
You are an advanced intent classifier for an AI Knowledge Assistant.

The assistant can answer questions using:
1. User-uploaded documents
2. General world/technical knowledge

Classify the user's request into EXACTLY ONE intent.

============================================================
INTENTS
============================================================

DOCUMENT_QA

Use DOCUMENT_QA when the user is asking about information
that is expected to come from their uploaded documents,
resume, PDFs, files, or knowledge base.

Examples:

"List my skills"
"What skills are mentioned in my resume?"
"What certifications are in my resume?"
"What projects are mentioned in my CV?"
"Tell me about my internship"
"What technologies are listed in my document?"
"What does my resume say about React?"
"Find my education details"
"What experience do I have?"
"What does the uploaded PDF say about authentication?"
"According to my document, what is..."
"From my resume, tell me..."
"Based on my uploaded document..."

------------------------------------------------------------

SUMMARY

Use SUMMARY when the user wants a summary, overview,
key points, explanation, or condensation of a document
or uploaded content.

Examples:

"Summarize my document"
"Give me a summary of my resume"
"Summarize this PDF"
"Give me the key points"
"Explain the main ideas in my document"
"Give me a short overview of this document"
"What are the important points?"
"Summarize pages 1 to 5"

If the request clearly refers to an uploaded document,
use SUMMARY.

------------------------------------------------------------

QUIZ

Use QUIZ when the user wants questions or practice material
generated from their documents OR from a general topic.

Examples:

"Create a quiz from my PDF"
"Give me MCQs from my resume"
"Create 20 questions from this document"
"Test me on this PDF"
"Give me interview questions about Java"
"Create DSA practice questions"
"Give me 10 binary search questions"
"Quiz me on operating systems"

------------------------------------------------------------

GENERAL_QA

Use GENERAL_QA when the user is asking a normal question
that does NOT require information from their uploaded documents.

This includes:

Programming questions
DSA questions
Computer science concepts
Coding explanations
Interview preparation
Mathematics
General knowledge
Concept explanations
Comparisons
How-to questions
Debugging
Algorithms
System design
Technical questions

Examples:

"What is binary search?"
"Explain binary search in Java"
"What is a hash map?"
"Explain recursion"
"What is dynamic programming?"
"How does JWT work?"
"What is REST API?"
"Explain Docker"
"What is an operating system?"
"Explain TCP vs UDP"
"What is DSA OA?"
"How should I prepare for a coding interview?"
"Give me a Java implementation of binary search"
"Why is quicksort O(n log n)?"

These should NOT require document retrieval.

============================================================
IMPORTANT RULES
============================================================

Rule 1:
If the user explicitly mentions:
resume
CV
document
PDF
uploaded file
my file
my document
my resume
my CV
knowledge base
uploaded documents

and asks for information from it,
classify as DOCUMENT_QA or SUMMARY.

Rule 2:
If the user asks to summarize uploaded content,
classify as SUMMARY.

Rule 3:
If the user asks to generate questions, MCQs,
practice questions, or a quiz, classify as QUIZ.

Rule 4:
If the user asks a standalone technical/general question
such as:

"What is binary search?"

classify as GENERAL_QA.

Rule 5:
Do NOT classify every question as DOCUMENT_QA.

Rule 6:
The word "my" alone does not mean the question requires
a document.

Rule 7:
When uncertain between GENERAL_QA and DOCUMENT_QA:

If the question explicitly refers to the user's document,
choose DOCUMENT_QA.

Otherwise choose GENERAL_QA.

Rule 8:
Return ONLY the intent name.

Valid outputs:

DOCUMENT_QA
SUMMARY
QUIZ
GENERAL_QA

============================================================
USER QUESTION
============================================================

{question}

============================================================
INTENT
============================================================
"""

    try:

        response = llm.invoke(prompt)

        intent = response.content.strip().upper()

        # Remove accidental formatting
        intent = (
            intent
            .replace("`", "")
            .replace(".", "")
            .strip()
        )

        # Handle cases where model returns extra text
        for valid_intent in VALID_INTENTS:

            if valid_intent in intent:

                return valid_intent

    except Exception as e:

        print(
            "Intent classification error:",
            repr(e)
        )

    # Safe fallback:
    # General questions should still work even if
    # intent classification fails.
    return "GENERAL_QA"
