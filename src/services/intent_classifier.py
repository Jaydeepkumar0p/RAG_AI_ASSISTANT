from src.services.llm_service import llm


# ============================================================
# INTENT CLASSIFIER
# ============================================================

def classify_intent(question: str) -> str:

    prompt = f"""
You are the intent classifier for an AI Study Assistant
that answers questions from the user's uploaded documents.

Your job is to understand the user's actual intention.

Classify the request into EXACTLY ONE of these categories:

QA
SUMMARY
QUIZ


============================================================
QA
============================================================

Use QA when the user wants specific information, facts,
details, extraction, lists, or answers from their documents.

Examples:

"What is my CGPA?"
"What technologies are in my resume?"
"List my skills"
"List my technical skills"
"What programming languages do I know?"
"What projects have I built?"
"Which companies have I worked with?"
"What certifications do I have?"
"Where did I complete my internship?"
"What is my education?"
"What is my email?"
"Tell me about my React project"
"Explain my internship"
"How many projects are mentioned?"
"What tools are mentioned in my resume?"

IMPORTANT:

Requests such as:

"list my skills"
"show my skills"
"what are my skills"
"list technologies from my resume"
"what technologies do I know"

are still QA because the user is asking for
specific information from the uploaded document.


============================================================
SUMMARY
============================================================

Use SUMMARY when the user wants a broad overview,
summary, complete understanding, or main points
of a document.

Examples:

"Summarize my document"
"Summarize my resume"
"Give me a summary of my resume"
"Give me a complete summary"
"Give me an overview of this document"
"What is this document about?"
"Tell me everything important in my document"
"Give me the main points"
"Explain my resume overall"
"Give me a detailed overview of my PDF"
"Summarize the entire document"


IMPORTANT:

If the user asks to summarize the ENTIRE document,
choose SUMMARY.

If the user asks about ONE specific piece of information,
choose QA.


============================================================
QUIZ
============================================================

Use QUIZ when the user wants questions or practice
generated from their documents.

Examples:

"Quiz me"
"Create a quiz"
"Give me MCQs"
"Generate multiple choice questions"
"Test my knowledge"
"Give me practice questions"
"Create 10 questions from my document"
"Make a quiz from my resume"
"Ask me questions about this PDF"
"Generate interview questions from my resume"


============================================================
IMPORTANT CLASSIFICATION RULES
============================================================

1. Understand the meaning of the complete request.

2. Do NOT classify only based on keywords.

3. If the user wants a broad summary or overview
   of an entire document → SUMMARY.

4. If the user asks for specific information
   from the document → QA.

5. If the user wants questions or tests generated
   from the document → QUIZ.

6. "List my skills" → QA.

7. "List all technologies in my resume" → QA.

8. "Tell me everything about my resume" → SUMMARY.

9. "Summarize my resume" → SUMMARY.

10. "Create interview questions from my resume" → QUIZ.

11. "What projects are in my resume?" → QA.

12. "Explain my entire resume" → SUMMARY.

13. Never return explanations.

14. Never return multiple categories.

15. Return ONLY ONE WORD.


============================================================
USER REQUEST
============================================================

{question}


============================================================
INTENT
============================================================
"""

    try:

        response = llm.invoke(
            prompt
        )

        intent = (
            response.content
            .strip()
            .upper()
        )

    except Exception as e:

        print(
            "Intent classification error:",
            repr(e)
        )

        # Safe fallback
        return "QA"


    # ========================================================
    # VALIDATE INTENT
    # ========================================================

    if intent not in {
        "QA",
        "SUMMARY",
        "QUIZ"
    }:

        return "QA"


    return intent
