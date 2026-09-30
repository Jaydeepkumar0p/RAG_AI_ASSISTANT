"""
Intent classification for the AI Study Assistant.

Supported intents:

    QA
    SUMMARY
    QUIZ
    CODING
    GENERAL

Design goals:

1. Fast:
   Obvious questions are classified without an LLM call.

2. Document-aware:
   Questions about "me", "my resume", certificates, skills,
   projects, experience, etc. are routed to document retrieval.

3. Conversation-aware:
   Follow-up questions such as "which one?" or "who issued it?"
   can inherit the previous document context.

4. Conservative:
   Coding questions are not allowed to incorrectly override
   an obvious resume/document question.

5. Deterministic:
   The same input produces the same result.
"""

from __future__ import annotations

import re
from typing import Any


# ============================================================
# VALID INTENTS
# ============================================================

QA = "QA"
SUMMARY = "SUMMARY"
QUIZ = "QUIZ"
CODING = "CODING"
GENERAL = "GENERAL"


VALID_INTENTS = {
    QA,
    SUMMARY,
    QUIZ,
    CODING,
    GENERAL,
}


# ============================================================
# NORMALIZATION
# ============================================================

def _normalize(
    text: str | None,
) -> str:
    """
    Normalize user text for reliable matching.
    """

    return " ".join(
        str(
            text or ""
        )
        .strip()
        .lower()
        .split()
    )


# ============================================================
# HISTORY HELPERS
# ============================================================

def _history_text(
    history: list[dict[str, Any]] | None,
) -> str:
    """
    Convert recent conversation history to searchable text.
    """

    if not history:
        return ""

    recent_messages = history[-8:]

    parts: list[str] = []

    for message in recent_messages:

        if not isinstance(
            message,
            dict,
        ):
            continue

        content = message.get(
            "content",
            "",
        )

        if content:
            parts.append(
                str(content)
            )

    return _normalize(
        " ".join(parts)
    )


# ============================================================
# REGEX HELPER
# ============================================================

def _contains_pattern(
    text: str,
    patterns: list[str],
) -> bool:
    """
    Return True if any regex pattern matches.
    """

    for pattern in patterns:

        if re.search(
            pattern,
            text,
            re.IGNORECASE,
        ):
            return True

    return False


# ============================================================
# EXPLICIT DOCUMENT SIGNALS
# ============================================================

DOCUMENT_PATTERNS = [

    # Resume / CV
    r"\bresume\b",
    r"\bmy resume\b",
    r"\bthe resume\b",
    r"\bcv\b",
    r"\bmy cv\b",
    r"\bthe cv\b",

    # Documents / PDF
    r"\bmy document\b",
    r"\bmy documents\b",
    r"\bthe document\b",
    r"\bmy pdf\b",
    r"\bthe pdf\b",
    r"\buploaded document\b",
    r"\buploaded documents\b",
    r"\buploaded pdf\b",
    r"\buploaded file\b",

    # Explicit document relation
    r"\bin my resume\b",
    r"\bin my cv\b",
    r"\bfrom my resume\b",
    r"\bfrom my cv\b",
    r"\baccording to my resume\b",
    r"\baccording to my cv\b",
    r"\bmentioned in my resume\b",
    r"\bmentioned in my cv\b",
    r"\blisted in my resume\b",
    r"\blisted in my cv\b",
    r"\bshown in my resume\b",
    r"\bshown in my cv\b",
]


# ============================================================
# PERSONAL PROFILE / RESUME SIGNALS
# ============================================================

PERSONAL_PROFILE_PATTERNS = [

    # About me
    r"\btell me about me\b",
    r"\btell me about myself\b",
    r"\bwhat do you know about me\b",
    r"\bwhat can you tell me about me\b",
    r"\bwho am i\b",
    r"\bdescribe me\b",
    r"\bdescribe myself\b",

    # Profile / background
    r"\bmy profile\b",
    r"\bmy background\b",
    r"\bmy professional profile\b",
    r"\bmy professional background\b",
    r"\bmy career\b",
    r"\bmy career profile\b",

    # Skills
    r"\bmy skills\b",
    r"\bwhat are my skills\b",
    r"\bwhat skills do i have\b",
    r"\blist my skills\b",
    r"\bshow my skills\b",

    # Technologies
    r"\bmy technologies\b",
    r"\bmy tech stack\b",
    r"\bwhat technologies do i know\b",
    r"\bwhat technologies do i have\b",
    r"\blist my technologies\b",

    # Projects
    r"\bmy projects\b",
    r"\bwhat projects do i have\b",
    r"\blist my projects\b",
    r"\bshow my projects\b",

    # Experience
    r"\bmy experience\b",
    r"\bmy work experience\b",
    r"\bmy professional experience\b",
    r"\bwhat experience do i have\b",
    r"\blist my experience\b",

    # Education
    r"\bmy education\b",
    r"\bwhat education do i have\b",
    r"\bmy degree\b",
    r"\bmy university\b",
    r"\bmy qualification\b",
    r"\bmy qualifications\b",

    # Internships
    r"\bmy internship\b",
    r"\bmy internships\b",

    # Achievements
    r"\bmy achievement\b",
    r"\bmy achievements\b",
    r"\bwhat achievements do i have\b",

    # Certificates
    r"\bmy certificate\b",
    r"\bmy certificates\b",
    r"\bmy certification\b",
    r"\bmy certifications\b",
    r"\bwhat certificates do i have\b",
    r"\bwhat certifications do i have\b",
    r"\blist my certificates\b",
    r"\blist my certifications\b",
    r"\bshow my certificates\b",
    r"\bshow my certifications\b",
]


# ============================================================
# CERTIFICATE PATTERNS
# ============================================================

CERTIFICATE_PATTERNS = [

    r"\bcertificate\b",
    r"\bcertificates\b",
    r"\bcertification\b",
    r"\bcertifications\b",
    r"\bcredential\b",
    r"\bcredentials\b",
]


CERTIFICATE_ACTION_PATTERNS = [

    r"\blist\b",
    r"\bshow\b",
    r"\btell me\b",
    r"\bwhat\b",
    r"\bwhich\b",
    r"\bhave\b",
    r"\bearned\b",
    r"\bcompleted\b",
    r"\bmentioned\b",
    r"\blisted\b",
    r"\bissued\b",
    r"\bissuer\b",
]


# ============================================================
# SUMMARY PATTERNS
# ============================================================

SUMMARY_PATTERNS = [

    r"\bsummarize\b",
    r"\bsummarise\b",
    r"\bsummary\b",
    r"\bgive me a summary\b",
    r"\bmake a summary\b",
    r"\bbrief summary\b",
    r"\bshort summary\b",
    r"\bprovide a summary\b",
]


# ============================================================
# QUIZ PATTERNS
# ============================================================

QUIZ_PATTERNS = [

    r"\bquiz\b",
    r"\bmcq\b",
    r"\bmcqs\b",
    r"\bmultiple choice\b",
    r"\bmultiple-choice\b",
    r"\btest me\b",
    r"\bcreate questions\b",
    r"\bgenerate questions\b",
    r"\bmake questions\b",
]


# ============================================================
# CODING PATTERNS
# ============================================================

CODING_PATTERNS = [

    # Generic coding
    r"\bcoding\b",
    r"\bprogramming\b",
    r"\bwrite code\b",
    r"\bprovide code\b",
    r"\bgive me code\b",
    r"\bshow me code\b",
    r"\bcode for\b",
    r"\bimplement\b",
    r"\bimplementation\b",

    # Debugging
    r"\bdebug\b",
    r"\bdebugging\b",
    r"\bdebug this\b",
    r"\bfix my code\b",
    r"\bfix this code\b",
    r"\bwhy does this code\b",
    r"\bwhy is this code\b",
    r"\berror in my code\b",
    r"\boptimize this code\b",
    r"\boptimize my code\b",
    r"\brefactor\b",

    # DSA
    r"\bdsa\b",
    r"\bdata structure\b",
    r"\bdata structures\b",
    r"\balgorithm\b",
    r"\balgorithms\b",
    r"\bleetcode\b",
    r"\bhackerrank\b",
    r"\bhackwithinfy\b",

    # Common algorithms / patterns
    r"\bbinary search\b",
    r"\blinear search\b",
    r"\bdynamic programming\b",
    r"\bsliding window\b",
    r"\btwo pointers\b",
    r"\bbacktracking\b",
    r"\brecursion\b",
    r"\bdfs\b",
    r"\bbfs\b",
    r"\bdepth first search\b",
    r"\bbreadth first search\b",
    r"\btopological sort\b",
    r"\bdijkstra\b",
    r"\bshortest path\b",
    r"\bminimum spanning tree\b",
    r"\bunion find\b",
    r"\bdisjoint set\b",

    # Data structures
    r"\blinked list\b",
    r"\barray\b",
    r"\bstack\b",
    r"\bqueue\b",
    r"\bdeque\b",
    r"\bheap\b",
    r"\bpriority queue\b",
    r"\bhashmap\b",
    r"\bhash map\b",
    r"\bhash table\b",
    r"\btree\b",
    r"\bbinary tree\b",
    r"\bbinary search tree\b",
    r"\bgraph\b",
    r"\btrie\b",

    # Complexity
    r"\btime complexity\b",
    r"\bspace complexity\b",
    r"\bbig o\b",
    r"\bo\(\w+\)\b",

    # Languages
    r"\bpython code\b",
    r"\bjava code\b",
    r"\bc\+\+ code\b",
    r"\bcpp code\b",
    r"\bjavascript code\b",
    r"\btypescript code\b",
    r"\breact code\b",
    r"\bnode\.?js code\b",
    r"\bsql query\b",
]


# ============================================================
# GENERAL AI SIGNALS
# ============================================================

GENERAL_PATTERNS = [

    r"\bwhat is\b",
    r"\bwhat are\b",
    r"\bwhy is\b",
    r"\bwhy are\b",
    r"\bhow does\b",
    r"\bhow do\b",
    r"\bhow can\b",
    r"\bexplain\b",
    r"\bteach me\b",
    r"\blearn\b",
    r"\bhelp me learn\b",
    r"\bcan you help me\b",
    r"\bwhat does\b",
    r"\bdefine\b",
    r"\bdifference between\b",
    r"\bcompare\b",
]


# ============================================================
# DOCUMENT HISTORY TERMS
# ============================================================

DOCUMENT_HISTORY_TERMS = [

    "resume",
    "cv",
    "document",
    "pdf",
    "uploaded",
    "certificate",
    "certificates",
    "certification",
    "certifications",
    "skills",
    "technologies",
    "technology",
    "projects",
    "experience",
    "education",
    "internship",
    "internships",
    "achievement",
    "achievements",
    "profile",
    "background",
]


# ============================================================
# FOLLOW-UP PATTERNS
# ============================================================

FOLLOWUP_PATTERNS = [

    # Pronoun/reference follow-ups
    r"^it$",
    r"^that$",
    r"^this$",
    r"^they$",
    r"^them$",
    r"^that one$",
    r"^this one$",
    r"^the other one$",

    # Selection / continuation
    r"^which one$",
    r"^which one was",
    r"^which was",
    r"^what about it$",
    r"^what about that$",

    # Follow-up details
    r"^who issued it$",
    r"^who issued that$",
    r"^when was it$",
    r"^when was that$",
    r"^when did it$",
    r"^tell me more$",
    r"^tell me more about it$",
    r"^explain that$",
    r"^explain it$",

    # Context references
    r"^the previous one$",
    r"^the previous answer$",
    r"^the above$",
    r"^previous answer$",
    r"^previous question$",
]


# ============================================================
# DOCUMENT QUESTION DETECTION
# ============================================================

def _is_explicit_document_question(
    text: str,
) -> bool:
    """
    Detect questions that clearly refer to uploaded documents.
    """

    if _contains_pattern(
        text,
        DOCUMENT_PATTERNS,
    ):
        return True


    if _contains_pattern(
        text,
        PERSONAL_PROFILE_PATTERNS,
    ):
        return True


    # --------------------------------------------------------
    # Certificate / certification
    # --------------------------------------------------------

    has_certificate = _contains_pattern(
        text,
        CERTIFICATE_PATTERNS,
    )

    has_certificate_action = _contains_pattern(
        text,
        CERTIFICATE_ACTION_PATTERNS,
    )

    if (
        has_certificate
        and has_certificate_action
    ):
        return True


    return False


# ============================================================
# DOCUMENT FOLLOW-UP DETECTION
# ============================================================

def _is_document_followup(
    text: str,
    history: list[dict[str, Any]] | None,
) -> bool:
    """
    Detect contextual follow-ups such as:

        Q1: What certificates do I have?
        Q2: Which one was completed in 2025?

    The second question should remain in the document workflow.
    """

    if not history:
        return False


    history_text = _history_text(
        history
    )

    if not history_text:
        return False


    has_document_context = any(
        term in history_text
        for term in DOCUMENT_HISTORY_TERMS
    )

    if not has_document_context:
        return False


    return _contains_pattern(
        text,
        FOLLOWUP_PATTERNS,
    )


# ============================================================
# DIRECT SUMMARY DETECTION
# ============================================================

def _is_summary_question(
    text: str,
) -> bool:

    return _contains_pattern(
        text,
        SUMMARY_PATTERNS,
    )


# ============================================================
# DIRECT QUIZ DETECTION
# ============================================================

def _is_quiz_question(
    text: str,
) -> bool:

    return _contains_pattern(
        text,
        QUIZ_PATTERNS,
    )


# ============================================================
# DIRECT CODING DETECTION
# ============================================================

def _is_coding_question(
    text: str,
) -> bool:

    return _contains_pattern(
        text,
        CODING_PATTERNS,
    )


# ============================================================
# MAIN CLASSIFIER
# ============================================================

def classify_intent(
    question: str,
    history: list[dict[str, Any]] | None = None,
) -> str:
    """
    Classify the user's request.

    Priority:

        1. Explicit document/profile request
        2. Document follow-up
        3. Document summary
        4. Document quiz
        5. Coding
        6. General

    Why document first?

    Example:

        "What programming languages are in my resume?"

    contains "programming", but this is NOT a coding request.
    It is a resume QA request.

    Therefore document context has priority over coding.
    """

    text = _normalize(
        question
    )


    # --------------------------------------------------------
    # EMPTY
    # --------------------------------------------------------

    if not text:
        return GENERAL


    # --------------------------------------------------------
    # EXPLICIT DOCUMENT QUESTION
    # --------------------------------------------------------

    document_question = (
        _is_explicit_document_question(
            text
        )
    )


    # --------------------------------------------------------
    # CONTEXTUAL DOCUMENT FOLLOW-UP
    # --------------------------------------------------------

    if _is_document_followup(
        text,
        history,
    ):
        document_question = True


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    if (
        document_question
        and _is_summary_question(
            text
        )
    ):
        return SUMMARY


    # --------------------------------------------------------
    # QUIZ
    # --------------------------------------------------------

    if (
        document_question
        and _is_quiz_question(
            text
        )
    ):
        return QUIZ


    # --------------------------------------------------------
    # DOCUMENT QA
    # --------------------------------------------------------

    if document_question:
        return QA


    # --------------------------------------------------------
    # CODING
    # --------------------------------------------------------

    if _is_coding_question(
        text
    ):
        return CODING


    # --------------------------------------------------------
    # GENERAL
    # --------------------------------------------------------

    return GENERAL
