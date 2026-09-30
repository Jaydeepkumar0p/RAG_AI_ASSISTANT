import re


VALID_INTENTS = {
    "QA",
    "SUMMARY",
    "QUIZ",
    "CODING",
    "GENERAL",
}


def _normalize(
    text: str,
) -> str:
    return " ".join(
        str(text or "")
        .strip()
        .lower()
        .split()
    )


# ======================================================
# DOCUMENT REFERENCE
# ======================================================

def _document_reference(
    question: str,
    history: list[dict] | None = None,
) -> bool:
    text = _normalize(
        question
    )

    # --------------------------------------------------
    # Explicit document phrases
    # --------------------------------------------------

    document_patterns = [
        r"\bmy resume\b",
        r"\bmy cv\b",
        r"\bthe resume\b",
        r"\bthe cv\b",
        r"\bmy document\b",
        r"\bmy documents\b",
        r"\bthe document\b",
        r"\bmy pdf\b",
        r"\bthe pdf\b",
        r"\buploaded document\b",
        r"\buploaded pdf\b",
        r"\buploaded file\b",
        r"\bin my resume\b",
        r"\bin my cv\b",
        r"\bin the resume\b",
        r"\bin the cv\b",
        r"\bin my document\b",
        r"\bfrom my resume\b",
        r"\bfrom my cv\b",
        r"\bfrom my document\b",
        r"\baccording to my resume\b",
        r"\baccording to my cv\b",
        r"\baccording to my document\b",
        r"\bmentioned in my resume\b",
        r"\bmentioned in my cv\b",
        r"\blisted in my resume\b",
        r"\blisted in my cv\b",
        r"\bmy skills\b",
        r"\bmy education\b",
        r"\bmy experience\b",
        r"\bmy projects\b",
        r"\bmy internship\b",
        r"\bmy internships\b",
        r"\bmy certification\b",
        r"\bmy certifications\b",
        r"\bmy certificate\b",
        r"\bmy certificates\b",
        r"\bmy achievement\b",
        r"\bmy achievements\b",
    ]

    if any(
        re.search(
            pattern,
            text,
            re.IGNORECASE,
        )
        for pattern in document_patterns
    ):
        return True

    # --------------------------------------------------
    # Certificate / certification questions
    # --------------------------------------------------
    #
    # This fixes:
    #
    # "list my certificate"
    # "what certificates do I have?"
    # "which certifications are listed?"
    #

    certificate_patterns = [
        r"\bcertificate\b",
        r"\bcertificates\b",
        r"\bcertification\b",
        r"\bcertifications\b",
    ]

    has_certificate_term = any(
        re.search(
            pattern,
            text,
            re.IGNORECASE,
        )
        for pattern in certificate_patterns
    )

    certificate_context_patterns = [
        r"\bmy\b",
        r"\blist\b",
        r"\bshow\b",
        r"\bwhat\b",
        r"\bwhich\b",
        r"\bmentioned\b",
        r"\blisted\b",
        r"\bearned\b",
        r"\bcompleted\b",
        r"\bhave\b",
    ]

    has_certificate_context = any(
        re.search(
            pattern,
            text,
            re.IGNORECASE,
        )
        for pattern in certificate_context_patterns
    )

    if (
        has_certificate_term
        and has_certificate_context
    ):
        return True

    # --------------------------------------------------
    # Contextual follow-up
    # --------------------------------------------------

    if history:

        recent_text = _normalize(
            " ".join(
                str(
                    message.get(
                        "content",
                        "",
                    )
                )
                for message in history[-8:]
            )
        )

        document_words = [
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
            "education",
            "experience",
            "projects",
            "internship",
            "achievement",
        ]

        follow_up_patterns = [
            r"\bwhich one\b",
            r"\bwhich was\b",
            r"\bwho issued\b",
            r"\bwhen was\b",
            r"\bwhen did\b",
            r"\bwhat was the other\b",
            r"\bthat one\b",
            r"\bthis one\b",
            r"\bthe other one\b",
            r"\bthe previous\b",
            r"\bthe above\b",
            r"\bearlier\b",
            r"\bprevious answer\b",
            r"\bprevious question\b",
            r"\bmore about it\b",
            r"\btell me more\b",
            r"\bit\b",
            r"\bthey\b",
            r"\bthem\b",
        ]

        has_document_history = any(
            word in recent_text
            for word in document_words
        )

        has_follow_up = any(
            re.search(
                pattern,
                text,
                re.IGNORECASE,
            )
            for pattern in follow_up_patterns
        )

        if (
            has_document_history
            and has_follow_up
        ):
            return True

    return False


# ======================================================
# CODING DETECTION
# ======================================================

def _coding_request(
    question: str,
) -> bool:
    text = _normalize(
        question
    )

    coding_patterns = [
        r"\bwrite code\b",
        r"\bprovide code\b",
        r"\bgive me code\b",
        r"\bshow me code\b",
        r"\bcode for\b",
        r"\bimplement\b",
        r"\bimplementation\b",
        r"\bprogramming\b",
        r"\bcoding\b",
        r"\bdebug\b",
        r"\bdebugging\b",
        r"\bfix my code\b",
        r"\boptimize my code\b",
        r"\brefactor\b",
        r"\bdsa\b",
        r"\bdata structure\b",
        r"\bdata structures\b",
        r"\balgorithm\b",
        r"\bleetcode\b",
        r"\bdynamic programming\b",
        r"\bbinary search\b",
        r"\bdfs\b",
        r"\bbfs\b",
        r"\blinked list\b",
        r"\bstack\b",
        r"\bqueue\b",
        r"\btree\b",
        r"\btrie\b",
        r"\bheap\b",
        r"\bhashmap\b",
        r"\bhash map\b",
        r"\brecursion\b",
        r"\bbacktracking\b",
        r"\bsliding window\b",
        r"\btwo pointers\b",
        r"\btopological sort\b",
        r"\bshortest path\b",
        r"\bdijkstra\b",
        r"\bunion find\b",
        r"\bdisjoint set\b",
        r"\btime complexity\b",
        r"\bspace complexity\b",
        r"\bbig o\b",
        r"\bpython\b",
        r"\bjava\b",
        r"\bc\+\+\b",
        r"\bjavascript\b",
        r"\btypescript\b",
        r"\breact\b",
        r"\bnode\.?js\b",
        r"\bfastapi\b",
        r"\bspring boot\b",
        r"\bsql\b",
        r"\bjwt\b",
        r"\bmongodb\b",
        r"\bpostgresql\b",
        r"\bdocker\b",
    ]

    return any(
        re.search(
            pattern,
            text,
            re.IGNORECASE,
        )
        for pattern in coding_patterns
    )


# ======================================================
# QUIZ DETECTION
# ======================================================

def _quiz_request(
    question: str,
) -> bool:
    text = _normalize(
        question
    )

    patterns = [
        r"\bquiz\b",
        r"\bmcq\b",
        r"\bmcqs\b",
        r"\bmultiple choice\b",
        r"\bmultiple-choice\b",
        r"\bcreate.*questions\b",
        r"\bgenerate.*questions\b",
    ]

    return any(
        re.search(
            pattern,
            text,
            re.IGNORECASE,
        )
        for pattern in patterns
    )


# ======================================================
# SUMMARY DETECTION
# ======================================================

def _summary_request(
    question: str,
) -> bool:
    text = _normalize(
        question
    )

    patterns = [
        r"\bsummarize\b",
        r"\bsummarise\b",
        r"\bsummary\b",
        r"\bsummarize my resume\b",
        r"\bsummarise my resume\b",
    ]

    return any(
        re.search(
            pattern,
            text,
            re.IGNORECASE,
        )
        for pattern in patterns
    )


# ======================================================
# MAIN CLASSIFIER
# ======================================================

def classify_intent(
    question: str,
    history: list[dict] | None = None,
) -> str:

    if not question or not question.strip():
        return "GENERAL"

    text = _normalize(
        question
    )

    # --------------------------------------------------
    # CODING FIRST
    # --------------------------------------------------

    if _coding_request(
        text
    ):
        return "CODING"

    # --------------------------------------------------
    # DOCUMENT DETECTION
    # --------------------------------------------------

    is_document_question = (
        _document_reference(
            text,
            history,
        )
    )

    # --------------------------------------------------
    # QUIZ
    # --------------------------------------------------

    if (
        is_document_question
        and _quiz_request(
            text
        )
    ):
        return "QUIZ"

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    if (
        is_document_question
        and _summary_request(
            text
        )
    ):
        return "SUMMARY"

    # --------------------------------------------------
    # DOCUMENT QA
    # --------------------------------------------------

    if is_document_question:
        return "QA"

    # --------------------------------------------------
    # GENERAL
    # --------------------------------------------------

    return "GENERAL"
