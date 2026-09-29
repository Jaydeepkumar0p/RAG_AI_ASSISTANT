import re


VALID_INTENTS = {
    "QA",
    "SUMMARY",
    "QUIZ",
    "CODING",
    "GENERAL",
}


def _normalize(text: str) -> str:
    return " ".join(
        (text or "").strip().lower().split()
    )


def _document_reference(
    question: str,
    history: list[dict] | None = None,
) -> bool:
    text = _normalize(question)

    explicit_patterns = [
        r"\bmy resume\b",
        r"\bmy cv\b",
        r"\bmy document\b",
        r"\bmy documents\b",
        r"\bmy pdf\b",
        r"\buploaded document\b",
        r"\buploaded pdf\b",
        r"\buploaded file\b",
        r"\bin my resume\b",
        r"\bin my cv\b",
        r"\bin the resume\b",
        r"\bin the document\b",
        r"\bfrom my resume\b",
        r"\bfrom my cv\b",
        r"\bfrom my document\b",
        r"\baccording to my resume\b",
        r"\baccording to my document\b",
        r"\bmentioned in my resume\b",
        r"\blisted in my resume\b",
        r"\bmy internship\b",
        r"\bmy skills\b",
        r"\bmy education\b",
        r"\bmy experience\b",
        r"\bmy projects\b",
        r"\bmy certification\b",
        r"\bmy certifications\b",
    ]

    if any(
        re.search(pattern, text, re.IGNORECASE)
        for pattern in explicit_patterns
    ):
        return True

    if history:
        recent = _normalize(
            " ".join(
                str(
                    message.get(
                        "content",
                        "",
                    )
                )
                for message in history[-6:]
            )
        )

        document_words = [
            "resume",
            "cv",
            "certification",
            "internship",
            "document",
            "pdf",
            "uploaded",
            "education",
            "experience",
            "projects",
            "skills",
        ]

        follow_up_phrases = [
            "which one",
            "who issued it",
            "what was the other",
            "that one",
            "this one",
            "previous",
            "above",
            "earlier",
            "it",
            "they",
            "them",
        ]

        has_document_history = any(
            word in recent
            for word in document_words
        )

        has_follow_up = any(
            phrase in text
            for phrase in follow_up_phrases
        )

        if (
            has_document_history
            and has_follow_up
        ):
            return True

    return False


def _coding_request(question: str) -> bool:
    text = _normalize(question)

    patterns = [
        r"\bwrite code\b",
        r"\bprovide code\b",
        r"\bgive me code\b",
        r"\bshow me code\b",
        r"\bcode for\b",
        r"\bimplement\b",
        r"\bimplementation\b",
        r"\bsolve\b",
        r"\bsolution\b",
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
        r"\bneetcode\b",
        r"\bcompetitive programming\b",
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
        r"\bcsharp\b",
        r"\bc#\b",
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
        r"\bkubernetes\b",
    ]

    return any(
        re.search(
            pattern,
            text,
            re.IGNORECASE,
        )
        for pattern in patterns
    )


def _quiz_request(question: str) -> bool:
    text = _normalize(question)

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


def _summary_request(question: str) -> bool:
    text = _normalize(question)

    patterns = [
        r"\bsummarize\b",
        r"\bsummarise\b",
        r"\bsummary\b",
    ]

    return any(
        re.search(
            pattern,
            text,
            re.IGNORECASE,
        )
        for pattern in patterns
    )


def classify_intent(
    question: str,
    history: list[dict] | None = None,
) -> str:
    if not question or not question.strip():
        return "GENERAL"

    document_reference = _document_reference(
        question,
        history,
    )

    if _coding_request(question):
        return "CODING"

    if (
        document_reference
        and _quiz_request(question)
    ):
        return "QUIZ"

    if (
        document_reference
        and _summary_request(question)
    ):
        return "SUMMARY"

    if document_reference:
        return "QA"

    return "GENERAL"
