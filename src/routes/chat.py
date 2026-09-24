from fastapi import (
    APIRouter,
    Depends,
    HTTPException
)

from fastapi.responses import (
    StreamingResponse
)

from src.middleware.authmiddleware import (
    get_current_active_user
)

from src.services.qa_service import (
    answer_question
)

from src.services.stream_service import (
    stream_chat
)

from src.services.conversation_service import (
    create_conversation,
    get_conversation
)


router = APIRouter(
    prefix="/chat",
    tags=["Chat"]
)


# ======================================================
# NORMAL CHAT
# ======================================================

@router.post("")
async def chat(

    question: str,

    conversation_id: str | None = None,

    current_user=Depends(
        get_current_active_user
    )
):

    # --------------------------------------------------
    # Get authenticated user
    # --------------------------------------------------

    user_id = str(
        current_user["_id"]
    )

    # --------------------------------------------------
    # Validate question
    # --------------------------------------------------

    question = question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    # --------------------------------------------------
    # Create conversation
    # --------------------------------------------------

    if not conversation_id:

        conversation_id = create_conversation(
            user_id=user_id
        )

    # --------------------------------------------------
    # Verify existing conversation
    # --------------------------------------------------

    else:

        conversation = get_conversation(
            conversation_id=conversation_id,
            user_id=user_id
        )

        if not conversation:

            raise HTTPException(
                status_code=404,
                detail="Conversation not found"
            )

    # --------------------------------------------------
    # Run Agentic RAG
    # --------------------------------------------------

    result = answer_question(

        question=question,

        user_id=user_id,

        conversation_id=conversation_id
    )

    # --------------------------------------------------
    # Response
    # --------------------------------------------------

    return {

        "conversation_id":
            conversation_id,

        "question":
            result["question"],

        "intent":
            result.get(
                "intent",
                "QA"
            ),

        "rewritten_query":
            result.get(
                "rewritten_query",
                ""
            ),

        "retrieval_relevant":
            result.get(
                "retrieval_relevant",
                False
            ),

        "answer":
            result.get(
                "answer",
                ""
            ),

        "sources":
            result.get(
                "sources",
                []
            )
    }


# ======================================================
# STREAM CHAT
# ======================================================

@router.post("/stream")
async def stream_chat_endpoint(

    question: str,

    conversation_id: str | None = None,

    current_user=Depends(
        get_current_active_user
    )
):

    # --------------------------------------------------
    # Get authenticated user
    # --------------------------------------------------

    user_id = str(
        current_user["_id"]
    )

    # --------------------------------------------------
    # Validate question
    # --------------------------------------------------

    question = question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    # --------------------------------------------------
    # Create conversation
    # --------------------------------------------------

    if not conversation_id:

        conversation_id = create_conversation(
            user_id=user_id
        )

    # --------------------------------------------------
    # Verify existing conversation
    # --------------------------------------------------

    else:

        conversation = get_conversation(
            conversation_id=conversation_id,
            user_id=user_id
        )

        if not conversation:

            raise HTTPException(
                status_code=404,
                detail="Conversation not found"
            )

    # --------------------------------------------------
    # Streaming response
    # --------------------------------------------------

    return StreamingResponse(

        stream_chat(

            question=question,

            user_id=user_id,

            conversation_id=conversation_id
        ),

        media_type="text/event-stream",

        headers={

            "Cache-Control":
                "no-cache",

            "X-Accel-Buffering":
                "no",

            "Connection":
                "keep-alive"
        }
    )