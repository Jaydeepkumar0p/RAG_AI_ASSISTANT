# ======================================================
# CHAT ROUTE
# ======================================================

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from src.middleware.authmiddleware import (
    get_current_active_user,
)

from src.services.conversation_service import (
    create_conversation,
    get_conversation,
)

from src.services.qa_service import (
    answer_question,
)


# ======================================================
# ROUTER
# ======================================================

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


# ======================================================
# POST /chat
# ======================================================

@router.post("")
async def chat(
    question: str,
    conversation_id: str | None = None,
    current_user=Depends(
        get_current_active_user
    ),
):
    # --------------------------------------------------
    # Validate question
    # --------------------------------------------------

    question = question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    # --------------------------------------------------
    # Authenticated user
    # --------------------------------------------------

    user_id = str(
        current_user["_id"]
    )

    # --------------------------------------------------
    # Conversation
    # --------------------------------------------------

    if not conversation_id:
        conversation_id = create_conversation(
            user_id
        )

    else:
        conversation = get_conversation(
            conversation_id=conversation_id,
            user_id=user_id,
        )

        if not conversation:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found",
            )

    # --------------------------------------------------
    # Run AI assistant
    # --------------------------------------------------

    try:
        result = answer_question(
            question=question,
            user_id=user_id,
            conversation_id=conversation_id,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        print(
            "Chat processing error:",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to process the question.",
        )

    # --------------------------------------------------
    # Response
    # --------------------------------------------------

    return {
        "conversation_id": conversation_id,
        **result,
    }
