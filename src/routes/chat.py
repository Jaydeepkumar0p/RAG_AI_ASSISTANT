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
# USER ID HELPER
# ======================================================

def _get_user_id(
    current_user,
) -> str:
    if isinstance(
        current_user,
        dict,
    ):
        value = (
            current_user.get("_id")
            or current_user.get("id")
            or current_user.get("user_id")
        )

    else:
        value = (
            getattr(
                current_user,
                "_id",
                None,
            )
            or
            getattr(
                current_user,
                "id",
                None,
            )
            or
            getattr(
                current_user,
                "user_id",
                None,
            )
        )

    if value is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid authenticated user.",
        )

    return str(value)


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

    user_id = _get_user_id(
        current_user
    )

    # --------------------------------------------------
    # Conversation
    # --------------------------------------------------

    if not conversation_id:
        conversation_id = str(
            create_conversation(
                user_id
            )
        )

    else:
        conversation = get_conversation(
            conversation_id=conversation_id,
            user_id=user_id,
        )

        if not conversation:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found.",
            )

    # --------------------------------------------------
    # AI processing
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
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        print(
            "Chat processing error:",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to process the question.",
        ) from exc

    # --------------------------------------------------
    # Response
    # --------------------------------------------------

    return {
        "conversation_id": conversation_id,
        **result,
    }
