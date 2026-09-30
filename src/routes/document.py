from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from src.middleware.authmiddleware import (
    get_current_active_user,
)

from src.database.mongodb import (
    documents_collection,
)

from src.database.qdrant import (
    delete_document_vectors,
)


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


# ======================================================
# DELETE DOCUMENT
# ======================================================

@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    current_user=Depends(
        get_current_active_user
    ),
):
    # --------------------------------------------------
    # Get authenticated user ID
    # --------------------------------------------------

    if isinstance(
        current_user,
        dict,
    ):
        user_id = (
            current_user.get("_id")
            or current_user.get("id")
            or current_user.get("user_id")
        )

    else:
        user_id = (
            getattr(
                current_user,
                "_id",
                None,
            )
            or getattr(
                current_user,
                "id",
                None,
            )
            or getattr(
                current_user,
                "user_id",
                None,
            )
        )

    if user_id is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid authenticated user.",
        )

    user_id = str(user_id)

    # --------------------------------------------------
    # Check MongoDB document
    # --------------------------------------------------

    document = documents_collection.find_one(
        {
            "document_id": document_id,
            "user_id": user_id,
        }
    )

    if not document:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    # --------------------------------------------------
    # Delete Qdrant vectors
    # --------------------------------------------------

    try:
        delete_document_vectors(
            document_id=document_id,
            user_id=user_id,
        )

    except Exception as exc:

        print(
            "Qdrant document deletion failed:",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to delete document vectors "
                "from the knowledge base."
            ),
        ) from exc

    # --------------------------------------------------
    # Delete MongoDB document
    # --------------------------------------------------

    try:
        result = documents_collection.delete_one(
            {
                "document_id": document_id,
                "user_id": user_id,
            }
        )

    except Exception as exc:

        print(
            "MongoDB document deletion failed:",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Document vectors were deleted, "
                "but the document record could not "
                "be removed."
            ),
        ) from exc

    if result.deleted_count == 0:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    # --------------------------------------------------
    # Success
    # --------------------------------------------------

    return {
        "message": "Document deleted successfully.",
        "document_id": document_id,
    }
