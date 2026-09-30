import os

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from src.middleware.authmiddleware import (
    get_current_active_user,
)

from src.database.mongodb import db

from src.database.qdrant import (
    client,
    COLLECTION_NAME,
)

from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
)


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    current_user=Depends(
        get_current_active_user
    ),
):
    """
    Delete a user's document completely.

    Order:
    1. Resolve authenticated user
    2. Find MongoDB metadata
    3. Delete Qdrant vectors
    4. Delete physical PDF
    5. Delete MongoDB metadata
    """

    # ==================================================
    # 1. USER ID
    # ==================================================

    try:
        user_id = str(
            current_user["_id"]
        )
    except Exception as exc:
        print(
            "[DELETE] Failed to resolve user_id:",
            repr(exc),
        )

        raise HTTPException(
            status_code=401,
            detail="Invalid authenticated user.",
        ) from exc


    print(
        f"[DELETE] document_id={document_id}"
    )

    print(
        f"[DELETE] user_id={user_id}"
    )


    # ==================================================
    # 2. FIND DOCUMENT IN MONGODB
    # ==================================================

    try:
        document = db["documents"].find_one(
            {
                "document_id": document_id,
                "user_id": user_id,
            }
        )

    except Exception as exc:
        print(
            "[DELETE] MongoDB find failed:",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to access document metadata.",
        ) from exc


    if not document:
        print(
            "[DELETE] Document not found."
        )

        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )


    print(
        "[DELETE] MongoDB document found."
    )


    # ==================================================
    # 3. GET FILE PATH
    # ==================================================

    file_path = document.get(
        "file_path"
    )

    print(
        f"[DELETE] file_path={file_path}"
    )


    # ==================================================
    # 4. DELETE QDRANT VECTORS
    # ==================================================

    try:

        delete_filter = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(
                        value=document_id
                    ),
                ),
                FieldCondition(
                    key="user_id",
                    match=MatchValue(
                        value=user_id
                    ),
                ),
            ]
        )


        print(
            "[DELETE] Deleting Qdrant vectors..."
        )


        qdrant_result = client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=delete_filter,
            wait=True,
        )


        print(
            "[DELETE] Qdrant delete completed:",
            qdrant_result,
        )

    except Exception as exc:

        print(
            "[DELETE] Qdrant delete failed:",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to delete document "
                "vectors from Qdrant."
            ),
        ) from exc


    # ==================================================
    # 5. DELETE PHYSICAL PDF
    # ==================================================

    if file_path:

        try:

            if os.path.exists(
                file_path
            ):

                os.remove(
                    file_path
                )

                print(
                    "[DELETE] PDF file deleted."
                )

            else:

                print(
                    "[DELETE] PDF file already missing."
                )

        except Exception as exc:

            print(
                "[DELETE] PDF deletion failed:",
                repr(exc),
            )

            raise HTTPException(
                status_code=500,
                detail=(
                    "Qdrant vectors were deleted, "
                    "but the PDF file could not be removed."
                ),
            ) from exc


    # ==================================================
    # 6. DELETE MONGODB METADATA
    # ==================================================

    try:

        delete_result = (
            db["documents"].delete_one(
                {
                    "document_id": document_id,
                    "user_id": user_id,
                }
            )
        )


        print(
            "[DELETE] MongoDB delete count:",
            delete_result.deleted_count,
        )

    except Exception as exc:

        print(
            "[DELETE] MongoDB delete failed:",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Qdrant and file deletion succeeded, "
                "but MongoDB metadata deletion failed."
            ),
        ) from exc


    if (
        delete_result.deleted_count
        == 0
    ):

        raise HTTPException(
            status_code=404,
            detail="Document metadata not found.",
        )


    # ==================================================
    # 7. SUCCESS
    # ==================================================

    return {
        "message": "Document deleted successfully",
        "document_id": document_id,
    }
