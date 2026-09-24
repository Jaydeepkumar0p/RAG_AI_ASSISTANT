import os
import uuid

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Depends,
    HTTPException
)

from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue
)

from src.middleware.authmiddleware import get_current_active_user
from src.database.mongodb import db
from src.database.qdrant import client, COLLECTION_NAME
from src.services.rag_service import process_pdf


router = APIRouter(
    prefix="/documents",
    tags=["Documents"]
)


UPLOAD_DIR = "uploads"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


# ======================================================
# UPLOAD PDF
# ======================================================

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    current_user=Depends(get_current_active_user)
):

    # --------------------------------------------------
    # Validate PDF
    # --------------------------------------------------

    filename = file.filename or ""

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    # --------------------------------------------------
    # Generate document ID
    # --------------------------------------------------

    document_id = str(uuid.uuid4())

    # --------------------------------------------------
    # Save PDF
    # --------------------------------------------------

    file_path = os.path.join(
        UPLOAD_DIR,
        f"{document_id}.pdf"
    )

    content = await file.read()

    with open(file_path, "wb") as f:
        f.write(content)

    # --------------------------------------------------
    # Process PDF
    # --------------------------------------------------

    try:

        result = process_pdf(
            file_path=file_path,
            user_id=str(current_user["_id"]),
            document_id=document_id,
            filename=filename
        )

    except Exception as e:

        if os.path.exists(file_path):
            os.remove(file_path)

        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {str(e)}"
        )

    # --------------------------------------------------
    # Save metadata in MongoDB
    # --------------------------------------------------

    db["documents"].insert_one({
        "document_id": document_id,
        "user_id": str(current_user["_id"]),
        "filename": filename,
        "file_path": file_path,
        "chunk_count": result["chunks"],
        "status": "processed"
    })

    return {
        "message": "Document processed successfully",
        "document_id": document_id,
        "filename": filename,
        "chunks": result["chunks"]
    }


# ======================================================
# GET ALL USER DOCUMENTS
# ======================================================

@router.get("")
async def get_documents(
    current_user=Depends(get_current_active_user)
):

    user_id = str(current_user["_id"])

    documents = db["documents"].find({
        "user_id": user_id
    })

    result = []

    for document in documents:

        result.append({
            "document_id": document["document_id"],
            "filename": document["filename"],
            "chunk_count": document.get(
                "chunk_count",
                0
            ),
            "status": document.get(
                "status",
                "unknown"
            )
        })

    return {
        "documents": result
    }


# ======================================================
# QDRANT HEALTH
# ======================================================

@router.get("/qdrant/health")
async def qdrant_health():

    try:

        collections = client.get_collections()

        return {
            "status": "connected",
            "collections": [
                collection.name
                for collection in collections.collections
            ]
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Qdrant connection failed: {str(e)}"
        )


# ======================================================
# QDRANT POINTS
# ======================================================

@router.get("/qdrant/points")
async def qdrant_points():

    try:

        points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=3,
            with_payload=True,
            with_vectors=False
        )

        return {
            "collection": COLLECTION_NAME,
            "points": [
                {
                    "id": str(point.id),
                    "payload": point.payload
                }
                for point in points
            ]
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Unable to read Qdrant points: {str(e)}"
        )


# ======================================================
# QDRANT COUNT
# ======================================================

@router.get("/qdrant/count")
async def qdrant_count():

    try:

        result = client.count(
            collection_name=COLLECTION_NAME,
            exact=True
        )

        return {
            "collection": COLLECTION_NAME,
            "points_count": result.count
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Unable to count Qdrant points: {str(e)}"
        )


# ======================================================
# GET SINGLE DOCUMENT
# ======================================================

@router.get("/{document_id}")
async def get_document(
    document_id: str,
    current_user=Depends(get_current_active_user)
):

    user_id = str(current_user["_id"])

    document = db["documents"].find_one({
        "document_id": document_id,
        "user_id": user_id
    })

    if not document:

        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    return {
        "document_id": document["document_id"],
        "filename": document["filename"],
        "file_path": document["file_path"],
        "chunk_count": document.get(
            "chunk_count",
            0
        ),
        "status": document.get(
            "status",
            "unknown"
        )
    }


# ======================================================
# DELETE DOCUMENT
# ======================================================

@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    current_user=Depends(get_current_active_user)
):

    user_id = str(current_user["_id"])

    # --------------------------------------------------
    # Find user's document
    # --------------------------------------------------

    document = db["documents"].find_one({
        "document_id": document_id,
        "user_id": user_id
    })

    if not document:

        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    # --------------------------------------------------
    # Delete Qdrant vectors
    # --------------------------------------------------

    try:

        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(
                            value=document_id
                        )
                    ),
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(
                            value=user_id
                        )
                    )
                ]
            )
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete vectors: {str(e)}"
        )

    # --------------------------------------------------
    # Delete PDF file
    # --------------------------------------------------

    file_path = document.get("file_path")

    if file_path and os.path.exists(file_path):

        os.remove(file_path)

    # --------------------------------------------------
    # Delete MongoDB metadata
    # --------------------------------------------------

    db["documents"].delete_one({
        "document_id": document_id,
        "user_id": user_id
    })

    return {
        "message": "Document deleted successfully",
        "document_id": document_id
    }