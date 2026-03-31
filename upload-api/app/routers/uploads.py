import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import ALLOWED_EXT, MAX_BYTES, UPLOAD_DIR
from app.schemas.uploads import UploadRecord
from app.services.dataframe_loader import try_preview
from app.services.upload_store import load_index, resolve_path, save_index

router = APIRouter(prefix="/api", tags=["uploads"])


@router.get("/uploads", response_model=list[UploadRecord])
def list_uploads():
    items = load_index()
    return [UploadRecord.model_validate(x) for x in items]


@router.post("/upload", response_model=UploadRecord)
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported type. Allowed: {', '.join(sorted(ALLOWED_EXT))}",
        )

    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 50 MB)")

    file_id = str(uuid.uuid4())
    safe_name = f"{file_id}{suffix}"
    out_path = UPLOAD_DIR / safe_name
    out_path.write_bytes(data)

    preview = try_preview(out_path, suffix)
    created = datetime.now(timezone.utc).isoformat()

    record = {
        "id": file_id,
        "original_name": file.filename,
        "stored_path": safe_name,
        "size_bytes": len(data),
        "created_at": created,
        "preview": preview,
    }

    items = load_index()
    items.insert(0, record)
    save_index(items)

    return UploadRecord.model_validate(record)


@router.delete("/uploads/{file_id}")
def delete_upload(file_id: str):
    items = load_index()
    found = next((i for i, x in enumerate(items) if x["id"] == file_id), None)
    if found is None:
        raise HTTPException(status_code=404, detail="Not found")

    rec = items.pop(found)
    path = resolve_path(rec["stored_path"])
    if path.exists():
        path.unlink()
    save_index(items)
    return {"deleted": file_id}
