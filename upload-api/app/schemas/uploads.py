from pydantic import BaseModel, ConfigDict


class Preview(BaseModel):
    model_config = ConfigDict(extra="ignore")

    row_count: int
    column_count: int
    columns: list[str]
    dtypes: dict[str, str]
    null_counts: dict[str, int]
    sample_rows: list[dict]


class UploadRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    original_name: str
    stored_path: str
    size_bytes: int
    created_at: str
    preview: Preview | None = None
