"""FastAPI 入口：上传、轮询任务、查看结果、下载复现库。"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .errors import ImpactError, JobError, ScriptRejected, ValidationError
from .impact import analyze_query, normalize_targets, validated_temp_db
from .jobs import registry
from .samples import ensure_samples
from .service import run_pipeline

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    ensure_samples(STATIC_DIR)
    registry.cleanup_expired()

    async def _janitor() -> None:
        while True:
            await asyncio.sleep(60)
            registry.cleanup_expired()

    janitor = asyncio.create_task(_janitor())
    try:
        yield
    finally:
        janitor.cancel()


app = FastAPI(title="SQLite 复现库缩减服务", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/samples")
def list_samples() -> JSONResponse:
    return JSONResponse(ensure_samples(STATIC_DIR))


def _read_db_upload(raw: bytes) -> bytes:
    if len(raw) == 0:
        raise ValidationError("数据库文件为空。")
    if len(raw) > config.MAX_DB_BYTES:
        raise ValidationError(
            f"数据库大小 {len(raw)} 字节，超过上限 {config.MAX_DB_BYTES} 字节。"
        )
    if not raw.startswith(b"SQLite format 3\x00"):
        raise ValidationError("文件不是合法的 SQLite 数据库（魔数校验失败）。")
    return raw


@app.post("/api/jobs")
async def create_job(
    db: UploadFile = File(..., description="满足约束的 SQLite 数据库"),
    sql: UploadFile | None = File(None, description="确定性 SQL 脚本文件"),
    script_text: str = Form("", description="或直接粘贴 SQL 脚本"),
) -> JSONResponse:
    raw_db = await db.read()
    _read_db_upload(raw_db)

    if sql is not None:
        raw_script = await sql.read()
    else:
        raw_script = script_text.encode("utf-8")
    if not raw_script.strip():
        raise ValidationError("请上传或粘贴 SQL 脚本。")

    job = registry.create()
    upload_path = job.dir / "upload.db"
    upload_path.write_bytes(raw_db)
    # 原件落盘后只读保存，任何试跑都只接触副本
    Path(upload_path).chmod(0o444)

    registry.run_in_background(job, run_pipeline, str(upload_path), raw_script)
    return JSONResponse({"id": job.id, "status": job.status}, status_code=202)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    job = registry.get(job_id)
    return JSONResponse(job.public_dict())


@app.get("/api/jobs/{job_id}/download")
def download_reduced(job_id: str) -> FileResponse:
    job = registry.get(job_id)
    if job.status != "done" or not job.result.get("download_available"):
        raise JobError("缩减结果尚不可下载。")
    reduced = job.dir / "reduced.db"
    if not reduced.exists():
        raise JobError("缩减库文件已被清理。")
    return FileResponse(
        reduced,
        media_type="application/x-sqlite3",
        filename="repro.db",
    )


@app.post("/api/impact/schema")
async def impact_schema(db: UploadFile = File(...)) -> JSONResponse:
    """读取上传库的表结构（列名清单），用于勾选待下线列。"""
    raw = await db.read()
    _read_db_upload(raw)
    with validated_temp_db(raw) as (_schema, columns):
        return JSONResponse(
            {"tables": [{"name": t, "columns": c} for t, c in columns.items()]}
        )


@app.post("/api/impact")
async def impact_check(
    db: UploadFile = File(...),
    query: str = Form(""),
    targets: list[str] | None = Form(None),
) -> JSONResponse:
    """静态核查单条 SELECT 对目标列的依赖，不执行 SQL。"""
    raw = await db.read()
    _read_db_upload(raw)
    if not query.strip():
        raise ImpactError("请粘贴一条 SELECT 查询。")
    if len(query.encode("utf-8")) > config.MAX_SCRIPT_BYTES:
        raise ImpactError(
            f"查询文本大小超过上限 {config.MAX_SCRIPT_BYTES} 字节。"
        )
    if not targets:
        raise ImpactError("请至少选择一个待下线的源表列。")
    wanted: list[tuple[str, str]] = []
    for item in targets:
        table, sep, column = item.partition(".")
        if not sep or not table.strip() or not column.strip():
            raise ImpactError(f"目标列格式应为“表.列”：{item!r}。")
        wanted.append((table.strip(), column.strip()))
    with validated_temp_db(raw) as (_schema, columns):
        norm_targets = normalize_targets(wanted, columns)
        report = analyze_query(query, columns, norm_targets)
    return JSONResponse(report)


@app.exception_handler(ValidationError)
@app.exception_handler(ScriptRejected)
@app.exception_handler(JobError)
@app.exception_handler(ImpactError)
def _handle_bad_request(_request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})
