from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src import __version__
from src.app import storage
from src.app.settings import (
    ALLOWED_FORMATS,
    COOKIE_MAX_AGE_SECONDS,
    MAX_UPLOAD_BYTES,
    UPLOAD_COOKIE,
)
from src.paths import STATIC_DIR, TEMPLATES_DIR, UPLOADS_DIR, load_dataset_config


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title="Детектирование заболеваний зерновых растений",
    version=__version__,
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

CLASSES = load_dataset_config()["classes"]


def render(
    request: Request,
    *,
    error: str | None = None,
    notice: str | None = None,
    status_code: int = 200,
) -> HTMLResponse:
    token = request.cookies.get(UPLOAD_COOKIE)
    image = storage.describe(token)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "image": image,
            "error": error,
            "notice": notice,
            "classes": CLASSES,
            "allowed_formats": sorted(ALLOWED_FORMATS),
            "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
            "human_size": storage.human_size,
            "version": __version__,
        },
        status_code=status_code,
    )


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return render(request)


@app.post("/upload")
async def upload(request: Request, photo: UploadFile = File(...)) -> Response:
    raw = await photo.read()

    try:
        stored = storage.save(raw, photo.filename or "снимок")
    except storage.UploadError as error:
        return render(request, error=str(error), status_code=400)

    storage.delete(request.cookies.get(UPLOAD_COOKIE))

    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(
        UPLOAD_COOKIE,
        stored.token,
        max_age=COOKIE_MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
    )
    return response


@app.post("/reset")
def reset(request: Request) -> Response:
    storage.delete(request.cookies.get(UPLOAD_COOKIE))
    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie(UPLOAD_COOKIE)
    return response


@app.get("/image")
def current_image(request: Request) -> Response:
    image = storage.describe(request.cookies.get(UPLOAD_COOKIE))
    if image is None:
        return Response(status_code=404)
    return FileResponse(
        image.path,
        media_type=image.media_type,
        headers={"Cache-Control": "no-store"},
    )


@app.post("/predict")
def predict(request: Request) -> HTMLResponse:
    image = storage.describe(request.cookies.get(UPLOAD_COOKIE))
    if image is None:
        return render(request, error="Сначала выберите изображение.", status_code=400)
    return render(
        request,
        notice="Модель подключается на этапе 6, пока распознавание недоступно.",
        status_code=501,
    )


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "version": __version__,
        "classes": sorted(CLASSES),
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "model_loaded": False,
    }
