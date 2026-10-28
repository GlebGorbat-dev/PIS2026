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
    LOW_CONFIDENCE_THRESHOLD,
    MAX_UPLOAD_BYTES,
    UPLOAD_COOKIE,
)
from src.model.predict import Prediction, load_checkpoint
from src.model.predict import predict as run_model
from src.paths import (
    STAGE4_CHECKPOINT,
    STATIC_DIR,
    TEMPLATES_DIR,
    UPLOADS_DIR,
    load_dataset_config,
    relative_to_root,
)

MODEL_STATE = {"loaded": False}


def warm_up_model() -> None:
    try:
        load_checkpoint()
    except FileNotFoundError:
        MODEL_STATE["loaded"] = False
    else:
        MODEL_STATE["loaded"] = True


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    warm_up_model()
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
    result: dict | None = None,
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
            "result": result,
            "classes": CLASSES,
            "allowed_formats": sorted(ALLOWED_FORMATS),
            "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
            "human_size": storage.human_size,
            "version": __version__,
        },
        status_code=status_code,
    )


def describe_prediction(prediction: Prediction) -> dict:
    ranked = [
        {"class_name": name, "class_ru": CLASSES[name]["ru"], "probability": value}
        for name, value in sorted(prediction.probabilities.items(), key=lambda item: -item[1])
    ]
    return {
        "class_name": prediction.class_name,
        "class_ru": prediction.class_ru,
        "confidence": prediction.confidence,
        "low_confidence": prediction.confidence < LOW_CONFIDENCE_THRESHOLD,
        "runner_up": ranked[1] if len(ranked) > 1 else None,
        "ranked": ranked,
    }


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
    try:
        prediction = run_model(image.path)
    except FileNotFoundError:
        MODEL_STATE["loaded"] = False
        return render(
            request,
            error="Модель недоступна: не найден файл весов. Обратитесь к администратору.",
            status_code=503,
        )
    except (OSError, ValueError):
        return render(
            request,
            error="Не удалось обработать изображение моделью. Попробуйте другую фотографию.",
            status_code=422,
        )
    MODEL_STATE["loaded"] = True
    return render(request, result=describe_prediction(prediction))


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "version": __version__,
        "classes": sorted(CLASSES),
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "model_loaded": MODEL_STATE["loaded"],
        "model_weights": relative_to_root(STAGE4_CHECKPOINT),
    }
