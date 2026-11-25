from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src import __version__
from src.app import storage
from src.app.confidence import confidence_band
from src.app.settings import (
    ALLOWED_FORMATS,
    COOKIE_MAX_AGE_SECONDS,
    CROP_COOKIE,
    DEFAULT_CROP,
    LOW_CONFIDENCE_THRESHOLD,
    MAX_UPLOAD_BYTES,
    UNKNOWN_THRESHOLD,
    UPLOAD_COOKIE,
)
from src.model.crop_gate import check_crop, load as load_crop_gate
from src.model.predict import Prediction, load_checkpoint
from src.model.predict import predict as run_model
from src.paths import (
    CROP_CHECKPOINTS,
    CROP_DATASET_CONFIGS,
    CROP_GENITIVE,
    CROP_TITLES,
    STAGE4_CHECKPOINT,
    STATIC_DIR,
    TEMPLATES_DIR,
    UPLOADS_DIR,
    load_dataset_config,
    relative_to_root,
)

CROP_ORDER = tuple(CROP_TITLES)
CROP_CLASSES = {
    crop: load_dataset_config(CROP_DATASET_CONFIGS[crop])["classes"] for crop in CROP_ORDER
}
CLASSES = CROP_CLASSES[DEFAULT_CROP]
MODEL_STATE: dict[str, object] = {"loaded": False, "crops": {}}


def warm_up_model() -> None:
    try:
        load_crop_gate()
    except FileNotFoundError:
        pass
    loaded: dict[str, bool] = {}
    for crop, path in CROP_CHECKPOINTS.items():
        try:
            load_checkpoint(path)
        except FileNotFoundError:
            loaded[crop] = False
        else:
            loaded[crop] = True
    MODEL_STATE["crops"] = loaded
    MODEL_STATE["loaded"] = bool(loaded.get(DEFAULT_CROP))


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


def normalize_crop(value: str | None) -> str:
    crop = (value or "").strip().lower()
    return crop if crop in CROP_CHECKPOINTS else DEFAULT_CROP


def read_crop(request: Request, form_crop: str | None = None) -> str:
    return normalize_crop(form_crop or request.cookies.get(CROP_COOKIE) or DEFAULT_CROP)


def set_crop_cookie(response: Response, crop: str) -> None:
    response.set_cookie(
        CROP_COOKIE,
        crop,
        max_age=COOKIE_MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
    )


def crop_catalog(selected: str) -> list[dict]:
    crops_loaded = MODEL_STATE.get("crops") or {}
    return [
        {
            "id": crop,
            "ru": CROP_TITLES[crop],
            "selected": crop == selected,
            "available": bool(crops_loaded.get(crop, CROP_CHECKPOINTS[crop].exists())),
            "classes": CROP_CLASSES[crop],
        }
        for crop in CROP_ORDER
    ]


def render(
    request: Request,
    *,
    token: str | None = None,
    crop: str | None = None,
    error: str | None = None,
    notice: str | None = None,
    status_code: int = 200,
) -> HTMLResponse:
    token = token or request.cookies.get(UPLOAD_COOKIE)
    crop = read_crop(request, crop)
    image = storage.describe(token)
    stored = storage.load_prediction(token) if image else None
    if stored and normalize_crop(stored.get("crop")) != crop:
        stored = None
    classes = CROP_CLASSES[crop]
    result = describe_prediction(stored, classes) if stored else None
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "image": image,
            "error": error,
            "notice": notice,
            "result": result,
            "crop": crop,
            "crop_title": CROP_TITLES[crop],
            "crop_genitive": CROP_GENITIVE[crop],
            "crops": crop_catalog(crop),
            "classes": classes,
            "allowed_formats": sorted(ALLOWED_FORMATS),
            "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
            "low_confidence_percent": int(LOW_CONFIDENCE_THRESHOLD * 100),
            "unknown_percent": int(UNKNOWN_THRESHOLD * 100),
            "human_size": storage.human_size,
            "version": __version__,
        },
        status_code=status_code,
    )


def describe_prediction(stored: dict, classes: dict) -> dict:
    ranked = [
        {
            "class_name": name,
            "class_ru": classes[name]["ru"],
            "symptoms": classes[name].get("symptoms", ""),
            "probability": value,
        }
        for name, value in sorted(stored["probabilities"].items(), key=lambda item: -item[1])
        if name in classes
    ]
    if not ranked:
        return None
    best = ranked[0]
    band = confidence_band(best["probability"])
    return {
        "class_name": best["class_name"],
        "class_ru": best["class_ru"],
        "confidence": best["probability"],
        "band": band,
        "is_healthy": band == "confident" and best["class_name"] == "healthy",
        "low_confidence": band == "low",
        "unknown": band == "unknown",
        "runner_up": ranked[1] if len(ranked) > 1 else None,
        "compare": ranked[:1] if band == "confident" else ranked[:2],
        "ranked": ranked,
    }


def recognise(token: str, image: storage.StoredImage, crop: str) -> tuple[str, int] | None:
    try:
        mismatch = check_crop(image.path, crop)
    except FileNotFoundError:
        return "Проверка культуры недоступна: не найден файл весов. Обратитесь к администратору.", 503
    if mismatch:
        storage.clear_prediction(token)
        return mismatch, 422
    weights = CROP_CHECKPOINTS[crop]
    kwargs = {} if crop == DEFAULT_CROP else {"weights_path": weights}
    try:
        prediction: Prediction = run_model(image.path, **kwargs)
    except FileNotFoundError:
        crops_loaded = dict(MODEL_STATE.get("crops") or {})
        crops_loaded[crop] = False
        MODEL_STATE["crops"] = crops_loaded
        if crop == DEFAULT_CROP:
            MODEL_STATE["loaded"] = False
        return "Модель недоступна: не найден файл весов. Обратитесь к администратору.", 503
    except (OSError, ValueError):
        return "Не удалось обработать изображение моделью. Попробуйте другую фотографию.", 422
    crops_loaded = dict(MODEL_STATE.get("crops") or {})
    crops_loaded[crop] = True
    MODEL_STATE["crops"] = crops_loaded
    if crop == DEFAULT_CROP:
        MODEL_STATE["loaded"] = True
    storage.save_prediction(
        token,
        {
            "crop": crop,
            "class_name": prediction.class_name,
            "confidence": prediction.confidence,
            "probabilities": prediction.probabilities,
        },
    )
    return None


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return render(request)


@app.post("/crop")
def set_crop(request: Request, crop: str | None = Form(None)) -> Response:
    crop = read_crop(request, crop)
    storage.clear_prediction(request.cookies.get(UPLOAD_COOKIE))
    response = RedirectResponse(url="/", status_code=303)
    set_crop_cookie(response, crop)
    return response


@app.post("/upload")
async def upload(
    request: Request,
    photo: UploadFile = File(...),
    then: str = Form(""),
    crop: str | None = Form(None),
) -> Response:
    crop = read_crop(request, crop)
    raw = await photo.read()

    try:
        stored = storage.save(raw, photo.filename or "снимок")
    except storage.UploadError as error:
        response = render(request, crop=crop, error=str(error), status_code=400)
        set_crop_cookie(response, crop)
        return response

    storage.delete(request.cookies.get(UPLOAD_COOKIE))

    failure = recognise(stored.token, stored, crop) if then == "predict" else None
    if failure:
        message, status_code = failure
        response = render(
            request,
            token=stored.token,
            crop=crop,
            error=message,
            status_code=status_code,
        )
    else:
        response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(
        UPLOAD_COOKIE,
        stored.token,
        max_age=COOKIE_MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
    )
    set_crop_cookie(response, crop)
    return response


@app.post("/reset")
def reset(request: Request) -> Response:
    storage.delete(request.cookies.get(UPLOAD_COOKIE))
    crop = read_crop(request)
    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie(UPLOAD_COOKIE)
    set_crop_cookie(response, crop)
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
def predict(request: Request, crop: str | None = Form(None)) -> Response:
    crop = read_crop(request, crop)
    token = request.cookies.get(UPLOAD_COOKIE)
    image = storage.describe(token)
    if image is None:
        response = render(request, crop=crop, error="Сначала выберите изображение.", status_code=400)
        set_crop_cookie(response, crop)
        return response
    failure = recognise(str(token), image, crop)
    if failure:
        message, status_code = failure
        response = render(request, crop=crop, error=message, status_code=status_code)
        set_crop_cookie(response, crop)
        return response
    response = RedirectResponse(url="/", status_code=303)
    set_crop_cookie(response, crop)
    return response


@app.get("/health")
def health() -> dict[str, object]:
    crops_loaded = MODEL_STATE.get("crops") or {}
    return {
        "status": "ok",
        "version": __version__,
        "classes": sorted(CLASSES),
        "crops": {
            crop: {
                "title": CROP_TITLES[crop],
                "classes": list(CROP_CLASSES[crop]),
                "loaded": bool(crops_loaded.get(crop, False)),
                "weights": relative_to_root(CROP_CHECKPOINTS[crop]),
            }
            for crop in CROP_ORDER
        },
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "model_loaded": MODEL_STATE["loaded"],
        "model_weights": relative_to_root(STAGE4_CHECKPOINT),
    }
