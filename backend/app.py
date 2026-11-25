from contextlib import asynccontextmanager
import logging
import re
from pathlib import Path
from typing import Any, List, Optional

import torch
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import get_settings, setup_logging, log_hardware_diagnostic_banner
from backend.data import (
    ArtifactStore,
    collect_capabilities,
    collect_hardware,
    utc_now,
)
from backend.routes.training import router as training_router
from backend.routes.laboratory import router as laboratory_router
from backend.serving import ModelServer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("cuda_engine.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: configures logging & hardware banners on startup, graceful shutdown on exit."""
    setup_logging()
    log_hardware_diagnostic_banner()
    yield
    logger.info("Initiating graceful shutdown of CUDA Acceleration Engine...")
    try:
        from engine.training.orchestrator import TrainingOrchestrator
        TrainingOrchestrator.get_instance().shutdown()
    except Exception as e:
        logger.warning(f"Error shutting down training orchestrator: {e}")
    try:
        ModelServer.unload_model("all")
    except Exception as e:
        logger.warning(f"Error unloading serving models: {e}")
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    logger.info("CUDA Acceleration Engine shutdown completed cleanly.")


class LoadModelRequest(BaseModel):
    model_name: str = "ResNet18"
    checkpoint: Optional[str] = None
    device: Optional[str] = "auto"


class PredictRequest(BaseModel):
    sample_index: Optional[int] = None
    tensor: Optional[List[Any]] = None


class GenerateTextRequest(BaseModel):
    prompt: str = ""
    max_new_tokens: int = 32
    temperature: float = 0.8
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.95
    seed: Optional[int] = 42
    checkpoint: Optional[str] = None
    device: Optional[str] = "auto"
    model_name: Optional[str] = None


class RequestSizeLimitMiddleware:
    """
    Pure ASGI middleware enforcing 32 KB request size limit before body parsing in public demo mode.
    Protects against memory exhaustion from oversized payloads and unbounded chunked transfer streams.
    """

    def __init__(self, app, max_bytes: int = 32768):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            settings = get_settings()
            if settings.demo_mode:
                method = scope.get("method", "").upper()
                if method in ("POST", "PUT", "PATCH"):
                    headers = dict(scope.get("headers", []))
                    cl_header = headers.get(b"content-length")
                    if cl_header:
                        try:
                            cl_val = int(cl_header.decode("latin1"))
                            if cl_val > self.max_bytes:
                                response = JSONResponse(
                                    status_code=413,
                                    content={
                                        "status": "error",
                                        "error_type": "PAYLOAD_TOO_LARGE",
                                        "detail": f"Request payload exceeds maximum allowed size of {self.max_bytes} bytes in public demo mode.",
                                    },
                                )
                                await response(scope, receive, send)
                                return
                        except ValueError:
                            pass

                    # Guard against chunked streams exceeding limit
                    total_bytes = 0

                    async def bounded_receive():
                        nonlocal total_bytes
                        message = await receive()
                        if message["type"] == "http.request":
                            body = message.get("body", b"")
                            total_bytes += len(body)
                            if total_bytes > self.max_bytes:
                                class StreamPayloadTooLarge(Exception):
                                    pass
                                raise StreamPayloadTooLarge()
                        return message

                    try:
                        await self.app(scope, bounded_receive, send)
                        return
                    except Exception as exc:
                        if exc.__class__.__name__ == "StreamPayloadTooLarge":
                            response = JSONResponse(
                                status_code=413,
                                content={
                                    "status": "error",
                                    "error_type": "PAYLOAD_TOO_LARGE",
                                    "detail": f"Request stream payload exceeds maximum allowed size of {self.max_bytes} bytes in public demo mode.",
                                },
                            )
                            await response(scope, receive, send)
                            return
                        raise exc

        await self.app(scope, receive, send)


def create_app(root: Path = PROJECT_ROOT) -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="CUDA Acceleration Engine",
        version="1.0.0",
        description="FastAPI service for GPU benchmarks, AI training records, telemetry, and live ResNet-18 model serving.",
        lifespan=lifespan,
    )
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=settings.demo_max_request_bytes)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Accept", "Content-Type", "Authorization"],
    )

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

    # Global Exception Handlers
    @app.exception_handler(torch.cuda.OutOfMemoryError)
    async def cuda_oom_handler(request: Request, exc: torch.cuda.OutOfMemoryError):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.critical(f"CUDA OutOfMemory on request {request.url.path}: {exc}")
        return JSONResponse(
            status_code=507,
            content={
                "status": "error",
                "error_type": "CUDA_OUT_OF_MEMORY",
                "detail": "CUDA Out of Memory: Batch memory requirements exceeded 6GB VRAM on RTX 3050 Laptop GPU. Memory cache was evacuated. Reduce batch size or context length.",
            },
        )

    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(request: Request, exc: FileNotFoundError):
        logger.warning(f"Resource not found on {request.url.path}: {exc}")
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "error_type": "FILE_NOT_FOUND",
                "detail": str(exc),
            },
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        logger.warning(f"Invalid value on {request.url.path}: {exc}")
        return JSONResponse(
            status_code=422,
            content={
                "status": "error",
                "error_type": "VALUE_ERROR",
                "detail": str(exc),
            },
        )

    store = ArtifactStore(root)

    # Mount Training Orchestration Router (Phase 10 Milestone 5)
    app.include_router(training_router)

    # Mount Comparative Training Laboratory Router (Phase 13 Milestone 7)
    app.include_router(laboratory_router)

    # ---------------------------------------------------------
    # Core Historical & Telemetry Endpoints (Preserved 100%)
    # ---------------------------------------------------------
    @app.get("/api/health")
    def health():
        mode = "public-demo" if settings.demo_mode else "read-write"
        return {"status": "ok", "version": app.version, "time": utc_now(), "mode": mode}

    @app.get("/api/overview")
    def overview():
        return store.overview()

    @app.get("/api/benchmarks")
    def benchmarks():
        return store.benchmarks()

    @app.get("/api/training")
    def training():
        return store.experiments("training")

    @app.get("/api/inference")
    def inference():
        return store.experiments("inference")

    @app.get("/api/hardware")
    def hardware():
        return collect_hardware()

    @app.get("/api/verification")
    def verification():
        return store.verification()

    @app.get("/api/artifacts/{artifact_id}")
    def artifact(artifact_id: str):
        if not re.match(r"^[a-zA-Z0-9_-]+$", artifact_id):
            raise HTTPException(status_code=404, detail="Invalid artifact identifier")
        path = store.artifact_paths().get(artifact_id)
        if path is None:
            raise HTTPException(status_code=404, detail="Unknown artifact")
        return FileResponse(path, filename=path.name, media_type="application/octet-stream")

    # ---------------------------------------------------------
    # Phase 9: Model Registry, Checkpoints, & Serving Endpoints
    # ---------------------------------------------------------
    @app.get("/api/capabilities")
    def capabilities():
        return collect_capabilities()

    @app.get("/api/experiments")
    def experiments():
        training = store.phase9_experiments()
        benchmarks = store.phase9_benchmarks()
        return {
            "status": "available",
            "fetched_at": utc_now(),
            "phase9_training": training,
            "phase9_benchmarks": benchmarks,
        }

    @app.get("/api/checkpoints")
    def checkpoints(force_rehash: bool = False):
        return store.checkpoints_catalog(force_rehash=force_rehash)

    @app.get("/api/checkpoints/{checkpoint_id}")
    def checkpoint_detail(checkpoint_id: str, force_rehash: bool = False):
        catalog = store.checkpoints_catalog(force_rehash=force_rehash)
        for c in catalog.get("checkpoints", []):
            if c["id"] == checkpoint_id:
                return {"status": "available", "checkpoint": c}
        raise HTTPException(status_code=404, detail=f"Checkpoint '{checkpoint_id}' not found.")

    @app.get("/api/models")
    def list_models():
        return {"status": "available", "fetched_at": utc_now(), "models": ModelServer.list_available_models()}

    @app.post("/api/models/load")
    def load_model(req: LoadModelRequest):
        try:
            if settings.demo_mode:
                if req.checkpoint is not None and str(req.checkpoint).strip() != "":
                    raise HTTPException(
                        status_code=403,
                        detail="Arbitrary checkpoint paths are forbidden in demo mode. Only approved public model IDs are permitted.",
                    )
                norm = req.model_name.strip().lower()
                is_resnet = norm in ("resnet18_cifar10", "resnet18", "resnet-18")
                is_trans = norm in ("transformer_50m", "nexora gpt-86.8m", "transformer", "gpt", "nexora") and not ("100m" in norm or "pilot" in norm)
                is_mlp = norm in ("baseline_mlp", "baselinemlp", "mlp")
                if not (is_resnet or is_trans or is_mlp):
                    raise HTTPException(
                        status_code=403,
                        detail=f"Model '{req.model_name}' is not in the approved public demo whitelist. Permitted models: resnet18_cifar10, transformer_50m, baseline_mlp.",
                    )
            result = ModelServer.load_model(
                model_name=req.model_name,
                checkpoint_path=req.checkpoint,
                device_str=req.device,
            )
            return result
        except HTTPException:
            raise
        except FileNotFoundError as fnf:
            raise HTTPException(status_code=404, detail=str(fnf))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/models/unload")
    def unload_model():
        try:
            if settings.demo_mode:
                raise HTTPException(status_code=403, detail="Model unloading is forbidden in public demo mode.")
            return ModelServer.unload_model()
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/models/predict")
    def predict(req: PredictRequest):
        try:
            if settings.demo_mode and req.tensor is not None:
                def _count_floats(obj):
                    if isinstance(obj, (list, tuple)):
                        return sum(_count_floats(x) for x in obj)
                    return 1
                if _count_floats(req.tensor) > settings.demo_max_tensor_floats:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Input tensor exceeds maximum allowed {settings.demo_max_tensor_floats} floats in demo mode.",
                    )
            return ModelServer.predict(sample_index=req.sample_index, tensor_data=req.tensor)
        except HTTPException:
            raise
        except FileNotFoundError as fnf:
            raise HTTPException(status_code=404, detail=str(fnf))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/models/generate")
    def generate_text(req: GenerateTextRequest):
        try:
            if settings.demo_mode:
                if req.checkpoint is not None and str(req.checkpoint).strip() != "":
                    raise HTTPException(
                        status_code=403,
                        detail="Arbitrary checkpoint paths are forbidden in demo mode.",
                    )
                # Clamp generation parameters in demo mode
                req.max_new_tokens = min(max(1, req.max_new_tokens), settings.demo_max_gen_tokens)
                if req.prompt and len(req.prompt) > settings.demo_max_prompt_chars:
                    req.prompt = req.prompt[:settings.demo_max_prompt_chars]
            elif req.checkpoint:
                resolved_model_name = req.model_name or ("NEXORA GPT-86.8M" if "50m" in str(req.checkpoint) else "NEXORA GPT-100M")
                ModelServer.load_model(
                    model_name=resolved_model_name,
                    checkpoint_path=req.checkpoint,
                    device_str=req.device,
                )
            return ModelServer.generate(
                prompt=req.prompt,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                seed=req.seed,
                device_str=req.device,
            )
        except HTTPException:
            raise
        except FileNotFoundError as fnf:
            raise HTTPException(status_code=404, detail=str(fnf))
        except RuntimeError as re:
            if "actively reserved by training job" in str(re):
                raise HTTPException(status_code=409, detail=str(re))
            raise HTTPException(status_code=400, detail=str(re))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/models/evaluation")
    def evaluate_model():
        benchmarks = store.phase9_benchmarks()
        experiments = store.phase9_experiments()
        return {
            "status": "available",
            "fetched_at": utc_now(),
            "model_name": "ResNet18",
            "dataset": "CIFAR-10",
            "benchmarks": benchmarks.get("training_benchmarks", []),
            "inference_benchmarks": benchmarks.get("inference_benchmarks", []),
            "training_runs": experiments.get("rows", []),
        }

    # ---------------------------------------------------------
    # Single-Server Deployment & SPA Static Asset Serving (BLK-03)
    # ---------------------------------------------------------
    dist_dir = (root / "frontend" / "dist").resolve()
    assets_dir = dist_dir / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/", include_in_schema=False)
    def serve_root():
        index_file = dist_dir / "index.html"
        if index_file.is_file():
            return FileResponse(index_file)
        raise HTTPException(
            status_code=404,
            detail="Frontend static build not found. Please build the frontend via 'npm run build' inside frontend/.",
        )

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str):
        # Protect API endpoints: unmatched /api/... routes must return JSON 404, not HTML
        if full_path == "api" or full_path.startswith("api/"):
            raise HTTPException(
                status_code=404,
                detail=f"API endpoint '/{full_path}' not found",
            )

        # Direct static file in dist root (e.g. favicon, robots.txt, vite.svg)
        if dist_dir.is_dir():
            candidate = (dist_dir / full_path).resolve()
            if candidate.is_file() and str(candidate).startswith(str(dist_dir)):
                return FileResponse(candidate)

            index_path = dist_dir / "index.html"
            if index_path.is_file():
                return FileResponse(index_path)

        raise HTTPException(
            status_code=404,
            detail="Frontend static build not found. Please build the frontend via 'npm run build' inside frontend/.",
        )

    return app


app = create_app()
