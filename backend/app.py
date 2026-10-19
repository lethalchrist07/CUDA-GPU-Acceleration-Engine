import re
from pathlib import Path
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.data import (
    ArtifactStore,
    collect_capabilities,
    collect_hardware,
    utc_now,
)
from backend.serving import ModelServer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LoadModelRequest(BaseModel):
    model_name: str = "ResNet18"
    checkpoint: Optional[str] = None
    device: Optional[str] = "auto"


class PredictRequest(BaseModel):
    sample_index: Optional[int] = None
    tensor: Optional[List[Any]] = None


def create_app(root: Path = PROJECT_ROOT) -> FastAPI:
    app = FastAPI(
        title="CUDA Acceleration Engine",
        version="0.9.0",
        description="FastAPI service for GPU benchmarks, AI training records, telemetry, and live ResNet-18 model serving.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "http://127.0.0.1:4173",
            "http://localhost:4173",
        ],
        allow_methods=["GET", "POST"],
        allow_headers=["Accept", "Content-Type"],
    )

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

    store = ArtifactStore(root)

    # ---------------------------------------------------------
    # Core Historical & Telemetry Endpoints (Preserved 100%)
    # ---------------------------------------------------------
    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": app.version, "time": utc_now(), "mode": "read-write"}

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
    def checkpoints():
        return store.checkpoints_catalog()

    @app.get("/api/checkpoints/{checkpoint_id}")
    def checkpoint_detail(checkpoint_id: str):
        catalog = store.checkpoints_catalog()
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
            result = ModelServer.load_model(
                model_name=req.model_name,
                checkpoint_path=req.checkpoint,
                device_str=req.device,
            )
            return result
        except FileNotFoundError as fnf:
            raise HTTPException(status_code=404, detail=str(fnf))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/models/unload")
    def unload_model():
        try:
            return ModelServer.unload_model()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/models/predict")
    def predict(req: PredictRequest):
        try:
            return ModelServer.predict(sample_index=req.sample_index, tensor_data=req.tensor)
        except FileNotFoundError as fnf:
            raise HTTPException(status_code=404, detail=str(fnf))
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

    return app


app = create_app()
