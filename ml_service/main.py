from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover - joblib is provided in production image
    joblib = None  # type: ignore

import pickle

app = FastAPI(title="ML Service", version="1.0.0")

MODEL_PATH = Path(__file__).parent / "models" / "model.joblib"
_model = None


class SimpleBaselineModel:
    """Deterministic fallback when no trained artifact is supplied."""

    def predict(self, features):
        # Expect shape: List[List[float]]
        processed = []
        for row in features:
            if not row:
                processed.append(0.0)
                continue
            processed.append(sum(row) / len(row))
        return processed


class PredictPayload(BaseModel):
    features: List[float] = Field(..., min_length=1, description="Ordered feature values")


def _load_with_pickle(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def load_model():
    """Load the serialized model from disk, caching it in memory."""
    global _model
    if _model is not None:
        return _model

    if not MODEL_PATH.exists():
        _model = SimpleBaselineModel()
        return _model

    if joblib is not None:
        try:
            _model = joblib.load(MODEL_PATH)
            return _model
        except Exception:
            # Fall back to pickle loader if joblib cannot interpret the file
            pass

    _model = _load_with_pickle(MODEL_PATH)
    return _model


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictPayload) -> dict[str, float]:
    model = load_model()

    # Accept either sklearn-style estimators or simple callables/classes with predict
    if hasattr(model, "predict"):
        try:
            prediction = model.predict([payload.features])[0]
        except Exception as exc:  # pragma: no cover - prediction errors surface to client
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    elif callable(model):
        try:
            prediction = model(payload.features)
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    else:  # pragma: no cover
        raise HTTPException(status_code=500, detail="Loaded model is not callable")

    # Ensure JSON serializable return
    if isinstance(prediction, (list, tuple)):
        if not prediction:
            raise HTTPException(status_code=500, detail="Empty prediction returned")
        value = float(prediction[0])
    else:
        value = float(prediction)

    return {"prediction": value}
