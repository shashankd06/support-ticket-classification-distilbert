import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from .inference import Predictor


class Ticket(BaseModel):
    text: str = Field(min_length=1, max_length=4000, strict=True)

    @field_validator("text")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Ticket must contain non-whitespace text")
        return value


class Prediction(BaseModel):
    label: str
    score: float = Field(ge=0, le=1)
    model: str
    mode: str
    truncated: bool


def create_app(predictor_factory=None):
    @asynccontextmanager
    async def lifespan(app):
        # Once per server process, never per request. Missing artifacts fail startup.
        app.state.predictor = (
            predictor_factory()
            if predictor_factory
            else Predictor(
                os.getenv("MODEL_DIR", "artifacts/full"),
                os.getenv("MODEL_BACKEND", "distilbert"),
            )
        )
        yield

    app = FastAPI(
        title="Support Ticket Intent Classifier",
        lifespan=lifespan,
        description="Scores are model probabilities, not calibrated confidence. English banking intents only.",
    )

    @app.post("/predict", response_model=Prediction)
    def predict(ticket: Ticket):
        return app.state.predictor(ticket.text)

    @app.get("/health")
    def health():
        return {"status": "ready"}

    return app


app = create_app()
