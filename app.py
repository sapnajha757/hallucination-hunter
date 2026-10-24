"""FastAPI application entrypoint for AI Hallucination Hunter.

Provides health checks, API documentation, and the primary /hunt endpoint with
comprehensive exception mapping and dependency injection.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ai_engine import (
    AIEngine,
    LLMConfigurationError,
    LLMProviderError,
    LLMResponseMalformedError,
    LLMTimeoutError,
)
from config import Settings, get_settings
from detector import HallucinationDetector, InputValidationError
from evidence import EvidenceValidationError
from schemas import (
    ErrorResponse,
    HealthResponse,
    HuntRequest,
    HuntResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager for startup and shutdown hooks."""
    # Startup validation
    settings = get_settings()
    app.state.settings = settings
    app.state.ai_engine = AIEngine(settings=settings)
    app.state.detector = HallucinationDetector(ai_engine=app.state.ai_engine, settings=settings)
    yield
    # Shutdown cleanup if needed


app = FastAPI(
    title="AI Hallucination Hunter API",
    description=(
        "A factual hallucination detection service that decomposes AI-generated text into "
        "atomic claims and verifies each claim against trusted reference evidence."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan,
)


def get_detector(request: Request) -> HallucinationDetector:
    """Dependency provider for HallucinationDetector."""
    return getattr(request.app.state, "detector", None) or HallucinationDetector(
        ai_engine=AIEngine(settings=get_settings()), settings=get_settings()
    )


# Exception Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle FastAPI / Pydantic schema validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=ErrorResponse(
            error="Validation Error",
            detail=jsonable_encoder(exc.errors()),
        ).model_dump(),
    )


@app.exception_handler(EvidenceValidationError)
async def evidence_validation_exception_handler(
    request: Request, exc: EvidenceValidationError
) -> JSONResponse:
    """Handle reference evidence format, uniqueness, and constraint errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=ErrorResponse(
            error="Invalid Evidence",
            detail=str(exc),
        ).model_dump(),
    )


@app.exception_handler(InputValidationError)
async def input_validation_exception_handler(
    request: Request, exc: InputValidationError
) -> JSONResponse:
    """Handle resource limit and payload length violations."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="Input Limit Exceeded",
            detail=str(exc),
        ).model_dump(),
    )


@app.exception_handler(LLMConfigurationError)
async def llm_config_exception_handler(
    request: Request, exc: LLMConfigurationError
) -> JSONResponse:
    """Handle unconfigured LLM provider errors."""
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(
            error="LLM Provider Not Configured",
            detail=str(exc),
        ).model_dump(),
    )


@app.exception_handler(LLMTimeoutError)
async def llm_timeout_exception_handler(
    request: Request, exc: LLMTimeoutError
) -> JSONResponse:
    """Handle upstream LLM timeouts."""
    return JSONResponse(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        content=ErrorResponse(
            error="LLM Provider Timeout",
            detail=str(exc),
        ).model_dump(),
    )


@app.exception_handler(LLMProviderError)
async def llm_provider_exception_handler(
    request: Request, exc: LLMProviderError
) -> JSONResponse:
    """Handle upstream provider API and connection errors."""
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content=ErrorResponse(
            error="LLM Provider Error",
            detail=str(exc),
        ).model_dump(),
    )


@app.exception_handler(LLMResponseMalformedError)
async def llm_malformed_exception_handler(
    request: Request, exc: LLMResponseMalformedError
) -> JSONResponse:
    """Handle invalid JSON or schema violations returned by the model."""
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content=ErrorResponse(
            error="Malformed Model Output",
            detail=str(exc),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unhandled internal server errors without leaking internals."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal Server Error",
            detail="An unexpected internal server error occurred.",
        ).model_dump(),
    )


# Endpoints
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check and provider status",
    tags=["System"],
)
async def health_check(
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Return service status and verify whether the LLM provider has an API key configured."""
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.app_version,
        llm_provider_configured=bool(settings.llm_api_key),
        model=settings.llm_model,
    )


@app.post(
    "/hunt",
    response_model=HuntResponse,
    responses={
        200: {"model": HuntResponse, "description": "Successful hallucination evaluation"},
        400: {"model": ErrorResponse, "description": "Input limit exceeded or invalid parameter"},
        422: {"model": ErrorResponse, "description": "Validation error or invalid evidence format"},
        502: {"model": ErrorResponse, "description": "LLM provider error or malformed output"},
        503: {"model": ErrorResponse, "description": "LLM provider is not configured"},
        504: {"model": ErrorResponse, "description": "LLM provider request timed out"},
    },
    summary="Evaluate candidate answer against reference evidence",
    tags=["Hallucination Detection"],
)
async def hunt_hallucinations(
    request: HuntRequest,
    detector: HallucinationDetector = Depends(get_detector),
) -> HuntResponse:
    """Decompose candidate answer into factual claims and evaluate against reference evidence."""
    return await detector.hunt(request)
