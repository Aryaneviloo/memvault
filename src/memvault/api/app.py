import time

from fastapi import FastAPI, Request, Response

from memvault.api.routes import router
from memvault.observability.logging import get_logger, setup_logging
from memvault.observability.metrics import get_metrics

VERSION = "0.1.0"
logger = get_logger(__name__)


def create_app() -> FastAPI:
    """
    Application factory: creates and configures the FASTAPI app
    Using a factory function (rather than module level app) makes it easier
    to test — tests can call create_app() to get a fresh instance with
    test dependencies injected.

    """
    
    setup_logging()

    app = FastAPI(
        title="MemVault API",
        description="Open-source memory infrastructure for AI agents",
        version=VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next) -> Response:
        """
        LOG every request with method, path, status, amd latency

        """
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start) * 1000

        metrics = get_metrics()
        metrics.record_request(latency_ms)

        logger.info(
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_ms=round(latency_ms, 2),
        )
        return response

    app.include_router(router)

    return app


app = create_app()
