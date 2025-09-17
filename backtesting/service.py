"""FastAPI application scaffolding for the backtesting service."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .config import BacktestRunConfig
from .metadata import load_backtest_metadata
from .model_runner import BacktestRunner


class BacktestRequest(BaseModel):
    dataset_path: Path
    metadata_path: Path
    model_path: Optional[Path] = None
    batch_size: int = 1


def _build_run_config(payload: BacktestRequest) -> BacktestRunConfig:
    metadata = load_backtest_metadata(payload.metadata_path)
    return BacktestRunConfig.from_metadata(
        dataset_path=payload.dataset_path,
        metadata_path=payload.metadata_path,
        metadata=metadata,
        model_path=payload.model_path,
        batch_size=payload.batch_size,
    )


def create_app() -> FastAPI:
    app = FastAPI(title="Feature Genesis Backtesting API", version="0.1.0")

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.post("/backtest/run")
    async def run_backtest(request: BacktestRequest) -> JSONResponse:
        try:
            run_cfg = _build_run_config(request)
            runner = BacktestRunner(run_cfg)
            events: List[dict] = await asyncio.to_thread(lambda: list(runner.iterate_events()))
            return JSONResponse({"events": events, "metrics": runner.metrics, "trades": runner.trade_log})
        except Exception as exc:  # pragma: no cover - surfaced to client
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.websocket("/backtest")
    async def backtest_socket(websocket: WebSocket):
        await websocket.accept()
        try:
            initial = await websocket.receive_text()
            try:
                payload = json.loads(initial)
            except json.JSONDecodeError as exc:  # pragma: no cover - client error
                await websocket.send_json({"type": "error", "message": f"Invalid JSON: {exc}"})
                await websocket.close()
                return
            request = BacktestRequest(**payload)
            run_cfg = _build_run_config(request)
            runner = BacktestRunner(run_cfg)
            iterator = runner.iterate_events()
            while True:
                try:
                    event = await asyncio.to_thread(next, iterator)
                except StopIteration:
                    break
                await websocket.send_json(event)
        except WebSocketDisconnect:
            return
        except Exception as exc:  # pragma: no cover - surfaced to client
            await websocket.send_json({"type": "error", "message": str(exc)})
        finally:
            await websocket.close()

    return app
