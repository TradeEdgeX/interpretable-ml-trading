"""Court Lab HTTP app: /rd, /rd/qa, /browse. No auxiliary or CMS routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from src.lab import browse as browse_mod
from src.lab import experiments as rd_experiments
from src.lab.paths import experiments_root, repo_root, results_root, static_dir
from src.research.rd_qa import qa_payload

STATIC = static_dir()

app = FastAPI(
    title="MLBot Lab",
    description="Local court Lab: experiments, Q&A, results browse. No auxiliary trading.",
)
if STATIC.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


def ok(data: Any, *, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"ok": True, "data": data, "meta": meta or {}}


def _page(name: str) -> FileResponse:
    path = STATIC / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"missing page {name}")
    return FileResponse(path, media_type="text/html")


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/rd")


@app.get("/rd")
def rd_page() -> FileResponse:
    return _page("rd.html")


@app.get("/rd/qa")
def qa_page() -> FileResponse:
    return _page("qa.html")


@app.get("/browse")
def browse_page() -> FileResponse:
    return _page("browse.html")


@app.get("/api/rd/experiments")
def rd_experiments_list(
    strategy: Optional[str] = Query(None),
    q: Optional[str] = Query(None),
    since: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
) -> dict:
    rows = rd_experiments.list_experiments(
        strategy=strategy, q=q, since=since, category=category
    )
    return ok(
        rows,
        meta={
            "count": len(rows),
            "experiments_root": str(experiments_root()),
            "repo_root": str(repo_root()),
            "strategies": rd_experiments.list_strategies(),
        },
    )


@app.post("/api/rd/refresh")
def rd_refresh() -> dict:
    rd_experiments.clear_experiments_cache()
    rows = rd_experiments.list_experiments()
    return ok({"refreshed": True, "count": len(rows)})


@app.get("/api/rd/experiment/{experiment_id}")
def rd_experiment_detail(experiment_id: str) -> dict:
    row = rd_experiments.get_experiment(experiment_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"experiment not found: {experiment_id}")
    return ok(
        row,
        meta={
            "experiments_root": str(experiments_root()),
            "repo_root": str(repo_root()),
        },
    )


@app.get("/api/rd/qa")
def rd_qa() -> dict:
    return ok(qa_payload(repo_root()))


@app.get("/api/rd/experiment/{experiment_id}/raw/{filename}")
def rd_experiment_raw(experiment_id: str, filename: str) -> dict:
    payload = rd_experiments.get_experiment_raw_file(experiment_id, filename)
    if not payload:
        raise HTTPException(status_code=404, detail="file not found or not allowed")
    return ok(payload)


@app.get("/api/browse")
def browse_list(path: str = Query("", description="Relative path under results/")) -> dict:
    payload = browse_mod.list_browse(results_root(), path)
    if payload is None:
        raise HTTPException(status_code=404, detail="path not found")
    return ok(payload)


@app.get("/results/{file_path:path}")
def serve_results_file(file_path: str) -> FileResponse:
    listed = browse_mod.safe_browse_target(results_root(), file_path)
    if listed is None or not listed.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(listed)


def page_routes() -> frozenset[str]:
    """Declared Lab pages. Tests assert auxiliary / CMS paths are absent."""
    return frozenset({"/", "/rd", "/rd/qa", "/browse"})


def api_route_prefixes() -> frozenset[str]:
    return frozenset({"/api/rd", "/api/browse", "/results", "/static"})
