"""FastAPI interface for planning retrosynthetic routes with AiZynthFinder.

The service keeps one loaded finder per algorithm and serializes calls for each
finder.  The underlying finder mutates its search state for every plan, so this
keeps model loading inexpensive while avoiding cross-request state leakage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
import yaml

from aizynthfinder.aizynthfinder import AiZynthFinder
from aizynthfinder.context.stock import StockQueryMixin
from aizynthfinder.utils.exceptions import MoleculeException, PolicyException

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8001
DEFAULT_BUILDING_BLOCKS = os.environ.get("AIZYNTHFINDER_BUILDING_BLOCKS")
DISK_JOB_PROTOCOL = 1
DISK_JOB_ROOT = os.environ.get("AIZYNTHFINDER_JOB_ROOT")

# A child of ``uvicorn.error`` inherits Uvicorn's handlers and INFO level, so
# resource-loading reports are visible in the FastAPI server console.
_LOGGER = logging.getLogger("uvicorn.error.aizynthfinder")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = _PROJECT_ROOT / "contrib" / "data"
_CONFIG_PATHS = {
    "mcts": Path(
        os.environ.get("AIZYNTHFINDER_MCTS_CONFIG", _DEFAULT_DATA_DIR / "config.yml")
    ).expanduser(),
    "retrostar": Path(
        os.environ.get(
            "AIZYNTHFINDER_RETROSTAR_CONFIG",
            _DEFAULT_DATA_DIR / "config_retro_star.yml",
        )
    ).expanduser(),
}
_MODEL_SELECTIONS = {
    "uspto": ["uspto"],
    "ringbreaker": ["ringbreaker"],
    "reaxys": ["reaxys"],
    "multi": ["multi_expansion_strategy"],
}
_SEARCH_SETUPS = {
    "retrostar_uspto_zinc": {
        "algorithm": "retrostar",
        "model": "uspto",
        "stocks": ["zinc"],
    },
    "retrostar_reaxys_zinc": {
        "algorithm": "retrostar",
        "model": "reaxys",
        "stocks": ["zinc"],
    },
}


class PlanRequest(BaseModel):
    """Input accepted by ``POST /aizynthfinder_plan``."""

    smiles: str = Field(..., min_length=1, max_length=1000)
    algorithm: Literal["mcts", "retrostar", "original"] = "mcts"
    model: Literal["uspto", "ringbreaker", "reaxys", "multi"] = "uspto"
    iterations: int = Field(100, ge=1, le=10_000)
    time_limit: float = Field(120, gt=0, le=3_600)
    expansion_topk: int = Field(50, ge=1, le=500)
    max_transforms: int = Field(6, ge=1, le=20)
    max_routes: int = Field(25, ge=1, le=100)
    return_first: bool = False
    use_filter: bool = False
    exclude_target_from_stock: bool = True
    stocks: list[Literal["zinc", "uspto_stock"]] | None = None


class DiskJobSubmission(BaseModel):
    """A shared directory containing an asynchronous request.json file."""

    job_dir: str = Field(..., min_length=1)


class PlannerInputError(ValueError):
    """A request value that cannot be used to create a planning tree."""


class InMemorySmilesCsvQuery(StockQueryMixin):
    """Exact in-memory stock lookup backed by canonical SMILES in a CSV."""

    def __init__(self, path: str | Path, smiles_column: str = "smiles") -> None:
        stock_path = Path(path).expanduser()
        if not stock_path.is_file():
            raise FileNotFoundError(f"Building-block CSV was not found: {stock_path}")

        with stock_path.open(encoding="utf-8", newline="") as fileobj:
            reader = csv.DictReader(fileobj)
            if reader.fieldnames is None or smiles_column not in reader.fieldnames:
                raise ValueError(
                    f"Building-block CSV must contain a {smiles_column!r} column: "
                    f"{stock_path}"
                )
            self._stock_smiles = {
                value
                for row in reader
                if (value := (row.get(smiles_column) or "").strip())
            }

        _LOGGER.info(
            "Loaded %d canonical SMILES from %s",
            len(self._stock_smiles),
            stock_path,
        )

    def __contains__(self, mol: Any) -> bool:
        mol.sanitize(raise_exception=False)
        return mol.smiles in self._stock_smiles

    def __len__(self) -> int:
        return len(self._stock_smiles)


@dataclass
class _FinderSlot:
    finder: AiZynthFinder
    lock: threading.Lock
    configured_for: _PlanConfiguration | None = None


@dataclass(frozen=True)
class _PlanConfiguration:
    """Request settings that affect a finder, excluding the target SMILES."""

    model: str
    iterations: int
    time_limit: float
    expansion_topk: int
    max_transforms: int
    max_routes: int
    return_first: bool
    use_filter: bool
    exclude_target_from_stock: bool
    stocks: tuple[str, ...] | None

    @classmethod
    def from_request(cls, request: PlanRequest) -> _PlanConfiguration:
        """Create a stable comparison key for the mutable finder settings."""
        return cls(
            model=request.model,
            iterations=request.iterations,
            time_limit=request.time_limit,
            expansion_topk=request.expansion_topk,
            max_transforms=request.max_transforms,
            max_routes=request.max_routes,
            return_first=request.return_first,
            use_filter=request.use_filter,
            exclude_target_from_stock=request.exclude_target_from_stock,
            # An empty list has the same meaning as ``None`` in ``_configure``.
            stocks=tuple(request.stocks) if request.stocks else None,
        )


class AiZynthFinderPlanner:
    """Lazy model loader and request-safe facade around :class:`AiZynthFinder`."""

    def __init__(self, config_paths: dict[str, Path] | None = None) -> None:
        self._config_paths = config_paths or _CONFIG_PATHS
        self._slots: dict[str, _FinderSlot] = {}
        self._slots_lock = threading.Lock()
        self._building_block_query: InMemorySmilesCsvQuery | None = None

    @property
    def loaded_algorithms(self) -> list[str]:
        """Return algorithms whose finder and models have been initialized."""
        with self._slots_lock:
            return sorted(self._slots)

    def plan(self, request: PlanRequest) -> dict[str, Any]:
        """Run one request using a cached finder for the requested algorithm."""
        algorithm = self._resolve_algorithm(request.algorithm)
        slot = self._get_slot(algorithm)
        configuration = _PlanConfiguration.from_request(request)
        with slot.lock:
            if slot.configured_for != configuration:
                self._configure(slot.finder, request)
                slot.configured_for = configuration
            else:
                _LOGGER.debug(
                    "Reusing the loaded %s finder; only the target may have changed",
                    algorithm,
                )
            return self._run_plan(slot.finder, request, algorithm)

    def _get_slot(self, algorithm: str) -> _FinderSlot:
        with self._slots_lock:
            if algorithm in self._slots:
                return self._slots[algorithm]

            config_path = self._config_paths[algorithm]
            if not config_path.is_file():
                raise RuntimeError(
                    f"Configuration for '{algorithm}' was not found: {config_path}"
                )

            started_at = time.perf_counter()
            stock_source = DEFAULT_BUILDING_BLOCKS or "configuration file"
            _LOGGER.info(
                "Loading algorithm models and purchasable stock: "
                "algorithm=%s config=%s stock_source=%s reason=cache_miss",
                algorithm,
                config_path,
                stock_source,
            )
            try:
                if DEFAULT_BUILDING_BLOCKS:
                    with config_path.open(encoding="utf-8") as fileobj:
                        config = yaml.safe_load(fileobj)
                    config["stock"] = {}
                    finder = AiZynthFinder(configdict=config)
                    finder.stock.load(
                        self._get_building_block_query(),
                        "zinc",
                    )
                else:
                    finder = AiZynthFinder(configfile=str(config_path))
            except Exception:
                _LOGGER.exception(
                    "Failed to load algorithm models and purchasable stock: "
                    "algorithm=%s config=%s elapsed_seconds=%.3f",
                    algorithm,
                    config_path,
                    time.perf_counter() - started_at,
                )
                raise

            model_names = getattr(
                getattr(finder, "expansion_policy", None), "items", []
            )
            stock_names = getattr(getattr(finder, "stock", None), "items", [])
            _LOGGER.info(
                "Loaded algorithm models and purchasable stock: "
                "algorithm=%s models=%s stocks=%s elapsed_seconds=%.3f",
                algorithm,
                model_names,
                stock_names,
                time.perf_counter() - started_at,
            )

            slot = _FinderSlot(finder=finder, lock=threading.Lock())
            self._slots[algorithm] = slot
            return slot

    def _get_building_block_query(self) -> InMemorySmilesCsvQuery:
        """Load the configured ZINC CSV once and share it between algorithms."""
        if not DEFAULT_BUILDING_BLOCKS:
            raise RuntimeError("AIZYNTHFINDER_BUILDING_BLOCKS is not configured")
        # This method is called while ``_slots_lock`` is held.  Sharing the
        # immutable query prevents MCTS and Retro* from each retaining a second
        # copy of the multi-million-entry SMILES set.
        if self._building_block_query is None:
            self._building_block_query = InMemorySmilesCsvQuery(DEFAULT_BUILDING_BLOCKS)
        return self._building_block_query

    @staticmethod
    def _resolve_algorithm(requested_algorithm: str) -> str:
        # ``original`` is a readable alias for AiZynthFinder's default MCTS setup.
        return "mcts" if requested_algorithm == "original" else requested_algorithm

    @staticmethod
    def _configure(finder: AiZynthFinder, request: PlanRequest) -> None:
        model_keys = _MODEL_SELECTIONS[request.model]
        finder.expansion_policy.select(model_keys)
        if request.use_filter:
            finder.filter_policy.select_all()
        else:
            finder.filter_policy.deselect()

        if request.stocks:
            finder.stock.select(request.stocks)
        else:
            finder.stock.select_all()

        finder.config.search.iteration_limit = request.iterations
        finder.config.search.time_limit = request.time_limit
        finder.config.search.max_transforms = request.max_transforms
        finder.config.search.return_first = request.return_first
        finder.config.search.exclude_target_from_stock = (
            request.exclude_target_from_stock
        )
        finder.config.post_processing.min_routes = min(5, request.max_routes)
        finder.config.post_processing.max_routes = request.max_routes
        finder.config.post_processing.all_routes = False

        # Template and multi-expansion strategies both expose ``cutoff_number``.
        # Updating every loaded strategy makes the request setting effective when
        # a direct model or the combined model is selected.
        for key in finder.expansion_policy.items:
            strategy = finder.expansion_policy[key]
            if hasattr(strategy, "cutoff_number"):
                strategy.cutoff_number = request.expansion_topk

    @staticmethod
    def _run_plan(
        finder: AiZynthFinder,
        request: PlanRequest,
        algorithm: str,
    ) -> dict[str, Any]:
        finder.target_smiles = request.smiles.strip()
        try:
            finder.prepare_tree()
        except (MoleculeException, ValueError) as err:
            raise PlannerInputError(str(err)) from err

        finder.tree_search(show_progress=False)
        try:
            finder.build_routes()
            if len(finder.routes):
                finder.routes.compute_scores(*finder.scorers.objects())
            routes = finder.routes.dict_with_extra(
                include_metadata=True,
                include_scores=True,
            )
            statistics = finder.extract_statistics()
            stock_info = finder.stock_info()
        except IndexError:
            # An AND/OR search with no expandable reaction has no route to rank.
            # Return a successful, unsolved response instead of an internal error.
            routes = []
            statistics = {
                "target": finder.target_smiles,
                "search_time": finder.search_stats.get("time", 0.0),
                "iterations": finder.search_stats.get("iterations", 0),
                "is_solved": False,
                "number_of_routes": 0,
                "number_of_solved_routes": 0,
            }
            stock_info = {}

        return {
            "smiles": finder.target_smiles,
            "algorithm": algorithm,
            "requested_algorithm": request.algorithm,
            "model": request.model,
            "selected_expansion_policy": _MODEL_SELECTIONS[request.model],
            "parameters": {
                "iterations": request.iterations,
                "time_limit": request.time_limit,
                "expansion_topk": request.expansion_topk,
                "max_transforms": request.max_transforms,
                "max_routes": request.max_routes,
                "return_first": request.return_first,
                "use_filter": request.use_filter,
                "exclude_target_from_stock": request.exclude_target_from_stock,
                "stocks": request.stocks or finder.stock.items,
            },
            "statistics": statistics,
            "stock_info": stock_info,
            "routes": routes,
        }


def _json_compatible(value: Any) -> Any:
    """Convert NumPy scalar/array values returned by scorers into JSON values."""
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_compatible(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_compatible(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


_planner = AiZynthFinderPlanner()
_disk_job_executor = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="aizynthfinder-job",
)
_disk_jobs_lock = threading.Lock()
_active_disk_jobs: dict[str, str] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as fileobj:
            json.dump(payload, fileobj, ensure_ascii=False, indent=2)
            fileobj.write("\n")
            fileobj.flush()
            os.fsync(fileobj.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _payload_sha256(payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _resolve_job_dir(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        raise HTTPException(status_code=422, detail="job_dir must be an absolute path")
    resolved = path.resolve()
    if DISK_JOB_ROOT:
        allowed_root = Path(DISK_JOB_ROOT).expanduser().resolve()
        try:
            resolved.relative_to(allowed_root)
        except ValueError as err:
            raise HTTPException(
                status_code=403,
                detail="job_dir is outside AIZYNTHFINDER_JOB_ROOT",
            ) from err
    return resolved


def _load_disk_request(job_dir: Path) -> tuple[dict[str, Any], PlanRequest]:
    request_path = job_dir / "request.json"
    try:
        with request_path.open(encoding="utf-8") as fileobj:
            envelope = json.load(fileobj)
    except (OSError, ValueError) as err:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot read disk job request.json: {err}",
        ) from err
    if not isinstance(envelope, dict):
        raise HTTPException(
            status_code=422, detail="request.json must contain an object"
        )
    if envelope.get("protocol_version") != DISK_JOB_PROTOCOL:
        raise HTTPException(status_code=422, detail="Unsupported disk job protocol")
    if envelope.get("service") != "aizynthfinder":
        raise HTTPException(
            status_code=422,
            detail="request.json service must be aizynthfinder",
        )
    job_id = envelope.get("job_id")
    if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", job_id):
        raise HTTPException(status_code=422, detail="Invalid disk job_id")
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=422, detail="request.json payload must be an object"
        )
    digest = _payload_sha256(payload)
    if envelope.get("request_sha256") != digest:
        raise HTTPException(status_code=422, detail="request.json digest mismatch")
    try:
        plan_request = PlanRequest(**payload)
    except Exception as err:
        raise HTTPException(status_code=422, detail=str(err)) from err
    return envelope, plan_request


def _read_status(job_dir: Path) -> dict[str, Any] | None:
    try:
        with (job_dir / "status.json").open(encoding="utf-8") as fileobj:
            status_payload = json.load(fileobj)
        return status_payload if isinstance(status_payload, dict) else None
    except (OSError, ValueError):
        return None


def _write_status(
    job_dir: Path,
    envelope: dict[str, Any],
    state: str,
    **extra: Any,
) -> dict[str, Any]:
    status_payload = {
        "protocol_version": DISK_JOB_PROTOCOL,
        "job_id": envelope["job_id"],
        "service": "aizynthfinder",
        "request_sha256": envelope["request_sha256"],
        "state": state,
        "updated_at_utc": _utc_now(),
        **extra,
    }
    _atomic_write_json(job_dir / "status.json", status_payload)
    return status_payload


def _run_disk_job(
    job_dir: Path,
    envelope: dict[str, Any],
    plan_request: PlanRequest,
) -> None:
    key = str(job_dir)
    try:
        _write_status(job_dir, envelope, "running", started_at_utc=_utc_now())
        result = _planner.plan(plan_request)
        compatible_result = jsonable_encoder(_json_compatible(result))
        _atomic_write_json(job_dir / "result.json", compatible_result)
        _write_status(
            job_dir,
            envelope,
            "completed",
            completed_at_utc=_utc_now(),
            result_file="result.json",
        )
    except Exception as err:  # pragma: no cover - exercised at integration boundary
        _LOGGER.exception("AiZynthFinder disk job failed")
        try:
            _write_status(
                job_dir,
                envelope,
                "failed",
                failed_at_utc=_utc_now(),
                error_type=type(err).__name__,
                error=str(err),
                traceback=traceback.format_exc(),
            )
        except Exception:
            _LOGGER.exception("Could not persist AiZynthFinder disk-job failure")
    finally:
        with _disk_jobs_lock:
            _active_disk_jobs.pop(key, None)


app = FastAPI(
    title="AiZynthFinder API",
    version="4.4.1",
    description="Local API for MCTS and Retro* retrosynthesis planning.",
)


@app.get("/health")
def health() -> dict[str, Any]:
    """Return service readiness without loading models eagerly."""
    return {
        "status": "ok",
        "service": "aizynthfinder-fastapi",
        "default_port": DEFAULT_PORT,
        "available_algorithms": ["mcts", "retrostar", "original"],
        "available_models": list(_MODEL_SELECTIONS),
        "available_search_setups": _SEARCH_SETUPS,
        "loaded_algorithms": _planner.loaded_algorithms,
        "configuration_files": {
            name: str(path) for name, path in _planner._config_paths.items()
        },
        "building_blocks": (
            str(Path(DEFAULT_BUILDING_BLOCKS).expanduser().resolve())
            if DEFAULT_BUILDING_BLOCKS
            else None
        ),
        "exclude_target_from_stock_supported": True,
        "filter_policy_default_enabled": False,
        "disk_job_protocol": DISK_JOB_PROTOCOL,
        "async_endpoint": "/aizynthfinder_plan_async",
        "disk_job_root": DISK_JOB_ROOT,
    }


@app.post("/aizynthfinder_plan")
async def aizynthfinder_plan(request: PlanRequest) -> dict[str, Any]:
    """Generate retrosynthetic routes for one target SMILES string."""
    try:
        result = await run_in_threadpool(_planner.plan, request)
    except PlannerInputError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err
    except (KeyError, PolicyException) as err:
        raise HTTPException(status_code=422, detail=str(err)) from err
    except Exception as err:  # pragma: no cover - preserves a useful API error boundary
        _LOGGER.exception("AiZynthFinder planning failed")
        raise HTTPException(
            status_code=500, detail="AiZynthFinder planning failed"
        ) from err
    return jsonable_encoder(_json_compatible(result))


@app.post("/aizynthfinder_plan_async", status_code=202)
def submit_disk_job(submission: DiskJobSubmission) -> dict[str, Any]:
    """Queue a disk-backed plan and return without waiting for computation."""

    job_dir = _resolve_job_dir(submission.job_dir)
    envelope, plan_request = _load_disk_request(job_dir)
    key = str(job_dir)
    digest = envelope["request_sha256"]

    existing_status = _read_status(job_dir)
    if (
        existing_status
        and existing_status.get("request_sha256") == digest
        and existing_status.get("state") == "completed"
        and (job_dir / "result.json").is_file()
    ):
        return {
            "job_id": envelope["job_id"],
            "request_sha256": digest,
            "state": "completed",
        }

    with _disk_jobs_lock:
        active_digest = _active_disk_jobs.get(key)
        if active_digest == digest:
            return {
                "job_id": envelope["job_id"],
                "request_sha256": digest,
                "state": "running",
            }
        if active_digest is not None:
            raise HTTPException(
                status_code=409,
                detail="A different request is already active in job_dir",
            )
        _active_disk_jobs[key] = digest
        try:
            _write_status(job_dir, envelope, "queued", queued_at_utc=_utc_now())
            _disk_job_executor.submit(
                _run_disk_job,
                job_dir,
                envelope,
                plan_request,
            )
        except Exception:
            _active_disk_jobs.pop(key, None)
            raise

    return {
        "job_id": envelope["job_id"],
        "request_sha256": digest,
        "state": "queued",
    }


def main() -> None:
    """Run the local API server on port 8001 by default."""
    import uvicorn

    parser = argparse.ArgumentParser(
        description="Run the local AiZynthFinder FastAPI service"
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("AIZYNTHFINDER_API_HOST", DEFAULT_HOST),
        help="Interface to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("AIZYNTHFINDER_API_PORT", str(DEFAULT_PORT))),
        help="TCP port to bind (default: 8001)",
    )
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
