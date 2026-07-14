"""Tests for the FastAPI wrapper without loading the large public models."""

import json
import logging
import time

from fastapi.testclient import TestClient

from aizynthfinder.interfaces import aizynthapi


def test_health_endpoint():
    response = TestClient(aizynthapi.app).get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["default_port"] == 8001
    assert payload["available_models"] == ["uspto", "ringbreaker", "multi"]
    assert payload["available_search_setups"]["retrostar_uspto_zinc"] == {
        "algorithm": "retrostar",
        "model": "uspto",
        "stocks": ["zinc"],
    }
    assert payload["filter_policy_default_enabled"] is False
    assert payload["disk_job_protocol"] == 1


def test_plan_endpoint_forwards_parameters(monkeypatch):
    captured = {}

    def fake_plan(request):
        captured["request"] = request
        return {
            "smiles": request.smiles,
            "algorithm": "retrostar",
            "requested_algorithm": request.algorithm,
            "model": request.model,
            "statistics": {"is_solved": True},
            "stock_info": {},
            "routes": [],
        }

    monkeypatch.setattr(aizynthapi._planner, "plan", fake_plan)
    response = TestClient(aizynthapi.app).post(
        "/aizynthfinder_plan",
        json={
            "smiles": "CCOC(=O)c1ccccc1",
            "algorithm": "retrostar",
            "model": "ringbreaker",
            "use_filter": True,
            "iterations": 100,
            "expansion_topk": 50,
        },
    )

    assert response.status_code == 200
    assert captured["request"].algorithm == "retrostar"
    assert captured["request"].model == "ringbreaker"
    assert captured["request"].use_filter is True
    assert captured["request"].iterations == 100
    assert response.json()["statistics"]["is_solved"] is True


def test_plan_endpoint_validates_topk():
    response = TestClient(aizynthapi.app).post(
        "/aizynthfinder_plan",
        json={"smiles": "CCO", "expansion_topk": 0},
    )

    assert response.status_code == 422


def test_plan_endpoint_supports_retrostar_uspto_zinc(monkeypatch):
    captured = {}

    def fake_plan(request):
        captured["request"] = request
        return {
            "smiles": request.smiles,
            "algorithm": request.algorithm,
            "requested_algorithm": request.algorithm,
            "model": request.model,
            "selected_expansion_policy": ["uspto"],
            "parameters": {"stocks": request.stocks},
            "statistics": {"is_solved": False},
            "stock_info": {},
            "routes": [],
        }

    monkeypatch.setattr(aizynthapi._planner, "plan", fake_plan)
    response = TestClient(aizynthapi.app).post(
        "/aizynthfinder_plan",
        json={
            "smiles": "CCOC(=O)c1ccccc1",
            "algorithm": "retrostar",
            "model": "uspto",
            "stocks": ["zinc"],
            "iterations": 10,
            "expansion_topk": 5,
        },
    )

    assert response.status_code == 200
    assert captured["request"].algorithm == "retrostar"
    assert captured["request"].model == "uspto"
    assert captured["request"].stocks == ["zinc"]
    assert captured["request"].use_filter is False
    assert response.json()["selected_expansion_policy"] == ["uspto"]


def test_planner_shares_building_block_query_between_algorithms(monkeypatch, tmp_path):
    stock_path = tmp_path / "zinc.csv"
    stock_path.write_text("smiles\nCCO\nCCN\n", encoding="utf-8")
    monkeypatch.setattr(aizynthapi, "DEFAULT_BUILDING_BLOCKS", str(stock_path))
    planner = aizynthapi.AiZynthFinderPlanner()

    first = planner._get_building_block_query()
    second = planner._get_building_block_query()

    assert first is second
    assert len(first) == 2


def test_planner_reuses_finder_and_configuration_when_only_smiles_changes(
    monkeypatch, tmp_path
):
    config_path = tmp_path / "config.yml"
    config_path.write_text("search: {}\n", encoding="utf-8")
    finder = object()
    finder_init_calls = []
    configure_calls = []
    planned_smiles = []

    def fake_finder_init(**kwargs):
        finder_init_calls.append(kwargs)
        return finder

    monkeypatch.setattr(aizynthapi, "AiZynthFinder", fake_finder_init)
    planner = aizynthapi.AiZynthFinderPlanner({"mcts": config_path})
    monkeypatch.setattr(
        planner,
        "_configure",
        lambda configured_finder, request: configure_calls.append(
            (configured_finder, request.smiles, request.iterations)
        ),
    )
    monkeypatch.setattr(
        planner,
        "_run_plan",
        lambda planned_finder, request, algorithm: planned_smiles.append(
            (planned_finder, request.smiles, algorithm)
        )
        or {"smiles": request.smiles},
    )

    planner.plan(aizynthapi.PlanRequest(smiles="CCO", iterations=10))
    planner.plan(aizynthapi.PlanRequest(smiles="CCN", iterations=10))

    assert finder_init_calls == [{"configfile": str(config_path)}]
    assert configure_calls == [(finder, "CCO", 10)]
    assert planned_smiles == [
        (finder, "CCO", "mcts"),
        (finder, "CCN", "mcts"),
    ]


def test_planner_reconfigures_without_reloading_when_parameters_change(
    monkeypatch, tmp_path
):
    config_path = tmp_path / "config.yml"
    config_path.write_text("search: {}\n", encoding="utf-8")
    finder = object()
    finder_init_calls = []
    configured_iterations = []

    def fake_finder_init(**kwargs):
        finder_init_calls.append(kwargs)
        return finder

    monkeypatch.setattr(aizynthapi, "AiZynthFinder", fake_finder_init)
    planner = aizynthapi.AiZynthFinderPlanner({"mcts": config_path})
    monkeypatch.setattr(
        planner,
        "_configure",
        lambda configured_finder, request: configured_iterations.append(
            request.iterations
        ),
    )
    monkeypatch.setattr(
        planner,
        "_run_plan",
        lambda planned_finder, request, algorithm: {"smiles": request.smiles},
    )

    planner.plan(aizynthapi.PlanRequest(smiles="CCO", iterations=10))
    planner.plan(aizynthapi.PlanRequest(smiles="CCN", iterations=20))

    assert finder_init_calls == [{"configfile": str(config_path)}]
    assert configured_iterations == [10, 20]


def test_planner_reports_resource_loading_only_on_cache_miss(
    monkeypatch, tmp_path, caplog
):
    config_path = tmp_path / "config.yml"
    config_path.write_text("search: {}\n", encoding="utf-8")

    class FakeCollection:
        items = ["uspto"]

    class FakeStock:
        items = ["zinc"]

    class FakeFinder:
        expansion_policy = FakeCollection()
        stock = FakeStock()

    monkeypatch.setattr(aizynthapi, "AiZynthFinder", lambda **kwargs: FakeFinder())
    planner = aizynthapi.AiZynthFinderPlanner({"mcts": config_path})
    caplog.set_level(logging.INFO, logger="uvicorn.error.aizynthfinder")

    first = planner._get_slot("mcts")
    second = planner._get_slot("mcts")

    assert first is second
    loading_records = [
        record
        for record in caplog.records
        if "algorithm models and purchasable stock" in record.getMessage()
    ]
    assert len(loading_records) == 2
    assert "Loading algorithm models" in loading_records[0].getMessage()
    assert "reason=cache_miss" in loading_records[0].getMessage()
    assert "Loaded algorithm models" in loading_records[1].getMessage()
    assert "models=['uspto']" in loading_records[1].getMessage()
    assert "stocks=['zinc']" in loading_records[1].getMessage()


def test_async_endpoint_reads_and_writes_disk(monkeypatch, tmp_path):
    payload = {
        "smiles": "CCO",
        "algorithm": "mcts",
        "model": "uspto",
        "iterations": 1,
    }
    digest = aizynthapi._payload_sha256(payload)
    request_envelope = {
        "protocol_version": 1,
        "job_id": "test-aizynthfinder-job",
        "service": "aizynthfinder",
        "request_sha256": digest,
        "payload": payload,
    }
    (tmp_path / "request.json").write_text(
        json.dumps(request_envelope),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        aizynthapi._planner,
        "plan",
        lambda request: {"smiles": request.smiles, "routes": []},
    )

    response = TestClient(aizynthapi.app).post(
        "/aizynthfinder_plan_async",
        json={"job_dir": str(tmp_path)},
    )

    assert response.status_code == 202
    assert response.json()["request_sha256"] == digest
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
        if status["state"] == "completed":
            break
        time.sleep(0.01)
    assert status["state"] == "completed"
    assert json.loads((tmp_path / "result.json").read_text(encoding="utf-8")) == {
        "smiles": "CCO",
        "routes": [],
    }
