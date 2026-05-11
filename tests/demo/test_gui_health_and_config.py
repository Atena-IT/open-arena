from pathlib import Path
import sys

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.gui.backend.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_demo_config_endpoint() -> None:
    response = client.get("/api/demo/config")
    assert response.status_code == 200
    data = response.json()

    # sampleLimit
    assert data["sampleLimit"] == 20

    # dataset
    assert data["dataset"]["csvPath"] == "demo/show_me_how_open_arena/data/business_qa_demo.csv"
    assert data["dataset"]["rowCount"] >= 200

    # runtimeDatasetName contains sample limit
    assert "20" in data["runtimeDatasetName"]

    # modelMapping
    mapping = data["modelMapping"]
    assert isinstance(mapping, list)
    assert len(mapping) > 0
    for item in mapping:
        assert "experimentKey" in item
        assert "experimentName" in item
        assert "showcaseModel" in item
        assert "backendModel" in item
        assert item["experimentKey"]

    # heroMission
    hero = data["heroMission"]
    for field in [
        "missionTitle",
        "researchDomain",
        "timeframeStart",
        "timeframeEnd",
        "allowedDomains",
        "focusSemantics",
        "outputType",
        "question",
        "expectedAnswer",
    ]:
        assert field in hero, f"heroMission missing field: {field}"
        assert hero[field], f"heroMission field empty: {field}"

    # envStatus
    env = data["envStatus"]
    for key in [
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_HOST",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "ANTHROPIC_API_KEY",
        "HUGGINGFACE_API_KEY",
    ]:
        assert key in env, f"envStatus missing key: {key}"
        assert isinstance(env[key], bool), f"envStatus[{key}] not bool"

    # evaluationDefaults
    defaults = data["evaluationDefaults"]
    assert defaults["method"] == "llm_as_judge"
    assert defaults["label"] == "Notebook Judge"
    assert defaults["systemPrompt"]
    assert defaults["systemPromptNoReference"]

