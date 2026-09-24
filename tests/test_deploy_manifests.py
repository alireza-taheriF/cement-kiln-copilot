"""Static checks for the container ignore file and local Kubernetes manifests."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
K8S = ROOT / "deploy" / "k8s"


def _load(name: str) -> dict:
    with (K8S / name).open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    assert isinstance(document, dict)
    return document


def test_dockerignore_excludes_local_secrets_and_stores():
    text = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    for token in (".venv", "mlruns", ".env", "*.sqlite", "*.db"):
        assert token in text.splitlines()


def test_api_deployment_probes_and_port():
    deployment = _load("api-deployment.yaml")
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert container["ports"][0]["containerPort"] == 8000
    for probe_name in ("readinessProbe", "livenessProbe"):
        probe = container[probe_name]["httpGet"]
        assert probe["path"] == "/health"
        assert probe["port"] == "http"


def test_configmap_is_non_secret_and_secret_example_holds_database_url():
    config = _load("configmap.yaml")["data"]
    assert config["MODEL_PATH"] == "/models/kiln_zone1_temp_model.joblib"
    assert "DATABASE_URL" not in config
    assert "CHANGE_ME" not in yaml.safe_dump(config)

    secret = _load("secret.example.yaml")
    assert secret["kind"] == "Secret"
    url = secret["stringData"]["DATABASE_URL"]
    assert "CHANGE_ME" in url
    assert "AKIA" not in url
    assert "aws_secret" not in url.lower()


def test_dashboard_points_at_the_api_service():
    deployment = _load("dashboard-deployment.yaml")
    env = deployment["spec"]["template"]["spec"]["containers"][0]["env"]
    values = {item["name"]: item["value"] for item in env}
    assert values["API_BASE_URL"] == "http://cement-copilot-api:8000"


def test_mlflow_pvc_is_documented_as_local_only():
    pvc_text = (K8S / "mlflow-pvc.yaml").read_text(encoding="utf-8")
    assert "local kind" in pvc_text.lower() or "kind" in pvc_text.lower()
    assert "not a production" in pvc_text.lower()
    pvc = _load("mlflow-pvc.yaml")
    assert pvc["kind"] == "PersistentVolumeClaim"
    assert pvc["metadata"]["name"] == "cement-copilot-mlruns"

    deployment = _load("mlflow-deployment.yaml")
    command = deployment["spec"]["template"]["spec"]["containers"][0]["command"]
    assert "sqlite:////mlflow/mlflow.db" in command
    assert "/mlflow/mlruns" in command


def test_kustomization_lists_the_manifests():
    kustomization = _load("kustomization.yaml")
    resources = set(kustomization["resources"])
    assert "secret.example.yaml" in resources
    assert "api-deployment.yaml" in resources
    assert "mlflow-pvc.yaml" in resources
    assert (K8S / "README.md").is_file()
