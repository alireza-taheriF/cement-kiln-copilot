"""Dry-run snapshots for the three cloud training launchers.

No AWS, GCP, or Azure credentials are required. Submit is not called.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from src.cloud.azureml_job import REQUIRED_SUBMIT_ENV as AZURE_ENV
from src.cloud.azureml_job import build_job_spec as azure_spec
from src.cloud.azureml_job import main as azure_main
from src.cloud.sagemaker_job import REQUIRED_SUBMIT_ENV as SAGEMAKER_ENV
from src.cloud.sagemaker_job import build_job_spec as sagemaker_spec
from src.cloud.sagemaker_job import main as sagemaker_main
from src.cloud.training_command import build_training_command
from src.cloud.vertex_job import REQUIRED_SUBMIT_ENV as VERTEX_ENV
from src.cloud.vertex_job import build_job_spec as vertex_spec
from src.cloud.vertex_job import main as vertex_main

COMMAND = ["python", "-m", "src.ml.train", "--config", "config/model_config.yaml"]

SAGEMAKER_SNAPSHOT = {
    "provider": "aws-sagemaker",
    "command": COMMAND,
    "image": "PLACEHOLDER_ECR_IMAGE",
    "instance_type": "ml.m5.large",
    "training_job": {
        "TrainingJobName": "cement-kiln-copilot-train",
        "AlgorithmSpecification": {
            "TrainingImage": "PLACEHOLDER_ECR_IMAGE",
            "TrainingInputMode": "File",
            "ContainerEntrypoint": ["python", "-m", "src.ml.train"],
            "ContainerArguments": ["--config", "config/model_config.yaml"],
        },
        "RoleArn": "PLACEHOLDER_ROLE_ARN",
        "OutputDataConfig": {
            "S3OutputPath": "s3://PLACEHOLDER_BUCKET/cement-kiln-copilot"
        },
        "ResourceConfig": {
            "InstanceType": "ml.m5.large",
            "InstanceCount": 1,
            "VolumeSizeInGB": 30,
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 3600},
    },
}

VERTEX_SNAPSHOT = {
    "provider": "gcp-vertex-ai",
    "command": COMMAND,
    "image": "PLACEHOLDER_GCR_IMAGE",
    "instance_type": "n1-standard-4",
    "custom_job": {
        "display_name": "cement-kiln-copilot-train",
        "job_spec": {
            "worker_pool_specs": [
                {
                    "machine_spec": {"machine_type": "n1-standard-4"},
                    "replica_count": 1,
                    "container_spec": {
                        "image_uri": "PLACEHOLDER_GCR_IMAGE",
                        "command": COMMAND,
                    },
                }
            ]
        },
    },
}

AZURE_SNAPSHOT = {
    "provider": "azure-ml",
    "command": COMMAND,
    "image": "PLACEHOLDER_ACR_IMAGE",
    "instance_type": "Standard_DS3_v2",
    "command_job": {
        "type": "command",
        "display_name": "cement-kiln-copilot-train",
        "experiment_name": "cement-kiln-copilot",
        "command": "python -m src.ml.train --config config/model_config.yaml",
        "environment": "PLACEHOLDER_AZUREML_ENVIRONMENT",
        "compute": "PLACEHOLDER_COMPUTE",
        "resources": {
            "instance_type": "Standard_DS3_v2",
            "instance_count": 1,
        },
    },
}


def test_shared_training_command_is_the_local_trainer():
    assert build_training_command() == COMMAND


def test_sagemaker_spec_snapshot():
    spec = sagemaker_spec()
    assert spec == SAGEMAKER_SNAPSHOT
    assert spec["command"] == build_training_command()
    assert spec["image"] == "PLACEHOLDER_ECR_IMAGE"
    assert spec["instance_type"] == "ml.m5.large"
    assert spec["training_job"]["ResourceConfig"]["InstanceType"] == "ml.m5.large"
    assert spec["training_job"]["AlgorithmSpecification"]["TrainingImage"] == spec["image"]


def test_vertex_spec_snapshot():
    spec = vertex_spec()
    assert spec == VERTEX_SNAPSHOT
    assert spec["command"] == build_training_command()
    assert spec["image"] == "PLACEHOLDER_GCR_IMAGE"
    assert spec["instance_type"] == "n1-standard-4"
    pool = spec["custom_job"]["job_spec"]["worker_pool_specs"][0]
    assert pool["machine_spec"]["machine_type"] == "n1-standard-4"
    assert pool["container_spec"]["image_uri"] == spec["image"]
    assert pool["container_spec"]["command"] == spec["command"]


def test_azure_spec_snapshot():
    spec = azure_spec()
    assert spec == AZURE_SNAPSHOT
    assert spec["command"] == build_training_command()
    assert spec["image"] == "PLACEHOLDER_ACR_IMAGE"
    assert spec["instance_type"] == "Standard_DS3_v2"
    assert spec["command_job"]["resources"]["instance_type"] == "Standard_DS3_v2"
    assert spec["command_job"]["command"] == " ".join(spec["command"])


def test_three_specs_share_one_command():
    command = build_training_command()
    assert sagemaker_spec()["command"] == command
    assert vertex_spec()["command"] == command
    assert azure_spec()["command"] == command


@pytest.mark.parametrize(
    "module",
    [
        "src.cloud.sagemaker_job",
        "src.cloud.vertex_job",
        "src.cloud.azureml_job",
    ],
)
def test_dry_run_cli_exits_zero_without_cloud_credentials(module: str):
    canary = "canary-secret-do-not-print"
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "AWS_SECRET_ACCESS_KEY": canary,
        "AZURE_CLIENT_SECRET": canary,
        "GOOGLE_APPLICATION_CREDENTIALS": canary,
    }
    proc = subprocess.run(
        [sys.executable, "-m", module, "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert proc.returncode == 0, proc.stderr
    spec = json.loads(proc.stdout)
    assert spec["command"] == COMMAND
    assert "PLACEHOLDER" in spec["image"]
    assert spec["instance_type"]
    assert canary not in proc.stdout
    assert canary not in proc.stderr


@pytest.mark.parametrize(
    ("entry", "required"),
    [
        (sagemaker_main, SAGEMAKER_ENV),
        (vertex_main, VERTEX_ENV),
        (azure_main, AZURE_ENV),
    ],
)
def test_submit_exits_when_credentials_are_absent(monkeypatch, capsys, entry, required):
    for name in required:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "argv", ["job", "--submit"])
    with pytest.raises(SystemExit) as exc:
        entry()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert required[0] in err
    assert "not exercised in CI" in err
