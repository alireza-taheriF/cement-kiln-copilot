"""Azure Machine Learning command-job launcher.

Dry-run prints a stable job spec and does not need Azure credentials.
Submit creates a command job only when the expected environment is set.

    python -m src.cloud.azureml_job --dry-run
    python -m src.cloud.azureml_job --submit
"""

from __future__ import annotations

import json
import os

from src.cloud.cli import run_job_cli
from src.cloud.credentials import ensure_import, ensure_submit_env
from src.cloud.training_command import build_training_command

IMAGE_PLACEHOLDER = "PLACEHOLDER_ACR_IMAGE"
INSTANCE_TYPE = "Standard_DS3_v2"
ENVIRONMENT_PLACEHOLDER = "PLACEHOLDER_AZUREML_ENVIRONMENT"
COMPUTE_PLACEHOLDER = "PLACEHOLDER_COMPUTE"

REQUIRED_SUBMIT_ENV = (
    "AZURE_SUBSCRIPTION_ID",
    "AZURE_RESOURCE_GROUP",
    "AZURE_ML_WORKSPACE",
    "AZURE_CLIENT_ID",
    "AZURE_TENANT_ID",
    "AZURE_CLIENT_SECRET",
    "AZUREML_ENVIRONMENT",
    "AZUREML_COMPUTE",
)


def build_job_spec() -> dict:
    """Azure ML command-job spec. Placeholders only; no credentials."""
    command = build_training_command()
    return {
        "provider": "azure-ml",
        "command": command,
        "image": IMAGE_PLACEHOLDER,
        "instance_type": INSTANCE_TYPE,
        "command_job": {
            "type": "command",
            "display_name": "cement-kiln-copilot-train",
            "experiment_name": "cement-kiln-copilot",
            "command": " ".join(command),
            "environment": ENVIRONMENT_PLACEHOLDER,
            "compute": COMPUTE_PLACEHOLDER,
            "resources": {
                "instance_type": INSTANCE_TYPE,
                "instance_count": 1,
            },
        },
    }


def submit_job() -> None:
    """Create an Azure ML command job when Azure environment variables are set."""
    ensure_submit_env("Azure ML", REQUIRED_SUBMIT_ENV)
    ensure_import("azure.identity", "azure-identity")
    ensure_import("azure.ai.ml", "azure-ai-ml")
    from azure.ai.ml import MLClient, command
    from azure.identity import ClientSecretCredential

    credential = ClientSecretCredential(
        tenant_id=os.environ["AZURE_TENANT_ID"],
        client_id=os.environ["AZURE_CLIENT_ID"],
        client_secret=os.environ["AZURE_CLIENT_SECRET"],
    )
    client = MLClient(
        credential,
        os.environ["AZURE_SUBSCRIPTION_ID"],
        os.environ["AZURE_RESOURCE_GROUP"],
        os.environ["AZURE_ML_WORKSPACE"],
    )
    job = command(
        code=".",
        command=" ".join(build_training_command()),
        environment=os.environ["AZUREML_ENVIRONMENT"],
        compute=os.environ["AZUREML_COMPUTE"],
        display_name="cement-kiln-copilot-train",
        experiment_name="cement-kiln-copilot",
        resources={"instance_type": INSTANCE_TYPE, "instance_count": 1},
    )
    created = client.jobs.create_or_update(job)
    print(
        json.dumps(
            {
                "submitted": True,
                "provider": "azure-ml",
                "job_name": getattr(created, "name", None),
            }
        )
    )


def main() -> None:
    run_job_cli(
        build_job_spec,
        submit_job,
        description="Print or submit the Azure ML command job for src.ml.train.",
    )


if __name__ == "__main__":
    main()
