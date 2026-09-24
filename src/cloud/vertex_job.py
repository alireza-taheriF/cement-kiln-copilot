"""GCP Vertex AI custom-job launcher.

Dry-run prints a stable job spec and does not need GCP credentials.
Submit creates a custom job only when the expected environment is set.

    python -m src.cloud.vertex_job --dry-run
    python -m src.cloud.vertex_job --submit
"""

from __future__ import annotations

import json
import os
import sys

from src.cloud.cli import run_job_cli
from src.cloud.credentials import ensure_import, ensure_submit_env
from src.cloud.training_command import build_training_command

IMAGE_PLACEHOLDER = "PLACEHOLDER_GCR_IMAGE"
INSTANCE_TYPE = "n1-standard-4"

REQUIRED_SUBMIT_ENV = (
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_REGION",
    "VERTEX_TRAINING_IMAGE",
)


def build_job_spec() -> dict:
    """Vertex AI custom-job spec. Placeholders only; no credentials."""
    command = build_training_command()
    return {
        "provider": "gcp-vertex-ai",
        "command": command,
        "image": IMAGE_PLACEHOLDER,
        "instance_type": INSTANCE_TYPE,
        "custom_job": {
            "display_name": "cement-kiln-copilot-train",
            "job_spec": {
                "worker_pool_specs": [
                    {
                        "machine_spec": {"machine_type": INSTANCE_TYPE},
                        "replica_count": 1,
                        "container_spec": {
                            "image_uri": IMAGE_PLACEHOLDER,
                            "command": command,
                        },
                    }
                ]
            },
        },
    }


def submit_job() -> None:
    """Submit a Vertex AI custom job when GCP environment variables are set."""
    ensure_submit_env("GCP Vertex AI", REQUIRED_SUBMIT_ENV)
    credentials_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    if not os.path.isfile(credentials_path):
        print(
            "Cannot submit the GCP Vertex AI training job. "
            "GOOGLE_APPLICATION_CREDENTIALS is not a file. "
            "The submit path needs a real cloud account and is not exercised in CI. "
            "Train locally instead:\n"
            "  python -m src.ml.train --config config/model_config.yaml",
            file=sys.stderr,
        )
        raise SystemExit(1)

    aiplatform = ensure_import("google.cloud.aiplatform", "google-cloud-aiplatform")
    aiplatform.init(
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location=os.environ["GOOGLE_CLOUD_REGION"],
    )
    job = aiplatform.CustomJob(
        display_name="cement-kiln-copilot-train",
        worker_pool_specs=[
            {
                "machine_spec": {"machine_type": INSTANCE_TYPE},
                "replica_count": 1,
                "container_spec": {
                    "image_uri": os.environ["VERTEX_TRAINING_IMAGE"],
                    "command": build_training_command(),
                },
            }
        ],
    )
    job.submit()
    print(
        json.dumps(
            {
                "submitted": True,
                "provider": "gcp-vertex-ai",
                "display_name": "cement-kiln-copilot-train",
            }
        )
    )


def main() -> None:
    run_job_cli(
        build_job_spec,
        submit_job,
        description="Print or submit the Vertex AI custom job for src.ml.train.",
    )


if __name__ == "__main__":
    main()
