"""AWS SageMaker training launcher.

Dry-run prints a stable job spec and does not need AWS credentials.
Submit calls ``CreateTrainingJob`` only when the expected environment is set.

    python -m src.cloud.sagemaker_job --dry-run
    python -m src.cloud.sagemaker_job --submit
"""

from __future__ import annotations

import json
import os
import time

from src.cloud.cli import run_job_cli
from src.cloud.credentials import ensure_import, ensure_submit_env
from src.cloud.training_command import build_training_command

IMAGE_PLACEHOLDER = "PLACEHOLDER_ECR_IMAGE"
INSTANCE_TYPE = "ml.m5.large"
ROLE_PLACEHOLDER = "PLACEHOLDER_ROLE_ARN"
OUTPUT_PLACEHOLDER = "s3://PLACEHOLDER_BUCKET/cement-kiln-copilot"

REQUIRED_SUBMIT_ENV = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_DEFAULT_REGION",
    "SAGEMAKER_ROLE_ARN",
    "SAGEMAKER_TRAINING_IMAGE",
    "SAGEMAKER_OUTPUT_S3_URI",
)


def build_job_spec() -> dict:
    """SageMaker training-job spec. Placeholders only; no credentials."""
    command = build_training_command()
    return {
        "provider": "aws-sagemaker",
        "command": command,
        "image": IMAGE_PLACEHOLDER,
        "instance_type": INSTANCE_TYPE,
        "training_job": {
            "TrainingJobName": "cement-kiln-copilot-train",
            "AlgorithmSpecification": {
                "TrainingImage": IMAGE_PLACEHOLDER,
                "TrainingInputMode": "File",
                "ContainerEntrypoint": command[:3],
                "ContainerArguments": command[3:],
            },
            "RoleArn": ROLE_PLACEHOLDER,
            "OutputDataConfig": {"S3OutputPath": OUTPUT_PLACEHOLDER},
            "ResourceConfig": {
                "InstanceType": INSTANCE_TYPE,
                "InstanceCount": 1,
                "VolumeSizeInGB": 30,
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": 3600},
        },
    }


def submit_job() -> None:
    """Create a SageMaker training job when AWS environment variables are set."""
    ensure_submit_env("AWS SageMaker", REQUIRED_SUBMIT_ENV)
    boto3 = ensure_import("boto3", "boto3")
    command = build_training_command()
    job_name = "cement-kiln-copilot-" + time.strftime("%Y%m%d%H%M%S")
    client = boto3.client("sagemaker", region_name=os.environ["AWS_DEFAULT_REGION"])
    client.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            "TrainingImage": os.environ["SAGEMAKER_TRAINING_IMAGE"],
            "TrainingInputMode": "File",
            "ContainerEntrypoint": command[:3],
            "ContainerArguments": command[3:],
        },
        RoleArn=os.environ["SAGEMAKER_ROLE_ARN"],
        OutputDataConfig={"S3OutputPath": os.environ["SAGEMAKER_OUTPUT_S3_URI"]},
        ResourceConfig={
            "InstanceType": INSTANCE_TYPE,
            "InstanceCount": 1,
            "VolumeSizeInGB": 30,
        },
        StoppingCondition={"MaxRuntimeInSeconds": 3600},
    )
    print(
        json.dumps(
            {
                "submitted": True,
                "provider": "aws-sagemaker",
                "training_job_name": job_name,
            }
        )
    )


def main() -> None:
    run_job_cli(
        build_job_spec,
        submit_job,
        description="Print or submit the SageMaker training job for src.ml.train.",
    )


if __name__ == "__main__":
    main()
