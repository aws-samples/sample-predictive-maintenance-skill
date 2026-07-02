#!/usr/bin/env python3
"""CDK app for PdM batch inference infrastructure.

Deploys:
- EventBridge rule (daily cron)
- Lambda function (triggers SageMaker Processing Job)
- IAM roles for Lambda and SageMaker
"""
import aws_cdk as cdk

from batch_inference_stack import BatchInferenceStack

app = cdk.App()
BatchInferenceStack(
    app,
    "PdmBatchInferenceStack",
    env=cdk.Environment(
        account=app.node.try_get_context("account"),
        region=app.node.try_get_context("region") or "eu-central-1",
    ),
)
app.synth()
