"""CDK Stack: Daily batch inference via SageMaker Processing Job.

Architecture:
  EventBridge (daily cron) → Lambda → SageMaker Processing Job

All resource names are derived from `project_name` context variable for reusability.
"""
import aws_cdk as cdk
from aws_cdk import (
    Stack,
    Duration,
    aws_ec2 as ec2,
    aws_events as events,
    aws_events_targets as targets,
    aws_iam as iam,
    aws_kms as kms,
    aws_lambda as _lambda,
    aws_sqs as sqs,
)
from constructs import Construct


class BatchInferenceStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # --- Context variables (configurable per project) ---
        project_name = self.node.try_get_context("project_name") or "pdm"
        input_bucket = self.node.try_get_context("input_bucket")  # REQUIRED
        ecr_image_uri = self.node.try_get_context("ecr_image_uri")  # REQUIRED
        model_s3_uri = self.node.try_get_context("model_s3_uri")  # REQUIRED
        instance_type = self.node.try_get_context("instance_type") or "ml.m5.xlarge"
        schedule_expression = self.node.try_get_context("schedule") or "cron(0 6 * * ? *)"
        lookback_days = self.node.try_get_context("lookback_days") or "7"

        if not input_bucket:
            raise ValueError("Context variable 'input_bucket' is required")
        if not ecr_image_uri:
            raise ValueError("Context variable 'ecr_image_uri' is required")
        if not model_s3_uri:
            raise ValueError("Context variable 'model_s3_uri' is required")

        output_bucket = f"{input_bucket}-predictions"

        # --- SageMaker Execution Role ---
        sagemaker_role = iam.Role(
            self, "SageMakerRole",
            role_name=f"{project_name}-batch-sagemaker-role",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"),
            ],
        )
        # S3 access for input and output buckets
        sagemaker_role.add_to_policy(iam.PolicyStatement(
            actions=["s3:GetObject", "s3:ListBucket"],
            resources=[
                f"arn:aws:s3:::{input_bucket}",
                f"arn:aws:s3:::{input_bucket}/*",
            ],
        ))
        sagemaker_role.add_to_policy(iam.PolicyStatement(
            actions=["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
            resources=[
                f"arn:aws:s3:::{output_bucket}",
                f"arn:aws:s3:::{output_bucket}/*",
            ],
        ))
        # ECR pull
        sagemaker_role.add_to_policy(iam.PolicyStatement(
            actions=[
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage",
                "ecr:GetAuthorizationToken",
            ],
            resources=["*"],
        ))

        # --- Lambda Execution Role ---
        lambda_role = iam.Role(
            self, "LambdaRole",
            role_name=f"{project_name}-batch-lambda-role",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaVPCAccessExecutionRole"),
            ],
        )
        lambda_role.add_to_policy(iam.PolicyStatement(
            actions=["sagemaker:CreateProcessingJob", "sagemaker:DescribeProcessingJob"],
            resources=[
                f"arn:aws:sagemaker:{self.region}:{self.account}:processing-job/{project_name}-*",
            ],
        ))
        lambda_role.add_to_policy(iam.PolicyStatement(
            actions=["iam:PassRole"],
            resources=[sagemaker_role.role_arn],
        ))

        # --- Dead Letter Queue ---
        dlq = sqs.Queue(
            self, "TriggerDLQ",
            queue_name=f"{project_name}-batch-trigger-dlq",
            encryption=sqs.QueueEncryption.KMS_MANAGED,
        )

        # --- KMS key for Lambda env var encryption ---
        env_key = kms.Key(self, "LambdaEnvKey", enable_key_rotation=True)

        # --- VPC for Lambda (uses default VPC or specify vpc_id context) ---
        vpc_id = self.node.try_get_context("vpc_id")
        vpc = (
            ec2.Vpc.from_lookup(self, "Vpc", vpc_id=vpc_id)
            if vpc_id
            else ec2.Vpc.from_lookup(self, "Vpc", is_default=True)
        )

        # --- Lambda Function ---
        trigger_fn = _lambda.Function(
            self, "TriggerFunction",
            function_name=f"{project_name}-batch-trigger",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="trigger.handler",
            code=_lambda.Code.from_asset("lambda"),
            timeout=Duration.seconds(30),
            role=lambda_role,
            reserved_concurrent_executions=1,
            dead_letter_queue=dlq,
            environment_encryption=env_key,
            vpc=vpc,
            environment={
                "PROJECT_NAME": project_name,
                "INPUT_BUCKET": input_bucket,
                "OUTPUT_BUCKET": output_bucket,
                "ECR_IMAGE_URI": ecr_image_uri,
                "MODEL_S3_URI": model_s3_uri,
                "SAGEMAKER_ROLE_ARN": sagemaker_role.role_arn,
                "INSTANCE_TYPE": instance_type,
                "LOOKBACK_DAYS": str(lookback_days),
            },
        )

        # --- EventBridge Schedule ---
        rule = events.Rule(
            self, "DailySchedule",
            rule_name=f"{project_name}-batch-daily",
            schedule=events.Schedule.expression(schedule_expression),
        )
        rule.add_target(targets.LambdaFunction(trigger_fn))

        # --- Outputs ---
        cdk.CfnOutput(self, "LambdaFunctionName", value=trigger_fn.function_name)
        cdk.CfnOutput(self, "SageMakerRoleArn", value=sagemaker_role.role_arn)
        cdk.CfnOutput(self, "ScheduleExpression", value=schedule_expression)
        cdk.CfnOutput(self, "OutputBucket", value=output_bucket)
