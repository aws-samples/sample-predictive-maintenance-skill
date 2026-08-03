"""CDK Stack: IoT and Greengrass resources for PdM edge deployment.

Creates the persistent IoT infrastructure: policy, token exchange role,
S3 artifacts bucket, and optional prediction observability (IoT Rules,
CloudWatch metrics, dashboard).

These resources are independent of any specific device and persist across
device lifecycle changes.

Following the pattern from:
  greengrass_resources_stack.py (self-improving-robots)
  IoTResources.yaml (guidance-for-deploying-ai-agents-to-device-fleets)
"""

import json

import aws_cdk as cdk
from aws_cdk import (
    Stack,
    RemovalPolicy,
    aws_cloudwatch as cloudwatch,
    aws_iam as iam,
    aws_iot as iot,
    aws_s3 as s3,
)
from constructs import Construct


class PdmEdgeResourcesStack(Stack):
    """IoT/Greengrass resources: TES role, IoT policy, S3 bucket, observability.

    Exports:
        artifacts_bucket: S3 bucket for Greengrass component artifacts
        token_exchange_role: IAM role assumed via IoT certificate
        token_exchange_role_alias: IoT Role Alias (ref name)
        iot_policy_name: IoT Policy name to attach to device certificates
    """

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        project_name = self.node.try_get_context("project_name") or "pdm-edge"
        enable_s3_logging = self._bool_context("enable_s3_logging")
        enable_cw_metrics = self._bool_context("enable_cloudwatch_metrics")
        enable_dashboard = self._bool_context("enable_dashboard")
        metric_field = self.node.try_get_context("metric_field") or "machine_failure_proba"
        metric_name = self.node.try_get_context("metric_name") or "FailureProbability"

        # --- S3 Bucket for component artifacts ---
        # Auto-generated name avoids conflicts and GDK naming issues
        self.artifacts_bucket = s3.Bucket(
            self, "ArtifactsBucket",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        # --- Token Exchange Role ---
        # Following the guidance pattern: separate managed policies for each concern

        tes_base_policy = iam.ManagedPolicy(
            self, "TESBasePolicy",
            managed_policy_name=f"{project_name}-GGV2TokenExchangeRoleAccess",
            statements=[
                iam.PolicyStatement(
                    actions=[
                        "iot:DescribeCertificate",
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:PutLogEvents",
                        "logs:DescribeLogStreams",
                        "s3:GetBucketLocation",
                    ],
                    resources=["*"],
                ),
            ],
        )

        tes_artifacts_policy = iam.ManagedPolicy(
            self, "TESArtifactsPolicy",
            statements=[
                iam.PolicyStatement(
                    actions=["s3:GetObject"],
                    resources=[
                        self.artifacts_bucket.arn_for_objects("*"),
                        "arn:aws:s3:::greengrass-artifacts*/*",
                    ],
                ),
            ],
        )

        self.token_exchange_role = iam.Role(
            self, "TokenExchangeRole",
            role_name=f"{project_name}-GGV2TokenExchangeRole",
            assumed_by=iam.ServicePrincipal("credentials.iot.amazonaws.com"),
            managed_policies=[tes_base_policy, tes_artifacts_policy],
        )

        self.token_exchange_role_alias = iot.CfnRoleAlias(
            self, "TokenExchangeRoleAlias",
            role_arn=self.token_exchange_role.role_arn,
        )

        # --- IoT Policy ---
        # 6 statements matching the guidance repo pattern exactly
        self.iot_policy = iot.CfnPolicy(
            self, "IoTPolicy",
            policy_name=f"{project_name}-iot-policy",
            policy_document=json.dumps({
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "greengrass:GetComponentVersionArtifact",
                            "greengrass:ResolveComponentCandidates",
                            "greengrass:GetDeploymentConfiguration",
                            "greengrass:ListThingGroupsForCoreDevice",
                        ],
                        "Resource": ["*"],
                    },
                    {
                        "Effect": "Allow",
                        "Action": ["iot:Connect"],
                        "Resource": [
                            f"arn:aws:iot:{self.region}:{self.account}:client/*"
                        ],
                    },
                    {
                        "Effect": "Allow",
                        "Action": ["iot:Receive", "iot:Publish"],
                        "Resource": [
                            f"arn:aws:iot:{self.region}:{self.account}:topic/$aws/things/*/greengrass/health/json",
                            f"arn:aws:iot:{self.region}:{self.account}:topic/$aws/things/*/greengrassv2/health/json",
                            f"arn:aws:iot:{self.region}:{self.account}:topic/$aws/things/*/shadow/*",
                            f"arn:aws:iot:{self.region}:{self.account}:topic/$aws/things/*/jobs/*",
                            f"arn:aws:iot:{self.region}:{self.account}:topic/things/*",
                        ],
                    },
                    {
                        "Effect": "Allow",
                        "Action": ["iot:Subscribe"],
                        "Resource": [
                            f"arn:aws:iot:{self.region}:{self.account}:topicfilter/$aws/things/*/shadow/*",
                            f"arn:aws:iot:{self.region}:{self.account}:topicfilter/$aws/things/*/jobs/*",
                            f"arn:aws:iot:{self.region}:{self.account}:topicfilter/things/*",
                        ],
                    },
                    {
                        "Effect": "Allow",
                        "Action": ["iot:AssumeRoleWithCertificate"],
                        "Resource": [self.token_exchange_role_alias.attr_role_alias_arn],
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "iot:GetThingShadow",
                            "iot:UpdateThingShadow",
                            "iot:DeleteThingShadow",
                        ],
                        "Resource": [
                            f"arn:aws:iot:{self.region}:{self.account}:thing/*"
                        ],
                    },
                ],
            }),
        )

        # Expose the policy name as a property
        self.iot_policy_name = self.iot_policy.policy_name

        # --- Prediction Observability (optional) ---

        iot_rule_role = None
        if enable_s3_logging or enable_cw_metrics:
            iot_rule_role = iam.Role(
                self, "IoTRuleRole",
                assumed_by=iam.ServicePrincipal("iot.amazonaws.com"),
            )

        if enable_s3_logging:
            predictions_bucket = s3.Bucket(
                self, "PredictionsBucket",
                removal_policy=RemovalPolicy.DESTROY,
                auto_delete_objects=True,
            )
            predictions_bucket.grant_put(iot_rule_role)

            iot.CfnTopicRule(
                self, "S3LoggingRule",
                rule_name=f"{project_name.replace('-', '_')}_s3_logging",
                topic_rule_payload=iot.CfnTopicRule.TopicRulePayloadProperty(
                    sql="SELECT * FROM 'things/+/pdm/predictions'",
                    actions=[
                        iot.CfnTopicRule.ActionProperty(
                            s3=iot.CfnTopicRule.S3ActionProperty(
                                bucket_name=predictions_bucket.bucket_name,
                                key="predictions/${parse_time('yyyy', timestamp())}/${parse_time('MM', timestamp())}/${parse_time('dd', timestamp())}/${timestamp()}.json",
                                role_arn=iot_rule_role.role_arn,
                            )
                        )
                    ],
                ),
            )
            cdk.CfnOutput(self, "PredictionsBucketName", value=predictions_bucket.bucket_name)

        if enable_cw_metrics:
            iot_rule_role.add_to_policy(iam.PolicyStatement(
                actions=["cloudwatch:PutMetricData"],
                resources=["*"],
            ))

            iot.CfnTopicRule(
                self, "CloudWatchMetricsRule",
                rule_name=f"{project_name.replace('-', '_')}_cw_metrics",
                topic_rule_payload=iot.CfnTopicRule.TopicRulePayloadProperty(
                    sql=f"SELECT predictions.{metric_field} AS metric_value, device_id FROM 'things/+/pdm/predictions'",
                    actions=[
                        iot.CfnTopicRule.ActionProperty(
                            cloudwatch_metric=iot.CfnTopicRule.CloudwatchMetricActionProperty(
                                metric_name=metric_name,
                                metric_namespace="PdM/EdgeInference",
                                metric_value="${metric_value}",
                                metric_unit="None",
                                role_arn=iot_rule_role.role_arn,
                            )
                        )
                    ],
                ),
            )

        if enable_dashboard and enable_cw_metrics:
            dashboard = cloudwatch.Dashboard(
                self, "PdmDashboard",
                dashboard_name=f"{project_name}-dashboard",
            )
            dashboard.add_widgets(
                cloudwatch.GraphWidget(
                    title=f"{metric_name} Over Time",
                    left=[
                        cloudwatch.Metric(
                            namespace="PdM/EdgeInference",
                            metric_name=metric_name,
                            statistic="Average",
                            period=cdk.Duration.minutes(1),
                        )
                    ],
                    width=24,
                ),
            )
            cdk.CfnOutput(
                self, "DashboardURL",
                value=f"https://{self.region}.console.aws.amazon.com/cloudwatch/home?region={self.region}#dashboards:name={project_name}-dashboard",
            )

        # --- Outputs ---
        cdk.CfnOutput(self, "ArtifactsBucketName", value=self.artifacts_bucket.bucket_name)
        cdk.CfnOutput(self, "TokenExchangeRoleName", value=self.token_exchange_role.role_name)
        cdk.CfnOutput(self, "TokenExchangeRoleAliasName", value=self.token_exchange_role_alias.ref)
        cdk.CfnOutput(self, "IoTPolicyName", value=self.iot_policy.policy_name or "")

    def _bool_context(self, key: str) -> bool:
        """Parse a context variable as boolean, handling CLI string overrides."""
        val = self.node.try_get_context(key)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ("true", "1", "yes")
        return False
