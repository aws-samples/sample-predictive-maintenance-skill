"""CDK Stack: Demo edge device (EC2 instance running Greengrass Nucleus).

This stack is OPTIONAL — only needed for testing/demo when you don't have
a physical edge device. Deploy it to simulate a Greengrass core device on EC2.

Depends on:
  - PdmEdgeNetworkStack (VPC, security group)
  - PdmEdgeResourcesStack (IoT policy, TES role names)

Can be destroyed independently to stop EC2 costs while keeping IoT resources.

Following the EC2 provisioning pattern from:
  IoTResources.yaml (guidance-for-deploying-ai-agents-to-device-fleets)
"""

import os
import string
from pathlib import Path

import aws_cdk as cdk
from aws_cdk import (
    Stack,
    aws_ec2 as ec2,
    aws_iam as iam,
)
from constructs import Construct

from network_stack import PdmEdgeNetworkStack
from resources_stack import PdmEdgeResourcesStack

USERDATA_DIR = Path(os.path.dirname(__file__)) / "userdata"


class PdmEdgeDemoDeviceStack(Stack):
    """EC2 instance that auto-provisions as a Greengrass core device.

    This stack creates:
      - EC2 instance role (SSM + IoT provisioning permissions)
      - EC2 instance (t3.small, AL2023) with Greengrass auto-provisioning UserData
      - Explicit dependency on IGW attachment before instance launch

    The instance provisions itself as an IoT Thing and Greengrass core device
    on first boot via UserData.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        network: PdmEdgeNetworkStack,
        resources: PdmEdgeResourcesStack,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        project_name = self.node.try_get_context("project_name") or "pdm-edge"

        # --- EC2 Instance Role ---
        ec2_role = iam.Role(
            self, "EC2Role",
            role_name=f"{project_name}-ec2-role",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore"),
            ],
        )

        # IoT provisioning permissions (one-time, during first boot)
        ec2_role.add_to_policy(iam.PolicyStatement(
            actions=[
                "iot:CreateThing",
                "iot:CreateKeysAndCertificate",
                "iot:CreatePolicy",
                "iot:AttachThingPrincipal",
                "iot:AttachPrincipalPolicy",
                "iot:AttachPolicy",
                "iot:DescribeEndpoint",
                "iot:DescribeThing",
                "iot:DescribeThingGroup",
                "iot:DescribeRoleAlias",
                "iot:GetPolicy",
                "iot:ListThingPrincipals",
                "iot:AddThingToThingGroup",
                "iot:CreateThingGroup",
            ],
            resources=["*"],
        ))

        # Greengrass device registration
        ec2_role.add_to_policy(iam.PolicyStatement(
            actions=[
                "greengrass:GetCoreDevice",
                "greengrass:ListCoreDevices",
                "greengrass:TagResource",
                "greengrass:CreateDeployment",
            ],
            resources=["*"],
        ))

        # Allow IAM operations needed by Greengrass provisioner
        # (creates/attaches TES role policy during --provision true)
        ec2_role.add_to_policy(iam.PolicyStatement(
            actions=["iam:GetPolicy", "iam:CreatePolicy", "iam:AttachRolePolicy"],
            resources=[
                resources.token_exchange_role.role_arn,
                f"arn:aws:iam::{self.account}:policy/{project_name}-GGV2TokenExchangeRoleAccess",
                f"arn:aws:iam::{self.account}:policy/{resources.token_exchange_role.role_name}Access",
            ],
        ))
        ec2_role.add_to_policy(iam.PolicyStatement(
            actions=["iam:GetRole", "iam:CreateRole", "iam:PassRole"],
            resources=[resources.token_exchange_role.role_arn],
        ))

        # --- EC2 Instance ---
        thing_name = f"{project_name}-device"
        thing_group = f"{project_name}-group"

        user_data = ec2.UserData.for_linux()
        userdata_script = (USERDATA_DIR / "provision_greengrass.sh").read_text()
        userdata_rendered = string.Template(userdata_script).substitute(
            AWS_REGION=self.region,
            THING_NAME=thing_name,
            THING_GROUP=thing_group,
            IOT_POLICY_NAME=resources.iot_policy.policy_name,
            TES_ROLE_NAME=resources.token_exchange_role.role_name,
            TES_ROLE_ALIAS=resources.token_exchange_role_alias.ref,
        )
        user_data.add_commands(userdata_rendered)

        instance = ec2.Instance(
            self, "EdgeDevice",
            instance_type=ec2.InstanceType("t3.small"),
            machine_image=ec2.MachineImage.latest_amazon_linux2023(),
            vpc=network.vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            security_group=network.device_security_group,
            role=ec2_role,
            user_data=user_data,
            block_devices=[
                ec2.BlockDevice(
                    device_name="/dev/xvda",
                    volume=ec2.BlockDeviceVolume.ebs(
                        volume_size=20,
                        volume_type=ec2.EbsDeviceVolumeType.GP3,
                        encrypted=True,
                        delete_on_termination=True,
                    ),
                )
            ],
        )

        # Ensure IGW is attached before instance launches (matching guidance repo)
        igw_attachment = network.vpc.node.find_child("VPCGW")
        if igw_attachment:
            instance.node.add_dependency(igw_attachment)

        # --- Outputs ---
        cdk.CfnOutput(self, "InstanceId", value=instance.instance_id)
        cdk.CfnOutput(self, "ThingName", value=thing_name)
        cdk.CfnOutput(self, "ThingGroup", value=thing_group)
        cdk.CfnOutput(
            self, "PredictionTopic",
            value=f"things/{thing_name}/pdm/predictions",
        )
