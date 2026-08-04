"""CDK Stack: Network infrastructure for PdM edge deployment.

Creates the VPC, subnets, VPC endpoints, and security groups needed by
Greengrass edge devices. Designed to be shared by multiple devices.

Following the security pattern from:
  guidance-for-deploying-ai-agents-to-device-fleets-using-aws-iot-greengrass
"""

import aws_cdk as cdk
from aws_cdk import (
    Stack,
    aws_ec2 as ec2,
)
from constructs import Construct


# IoT VPC endpoints are not available in all AZs. This mapping provides
# a known-good AZ per region. Extend as needed for additional regions.
IOT_ENDPOINT_AZS: dict[str, str] = {
    "us-east-1": "us-east-1b",
    "us-east-2": "us-east-2b",
    "us-west-2": "us-west-2a",
    "eu-west-1": "eu-west-1a",
    "eu-west-2": "eu-west-2a",
    "eu-central-1": "eu-central-1a",
    "ap-northeast-1": "ap-northeast-1a",
    "ap-southeast-1": "ap-southeast-1a",
    "ap-southeast-2": "ap-southeast-2a",
}


class PdmEdgeNetworkStack(Stack):
    """Network layer for edge devices: VPC, VPC endpoints, security groups.

    Exports:
        vpc: The VPC object (for cross-stack reference)
        device_security_group: SG to attach to edge device instances
    """

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        project_name = self.node.try_get_context("project_name") or "pdm-edge"

        # Select an AZ that supports IoT VPC endpoint services
        az = IOT_ENDPOINT_AZS.get(self.region, f"{self.region}a")

        # --- VPC ---
        self.vpc = ec2.Vpc(
            self, "Vpc",
            vpc_name=f"{project_name}-vpc",
            nat_gateways=0,
            availability_zones=[az],
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    map_public_ip_on_launch=True,
                )
            ],
        )

        # --- S3 Gateway Endpoint (free, better performance) ---
        self.vpc.add_gateway_endpoint(
            "S3Endpoint",
            service=ec2.GatewayVpcEndpointAwsService.S3,
        )

        # --- Security Groups ---
        self.device_security_group = ec2.SecurityGroup(
            self, "DeviceSG",
            vpc=self.vpc,
            description="Greengrass edge device - outbound only",
            allow_all_outbound=True,
        )

        vpce_sg = ec2.SecurityGroup(
            self, "VpcEndpointSG",
            vpc=self.vpc,
            description="VPC endpoints - accepts HTTPS and MQTT from devices",
            allow_all_outbound=False,
        )
        vpce_sg.add_ingress_rule(
            self.device_security_group,
            ec2.Port.tcp(443),
            "HTTPS from edge devices",
        )
        vpce_sg.add_ingress_rule(
            self.device_security_group,
            ec2.Port.tcp(8883),
            "MQTT over TLS from edge devices",
        )

        # --- VPC Interface Endpoints ---
        # Matching the guidance-for-deploying-ai-agents-to-device-fleets pattern:
        # SSM, SSMMessages, EC2Messages, IoT Data, IoT Credentials, Greengrass

        self.vpc.add_interface_endpoint(
            "SSMEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.SSM,
            security_groups=[vpce_sg],
        )

        self.vpc.add_interface_endpoint(
            "SSMMessagesEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.SSM_MESSAGES,
            security_groups=[vpce_sg],
        )

        self.vpc.add_interface_endpoint(
            "EC2MessagesEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.EC2_MESSAGES,
            security_groups=[vpce_sg],
        )

        self.vpc.add_interface_endpoint(
            "IoTDataEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService("iot.data"),
            security_groups=[vpce_sg],
            private_dns_enabled=False,
        )

        self.vpc.add_interface_endpoint(
            "IoTCredentialsEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService("iot.credentials"),
            security_groups=[vpce_sg],
            private_dns_enabled=False,
        )

        self.vpc.add_interface_endpoint(
            "GreengrassEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService("greengrass"),
            security_groups=[vpce_sg],
        )

        # --- Outputs ---
        cdk.CfnOutput(self, "VpcId", value=self.vpc.vpc_id)
        cdk.CfnOutput(self, "AvailabilityZone", value=az)
