#!/usr/bin/env python3
"""CDK app for PdM edge deployment infrastructure.

Three independent stacks with clear responsibilities:

  1. PdmEdgeNetworkStack — VPC, VPC endpoints, security groups
     (the secure network layer for edge device connectivity)

  2. PdmEdgeResourcesStack — IoT Policy, TES Role, S3 bucket, IoT Rules
     (persistent IoT/Greengrass identity and observability)

  3. PdmEdgeDemoDeviceStack — EC2 instance with Greengrass auto-provisioning
     (optional demo device — destroy to stop costs, redeploy anytime)

Deploy all:     cdk deploy --all
Deploy infra:   cdk deploy PdmEdgeNetworkStack PdmEdgeResourcesStack
Demo device:    cdk deploy PdmEdgeDemoDeviceStack
Destroy device: cdk destroy PdmEdgeDemoDeviceStack
Destroy all:    cdk destroy --all
"""
import os

import aws_cdk as cdk

from network_stack import PdmEdgeNetworkStack
from resources_stack import PdmEdgeResourcesStack
from demo_device_stack import PdmEdgeDemoDeviceStack

app = cdk.App()

env = cdk.Environment(
    account=app.node.try_get_context("account") or os.environ.get("CDK_DEFAULT_ACCOUNT"),
    region=app.node.try_get_context("region") or os.environ.get("CDK_DEFAULT_REGION", "eu-central-1"),
)

# Stack 1: Network
network = PdmEdgeNetworkStack(app, "PdmEdgeNetworkStack", env=env)

# Stack 2: IoT/Greengrass Resources (independent of network)
resources = PdmEdgeResourcesStack(app, "PdmEdgeResourcesStack", env=env)

# Stack 3: Demo Device (depends on both network and resources)
demo_device = PdmEdgeDemoDeviceStack(
    app, "PdmEdgeDemoDeviceStack",
    network=network,
    resources=resources,
    env=env,
)
demo_device.add_dependency(network)
demo_device.add_dependency(resources)

app.synth()
