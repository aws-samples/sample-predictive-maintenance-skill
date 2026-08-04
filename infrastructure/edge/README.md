# PdM Edge Deployment Infrastructure (CDK)

Deploys infrastructure for running predictive maintenance inference at the edge using AWS IoT Greengrass. Designed as a production-ready foundation with an optional demo device for testing.

```
┌─────────────────────────────────────────────────────────────────┐
│ PdmEdgeNetworkStack        │ PdmEdgeResourcesStack              │
│                            │                                     │
│ VPC + Public Subnet        │ IoT Policy                          │
│ VPC Endpoints (6):         │ Token Exchange Role + Alias         │
│   SSM, SSMMessages,        │ S3 Artifacts Bucket                 │
│   EC2Messages, IoT Data,   │ IoT Rules (optional):               │
│   IoT Credentials,         │   → S3 prediction logging           │
│   Greengrass               │   → CloudWatch metrics              │
│ S3 Gateway Endpoint        │ CloudWatch Dashboard (optional)     │
│ Security Groups            │                                     │
└────────────┬───────────────┴──────────────────┬─────────────────┘
             │                                   │
             └───────────────┬───────────────────┘
                             ▼
              ┌──────────────────────────────┐
              │ PdmEdgeDemoDeviceStack       │
              │ (OPTIONAL — for testing)     │
              │                              │
              │ EC2 (t3.small, AL2023)       │
              │ Greengrass auto-provisioning │
              │ SSM-managed (no SSH)         │
              └──────────────────────────────┘
```

## 3-Stack Architecture

| Stack | Purpose | Lifecycle | Cost when idle |
|-------|---------|-----------|----------------|
| `PdmEdgeNetworkStack` | VPC, VPC endpoints, security groups | Persistent network | ~$44/month (6 interface endpoints) |
| `PdmEdgeResourcesStack` | IoT Policy, TES Role, S3, IoT Rules | Persistent identity | ~$0 (only S3 storage) |
| `PdmEdgeDemoDeviceStack` | EC2 instance with Greengrass | **Optional** demo device | ~$15/month (t3.small) |

**Key insight:** The demo device stack can be destroyed independently to stop EC2 costs. The IoT resources remain intact — ready for a real physical device or a new demo instance at any time.

For production with physical edge devices, deploy only `PdmEdgeNetworkStack` + `PdmEdgeResourcesStack`. The demo device stack is for testing without hardware.

## Prerequisites

- AWS CDK CLI installed (`npm install -g aws-cdk`)
- CDK bootstrapped in target account/region (`cdk bootstrap`)
- Python 3.11+ with `pip`

## Configuration

Edit `cdk.json` context variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `project_name` | Prefix for all resource names | `pdm-edge` |
| `enable_s3_logging` | Store predictions in S3 via IoT Rule | `false` |
| `enable_cloudwatch_metrics` | Publish prediction metrics to CloudWatch | `true` |
| `enable_dashboard` | Create CloudWatch Dashboard | `true` |
| `metric_field` | Prediction field to extract as CloudWatch metric | `machine_failure_proba` |
| `metric_name` | CloudWatch metric display name | `FailureProbability` |

## Deploy

### All stacks (including demo device)

```bash
cd infrastructure/edge/
pip install -r requirements.txt
cdk bootstrap  # first time only
cdk deploy --all -c account=$(aws sts get-caller-identity --query Account --output text) \
                 -c region=${AWS_REGION:-eu-central-1}
```

### Production only (no demo device)

```bash
cdk deploy PdmEdgeNetworkStack PdmEdgeResourcesStack \
    -c account=$(aws sts get-caller-identity --query Account --output text) \
    -c region=${AWS_REGION:-eu-central-1}
```

### Add demo device later

```bash
cdk deploy PdmEdgeDemoDeviceStack \
    -c account=$(aws sts get-caller-identity --query Account --output text) \
    -c region=${AWS_REGION:-eu-central-1}
```

## Verify

After deployment with demo device:
1. Wait 3-5 minutes for UserData to complete Greengrass provisioning
2. Check EC2 instance is running: AWS Console → EC2 → Instances
3. Connect via SSM Session Manager (no SSH needed):
   ```bash
   aws ssm start-session --target <instance-id>
   ```
4. Verify Greengrass: `sudo systemctl status greengrass`
5. Check IoT Thing: AWS Console → IoT Core → Things → `pdm-edge-device`

## Teardown

### Stop demo device costs only (keep IoT resources for reuse)

```bash
cdk destroy PdmEdgeDemoDeviceStack
```

### Destroy everything

```bash
cdk destroy --all
```

Or use the deploy script:
```bash
bash scripts/deploy_edge.sh --destroy
```

This removes all 3 stacks in reverse dependency order. The Greengrass component version in the registry is NOT deleted (it's just a version record, no ongoing cost).

## VPC Endpoints

The network stack creates 6 VPC Interface Endpoints + 1 Gateway Endpoint, matching the [guidance-for-deploying-ai-agents-to-device-fleets-using-aws-iot-greengrass](https://github.com/aws-solutions-library-samples/guidance-for-deploying-ai-agents-to-device-fleets-using-aws-iot-greengrass) security pattern:

| Endpoint | Purpose |
|----------|---------|
| SSM | Session Manager connectivity (no SSH) |
| SSM Messages | SSM session data channel |
| EC2 Messages | SSM agent communication |
| IoT Data | MQTT message delivery |
| IoT Credentials | Token Exchange Service (device → IAM) |
| Greengrass | Component deployment and management |
| S3 (Gateway) | Component artifact downloads (free) |

**AZ Compatibility:** IoT VPC endpoints are not available in all AZs. The network stack uses a region-aware AZ mapping to select a compatible AZ (e.g., `us-east-1b` instead of `us-east-1a`).

## Customizing the CloudWatch Metric

The IoT Rule extracts a configurable field from prediction payloads:

```sql
SELECT predictions.<metric_field> AS metric_value, device_id
FROM 'things/+/pdm/predictions'
```

### Configuration by Formulation

| Formulation | `metric_field` | `metric_name` |
|-------------|---------------|---------------|
| Classification (binary) | `machine_failure_proba` | `FailureProbability` |
| Classification (multi-label) | `<label>_proba` | `<Label>Probability` |
| Anomaly Detection | `anomaly_score` | `AnomalyScore` |
| RUL | `RUL_pred` | `PredictedRUL` |

**CLI override:**
```bash
cdk deploy --all -c metric_field=anomaly_score -c metric_name=AnomalyScore
```

## Connecting Physical Devices

For production deployments targeting real edge hardware (not the demo EC2):

1. Deploy `PdmEdgeNetworkStack` + `PdmEdgeResourcesStack` only
2. On the physical device, install Greengrass Nucleus and provision using the IoT Policy and TES Role from the stack outputs:
   ```bash
   # Get values from CDK outputs
   IOT_POLICY=$(aws cloudformation describe-stacks --stack-name PdmEdgeResourcesStack \
     --query "Stacks[0].Outputs[?OutputKey=='IoTPolicyName'].OutputValue" --output text)
   TES_ROLE=$(aws cloudformation describe-stacks --stack-name PdmEdgeResourcesStack \
     --query "Stacks[0].Outputs[?OutputKey=='TokenExchangeRoleName'].OutputValue" --output text)
   TES_ALIAS=$(aws cloudformation describe-stacks --stack-name PdmEdgeResourcesStack \
     --query "Stacks[0].Outputs[?OutputKey=='TokenExchangeRoleAliasName'].OutputValue" --output text)

   # Provision device
   java -Droot="/greengrass/v2" -jar GreengrassInstaller/lib/Greengrass.jar \
     --aws-region $REGION \
     --thing-name my-physical-device \
     --thing-group-name pdm-edge-group \
     --thing-policy-name $IOT_POLICY \
     --tes-role-name $TES_ROLE \
     --tes-role-alias-name $TES_ALIAS \
     --provision true --setup-system-service true
   ```
3. Deploy the PdM component using `bash scripts/deploy_edge.sh`
