#!/usr/bin/env bash
# Deploy (or teardown) the PdM edge inference component to AWS IoT Greengrass.
#
# Usage:
#   bash scripts/deploy_edge.sh --model-dir ./fault_prediction/baseline/model
#   bash scripts/deploy_edge.sh --model-dir ./anomaly_detection/baseline/model --region us-east-1
#   bash scripts/deploy_edge.sh --dry-run --model-dir ./fault_prediction/baseline/model
#   bash scripts/deploy_edge.sh --destroy
#
# Prerequisites:
#   - AWS CLI configured with appropriate credentials
#   - CDK CLI installed (npm install -g aws-cdk)
#   - GDK CLI installed (pip install git+https://github.com/aws-greengrass/aws-greengrass-gdk-cli.git@v1.6.2)
#   - CDK stack deployed (cd infrastructure/edge && cdk deploy)
#   - A trained model directory with metadata.json
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EDGE_COMPONENT_DIR="$PROJECT_DIR/edge_component"
INFRA_DIR="$PROJECT_DIR/infrastructure/edge"
NETWORK_STACK="PdmEdgeNetworkStack"
RESOURCES_STACK="PdmEdgeResourcesStack"
DEVICE_STACK="PdmEdgeDemoDeviceStack"
COMPONENT_NAME="com.example.PdmEdgeInference"

# --- Defaults ---
MODE="deploy"
DRY_RUN=false
MODEL_DIR=""
DATA_INPUT="$PROJECT_DIR/data/raw_test.csv"
REGION="${AWS_REGION:-eu-central-1}"
MAX_ROWS=""

# --- Usage ---
usage() {
    cat << 'EOF'
Usage: bash scripts/deploy_edge.sh [OPTIONS]

Modes:
  (default)    Deploy the edge inference component
  --destroy    Tear down all edge infrastructure (network + resources + device)
  --dry-run    Validate prerequisites without executing

Options:
  --model-dir PATH    Path to trained model directory (required for deploy)
  --data-input PATH   Path to raw test CSV (default: data/raw_test.csv)
  --region REGION     AWS region (default: $AWS_REGION or eu-central-1)
  --max-rows N        Cap sensor data rows for demo

Prerequisites:
  aws, cdk, gdk CLIs installed
  CDK stacks deployed: cd infrastructure/edge && cdk deploy --all
  A trained model with metadata.json

CDK Stacks (deployed in order):
  PdmEdgeNetworkStack     VPC, VPC endpoints, security groups
  PdmEdgeResourcesStack   IoT Policy, TES Role, S3 bucket, IoT Rules
  PdmEdgeDemoDeviceStack  EC2 instance with Greengrass (optional demo device)
EOF
}

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --destroy)    MODE="destroy"; shift ;;
        --dry-run)    DRY_RUN=true; shift ;;
        --model-dir)  MODEL_DIR="$2"; shift 2 ;;
        --data-input) DATA_INPUT="$2"; shift 2 ;;
        --region)     REGION="$2"; shift 2 ;;
        --max-rows)   MAX_ROWS="--max-rows $2"; shift 2 ;;
        -h|--help)    usage; exit 0 ;;
        *)            echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Utility functions ---
log()   { echo "[deploy_edge] $*"; }
ok()    { echo "[deploy_edge] ✅ $*"; }
fail()  { echo "[deploy_edge] ❌ $*" >&2; exit 1; }

check_command() {
    if command -v "$1" &>/dev/null; then
        ok "$1 found: $(command -v "$1")"
    else
        fail "$1 not found. Install it first."
    fi
}

get_stack_output() {
    local stack="$1"
    local key="$2"
    aws cloudformation describe-stacks \
        --stack-name "$stack" \
        --region "$REGION" \
        --query "Stacks[0].Outputs[?OutputKey=='$key'].OutputValue" \
        --output text 2>/dev/null
}

# --- Prerequisites check ---
check_prerequisites() {
    log "Checking prerequisites..."
    check_command aws
    check_command cdk
    check_command gdk

    if ! aws sts get-caller-identity &>/dev/null; then
        fail "AWS credentials not configured or expired"
    fi
    ok "AWS credentials valid"

    if [[ "$MODE" == "deploy" ]]; then
        if [[ -z "$MODEL_DIR" ]]; then
            fail "--model-dir is required for deployment"
        fi
        if [[ ! -f "$MODEL_DIR/metadata.json" ]]; then
            fail "No metadata.json in $MODEL_DIR"
        fi
        ok "Model directory: $MODEL_DIR"

        if [[ ! -f "$DATA_INPUT" ]]; then
            fail "Data input not found: $DATA_INPUT"
        fi
        ok "Data input: $DATA_INPUT"
    fi

    # Check CDK stacks are deployed
    local stacks_ok=true
    for stack in "$NETWORK_STACK" "$RESOURCES_STACK" "$DEVICE_STACK"; do
        if ! aws cloudformation describe-stacks --stack-name "$stack" --region "$REGION" &>/dev/null; then
            fail "CDK stack '$stack' not deployed. Run: cd infrastructure/edge && cdk deploy --all"
            stacks_ok=false
        fi
    done
    if $stacks_ok; then
        ok "All CDK stacks deployed"
    fi

    log "All prerequisites met."
}

# --- Deploy ---
do_deploy() {
    check_prerequisites
    if $DRY_RUN; then
        log "Dry run complete — no changes made."
        exit 0
    fi

    local ARTIFACTS_BUCKET THING_GROUP THING_NAME PREDICTION_TOPIC
    ARTIFACTS_BUCKET=$(get_stack_output "$RESOURCES_STACK" "ArtifactsBucketName")
    THING_GROUP=$(get_stack_output "$DEVICE_STACK" "ThingGroup")
    THING_NAME=$(get_stack_output "$DEVICE_STACK" "ThingName")
    PREDICTION_TOPIC=$(get_stack_output "$DEVICE_STACK" "PredictionTopic")

    log "Stack outputs:"
    log "  Artifacts bucket: $ARTIFACTS_BUCKET"
    log "  Thing group: $THING_GROUP"
    log "  Thing name: $THING_NAME"
    log "  Prediction topic: $PREDICTION_TOPIC"

    # Step 1: Copy model into edge component
    log "Copying model to edge component..."
    rm -rf "$EDGE_COMPONENT_DIR/model"
    cp -r "$MODEL_DIR" "$EDGE_COMPONENT_DIR/model"
    ok "Model copied"

    # Step 2: Copy pdm library
    log "Copying pdm library to edge component..."
    rm -rf "$EDGE_COMPONENT_DIR/pdm"
    cp -r "$PROJECT_DIR/pdm" "$EDGE_COMPONENT_DIR/pdm"
    # Replace __init__.py with edge-optimized version (lazy imports)
    cp "$EDGE_COMPONENT_DIR/pdm_init_edge.py" "$EDGE_COMPONENT_DIR/pdm/__init__.py"
    ok "pdm library copied (edge-optimized imports)"

    # Step 3: Extract device sensor data
    log "Extracting device sensor data..."
    PYTHON="${PROJECT_DIR}/.venv/bin/python"
    if [ ! -f "$PYTHON" ]; then
        PYTHON="python"
    fi
    (cd "$PROJECT_DIR" && "$PYTHON" -m pdm.benchmarks.extract_device \
        --input "$DATA_INPUT" \
        --model-dir "$MODEL_DIR" \
        --output "$EDGE_COMPONENT_DIR/data/device_sensors.csv" \
        $MAX_ROWS)
    ok "Sensor data extracted"

    # Step 4: Configure GDK publish bucket
    # GDK naming convention: bucket = "{prefix}-{region}-{account}"
    # CDK creates the bucket with this convention, so we extract the prefix.
    log "Configuring GDK publish settings..."
    local ACCOUNT
    ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
    local GDK_BUCKET_PREFIX="${ARTIFACTS_BUCKET%-${REGION}-${ACCOUNT}}"
    if [[ "$GDK_BUCKET_PREFIX" == "$ARTIFACTS_BUCKET" ]]; then
        # Fallback: use full bucket name as prefix (GDK will check if it exists)
        GDK_BUCKET_PREFIX="$ARTIFACTS_BUCKET"
    fi
    python3 -c "
import json
config = json.load(open('$EDGE_COMPONENT_DIR/gdk-config.json'))
comp = config['component']['$COMPONENT_NAME']
comp['publish']['bucket'] = '$GDK_BUCKET_PREFIX'
comp['publish']['region'] = '$REGION'
json.dump(config, open('$EDGE_COMPONENT_DIR/gdk-config.json', 'w'), indent=2)
"
    ok "GDK configured: prefix=$GDK_BUCKET_PREFIX region=$REGION (full bucket: $ARTIFACTS_BUCKET)"

    # Step 5: Build component
    log "Building Greengrass component..."
    (cd "$EDGE_COMPONENT_DIR" && gdk component build)
    ok "Component built"

    # Step 6: Publish component
    log "Publishing Greengrass component..."
    (cd "$EDGE_COMPONENT_DIR" && gdk component publish --bucket "$ARTIFACTS_BUCKET" --region "$REGION")
    ok "Component published"

    # Step 7: Get published version
    local COMPONENT_VERSION
    COMPONENT_VERSION=$(aws greengrassv2 list-component-versions \
        --arn "arn:aws:greengrass:${REGION}:$(aws sts get-caller-identity --query Account --output text):components:${COMPONENT_NAME}" \
        --region "$REGION" \
        --query "componentVersions[0].componentVersion" \
        --output text 2>/dev/null || echo "")

    if [[ -z "$COMPONENT_VERSION" || "$COMPONENT_VERSION" == "None" ]]; then
        fail "Could not determine published component version"
    fi
    ok "Published version: $COMPONENT_VERSION"

    # Step 8: Create Greengrass deployment
    log "Creating Greengrass deployment to thing group: $THING_GROUP..."
    local TARGET_ARN="arn:aws:iot:${REGION}:$(aws sts get-caller-identity --query Account --output text):thinggroup/${THING_GROUP}"
    local DEPLOYMENT_ID
    DEPLOYMENT_ID=$(aws greengrassv2 create-deployment \
        --target-arn "$TARGET_ARN" \
        --components "{\"${COMPONENT_NAME}\": {\"componentVersion\": \"${COMPONENT_VERSION}\"}}" \
        --region "$REGION" \
        --query "deploymentId" \
        --output text)
    ok "Deployment created: $DEPLOYMENT_ID"

    # Step 9: Wait for deployment
    log "Waiting for deployment to complete..."
    local STATUS="IN_PROGRESS"
    local ATTEMPTS=0
    while [[ "$STATUS" == "IN_PROGRESS" && $ATTEMPTS -lt 30 ]]; do
        sleep 10
        STATUS=$(aws greengrassv2 get-deployment \
            --deployment-id "$DEPLOYMENT_ID" \
            --region "$REGION" \
            --query "deploymentStatus" \
            --output text 2>/dev/null || echo "IN_PROGRESS")
        ATTEMPTS=$((ATTEMPTS + 1))
        log "  Status: $STATUS (attempt $ATTEMPTS/30)"
    done

    if [[ "$STATUS" == "COMPLETED" ]]; then
        ok "Deployment succeeded"
    else
        fail "Deployment did not complete (status: $STATUS). Check Greengrass logs on the device."
    fi

    # Step 10: Print summary
    echo ""
    echo "=========================================="
    echo "  Edge Deployment Complete"
    echo "=========================================="
    echo ""
    echo "  Thing name:       $THING_NAME"
    echo "  Component:        $COMPONENT_NAME v$COMPONENT_VERSION"
    echo "  Prediction topic: $PREDICTION_TOPIC"
    echo ""
    echo "  To view predictions live:"
    echo "    AWS Console → IoT Core → MQTT test client"
    echo "    Subscribe to: $PREDICTION_TOPIC"
    echo ""
    echo "  To check device logs:"
    echo "    Connect via SSM Session Manager, then:"
    echo "    sudo tail -f /greengrass/v2/logs/$COMPONENT_NAME.log"
    echo ""
    echo "  To tear down:"
    echo "    bash scripts/deploy_edge.sh --destroy"
    echo "=========================================="
}

# --- Destroy ---
do_destroy() {
    log "Tearing down edge deployment..."

    if $DRY_RUN; then
        log "Dry run: would destroy all edge stacks in $REGION"
        exit 0
    fi

    # Remove Greengrass deployment (reset devices to no components)
    local THING_GROUP
    THING_GROUP=$(get_stack_output "$DEVICE_STACK" "ThingGroup" || echo "")
    if [[ -n "$THING_GROUP" ]]; then
        local TARGET_ARN="arn:aws:iot:${REGION}:$(aws sts get-caller-identity --query Account --output text):thinggroup/${THING_GROUP}"
        log "Removing Greengrass deployment from $THING_GROUP..."
        aws greengrassv2 create-deployment \
            --target-arn "$TARGET_ARN" \
            --components "{}" \
            --region "$REGION" \
            --output text 2>/dev/null || true
        ok "Deployment removed (components cleared)"
    fi

    # Destroy stacks in reverse dependency order
    log "Destroying CDK stacks (device → resources → network)..."
    (cd "$INFRA_DIR" && cdk destroy "$DEVICE_STACK" --force 2>/dev/null || true)
    ok "Demo device stack destroyed"
    (cd "$INFRA_DIR" && cdk destroy "$RESOURCES_STACK" --force 2>/dev/null || true)
    ok "Resources stack destroyed"
    (cd "$INFRA_DIR" && cdk destroy "$NETWORK_STACK" --force 2>/dev/null || true)
    ok "Network stack destroyed"

    # Clean up local edge component artifacts
    rm -rf "$EDGE_COMPONENT_DIR/model" "$EDGE_COMPONENT_DIR/pdm" "$EDGE_COMPONENT_DIR/greengrass-build" "$EDGE_COMPONENT_DIR/zip-build"
    rm -f "$EDGE_COMPONENT_DIR/data/device_sensors.csv"
    ok "Local artifacts cleaned"

    echo ""
    log "Teardown complete. All edge resources removed."
}

# --- Main ---
case "$MODE" in
    deploy)  do_deploy ;;
    destroy) do_destroy ;;
    *)       fail "Unknown mode: $MODE" ;;
esac
