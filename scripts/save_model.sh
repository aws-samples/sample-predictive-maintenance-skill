#!/usr/bin/env bash
# Save trained model artifacts to S3 following the artifact contract.
# See references/artifacts.md for the full specification.
#
# Layout: s3://<bucket>/<YYYYMMDD_HHMM>/{dataset,training,model,inference,infrastructure,README.md}
set -euo pipefail

if [ -z "${1:-}" ]; then
  echo "Usage: bash save_model.sh <s3-bucket-name> [model-dir]"
  echo ""
  echo "  s3-bucket-name   Target S3 bucket (output bucket)"
  echo "  model-dir        Path to winning experiment dir (default: auto-detect)"
  echo ""
  echo "Artifacts saved:"
  echo "  dataset/       raw_dataset.py + raw CSVs"
  echo "  training/      runtime.py + train_config.json"
  echo "  model/         ag_model/ + metadata.json + metrics.json"
  echo "  inference/     inference.py + Dockerfile + serve.py"
  echo "  infrastructure/ CDK stack for deployment"
  exit 1
fi

BUCKET="$1"
TIMESTAMP=$(date +%Y%m%d_%H%M)
DEST="s3://${BUCKET}/${TIMESTAMP}"

# Resolve SKILL_DIR: first check if script is inside the skill, then search from project root
_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -d "${_SCRIPT_DIR}/../sagemaker_container" ]; then
  # Script is running from within the skill (e.g., .kiro/skills/predictive-maintenance/scripts/)
  SKILL_DIR="$(cd "${_SCRIPT_DIR}/.." && pwd)"
elif [ -d ".kiro/skills/predictive-maintenance/sagemaker_container" ]; then
  # Script is at project root, resolve relative to cwd
  SKILL_DIR="$(cd .kiro/skills/predictive-maintenance && pwd)"
else
  echo "⚠️  Cannot find skill directory (sagemaker_container). Inference and infrastructure artifacts will be skipped."
  SKILL_DIR=""
fi

# Auto-detect winning experiment directory
if [ -n "${2:-}" ]; then
  EXPERIMENT_DIR="$2"
else
  # Find newest metrics.json under fault_prediction/
  METRICS_FILE=$(find fault_prediction -path "*/model/metrics.json" -exec stat -f '%m %N' {} \; 2>/dev/null | sort -rn | head -1 | awk '{print $2}')
  if [ -z "$METRICS_FILE" ]; then
    echo "❌ No model found. Run training first."
    exit 1
  fi
  EXPERIMENT_DIR=$(dirname "$(dirname "$METRICS_FILE")")
fi

MODEL_DIR="$EXPERIMENT_DIR/model"
RUNTIME_PY="$EXPERIMENT_DIR/runtime.py"

# Validate required files exist
for f in "$MODEL_DIR/metadata.json" "$MODEL_DIR/metrics.json" "$RUNTIME_PY"; do
  [ -f "$f" ] || { echo "❌ Missing: $f"; exit 1; }
done

echo "=== Artifact Save ==="
echo "  Destination:  ${DEST}/"
echo "  Experiment:   ${EXPERIMENT_DIR}"
echo "  Model size:   $(du -sh "$MODEL_DIR" | cut -f1)"
echo ""

# --- 1. Dataset Generation ---
echo "[1/5] Dataset..."
[ -f pdm/raw_dataset.py ] && aws s3 cp pdm/raw_dataset.py "${DEST}/dataset/raw_dataset.py" --quiet
[ -f data/raw_train.csv ] && aws s3 cp data/raw_train.csv "${DEST}/dataset/raw_train.csv" --quiet
[ -f data/raw_test.csv ] && aws s3 cp data/raw_test.csv "${DEST}/dataset/raw_test.csv" --quiet

# --- 2. Training ---
echo "[2/5] Training..."
aws s3 cp "$RUNTIME_PY" "${DEST}/training/runtime.py" --quiet
# Generate train_config.json from model config
if [ -f "$MODEL_DIR/config.json" ]; then
  aws s3 cp "$MODEL_DIR/config.json" "${DEST}/training/train_config.json" --quiet
fi
# Copy pdm library (small, needed by runtime.py at inference time)
aws s3 sync pdm/ "${DEST}/training/pdm/" --exclude "__pycache__/*" --exclude "*.pyc" --quiet

# --- 3. Model (inference binaries) ---
echo "[3/5] Model..."
aws s3 cp "$MODEL_DIR/metadata.json" "${DEST}/model/metadata.json" --quiet
aws s3 cp "$MODEL_DIR/metrics.json" "${DEST}/model/metrics.json" --quiet
[ -f "$MODEL_DIR/baseline_stats.json" ] && aws s3 cp "$MODEL_DIR/baseline_stats.json" "${DEST}/model/baseline_stats.json" --quiet
aws s3 sync "$MODEL_DIR/ag_model/" "${DEST}/model/ag_model/" \
  --exclude "*/utils/oof.pkl" \
  --quiet

# --- 4. Inference runtime ---
echo "[4/5] Inference..."
if [ -n "$SKILL_DIR" ] && [ -d "${SKILL_DIR}/sagemaker_container" ]; then
  aws s3 cp "${SKILL_DIR}/sagemaker_container/inference.py" "${DEST}/inference/inference.py" --quiet
  aws s3 cp "${SKILL_DIR}/sagemaker_container/serve.py" "${DEST}/inference/serve.py" --quiet
  aws s3 cp "${SKILL_DIR}/sagemaker_container/Dockerfile" "${DEST}/inference/Dockerfile" --quiet
  aws s3 cp "${SKILL_DIR}/sagemaker_container/requirements.txt" "${DEST}/inference/requirements.txt" --quiet
else
  echo "  ⚠️  Skipped (skill directory not found)"
fi

# --- 5. Infrastructure ---
echo "[5/5] Infrastructure..."
if [ -n "$SKILL_DIR" ] && [ -d "${SKILL_DIR}/infrastructure" ]; then
  for f in app.py batch_inference_stack.py cdk.json requirements.txt; do
    [ -f "${SKILL_DIR}/infrastructure/$f" ] && aws s3 cp "${SKILL_DIR}/infrastructure/$f" "${DEST}/infrastructure/$f" --quiet
  done
  [ -d "${SKILL_DIR}/infrastructure/lambda" ] && aws s3 sync "${SKILL_DIR}/infrastructure/lambda/" "${DEST}/infrastructure/lambda/" --quiet
else
  # Fall back to local infrastructure/ if it was copied to project root
  if [ -d "infrastructure" ]; then
    for f in app.py batch_inference_stack.py cdk.json requirements.txt; do
      [ -f "infrastructure/$f" ] && aws s3 cp "infrastructure/$f" "${DEST}/infrastructure/$f" --quiet
    done
    [ -d "infrastructure/lambda" ] && aws s3 sync "infrastructure/lambda/" "${DEST}/infrastructure/lambda/" --quiet
  else
    echo "  ⚠️  Skipped (infrastructure directory not found)"
  fi
fi

# --- README ---
[ -f README.md ] && aws s3 cp README.md "${DEST}/README.md" --quiet

echo ""
echo "✅ Saved to ${DEST}/"
echo ""
echo "Artifact manifest:"
echo "  ${DEST}/dataset/          — raw_dataset.py + CSVs"
echo "  ${DEST}/training/         — runtime.py + config"
echo "  ${DEST}/model/            — AutoGluon binaries + metadata"
echo "  ${DEST}/inference/        — Container + inference handler"
echo "  ${DEST}/infrastructure/   — CDK stack"
