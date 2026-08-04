# PdM Infrastructure

This directory contains independent CDK apps for each deployment mode. Each subdirectory is a self-contained CDK application with its own `app.py`, `cdk.json`, and `requirements.txt`.

## Layout

```
infrastructure/
├── batch/          # Phase 8B: Daily batch inference (EventBridge + Lambda + SageMaker)
└── edge/           # Phase 8C: Edge deployment (IoT Core + Greengrass + EC2)
```

## Deploying

Each app is deployed independently:

```bash
# Batch inference pipeline
cd infrastructure/batch/
pip install -r requirements.txt
cdk deploy

# Edge deployment (IoT Greengrass on EC2)
cd infrastructure/edge/
pip install -r requirements.txt
cdk deploy
```

## Why Separate Apps

- **Independent lifecycles** — update batch schedule without touching edge devices, and vice versa
- **Independent teardown** — `cdk destroy` in one doesn't affect the other
- **Different AWS services** — batch uses SageMaker/EventBridge; edge uses IoT Core/Greengrass/EC2
- **Different audiences** — batch is for fleet-wide daily scoring; edge is for single-device real-time inference
