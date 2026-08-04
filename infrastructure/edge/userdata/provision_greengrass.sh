#!/bin/bash
# Provisions an EC2 instance as a Greengrass V2 core device.
#
# Template variables (injected by CDK at synth time):
#   ${AWS_REGION}         - AWS region
#   ${THING_NAME}         - IoT Thing name for this device
#   ${THING_GROUP}        - IoT Thing Group name
#   ${IOT_POLICY_NAME}    - IoT Policy name (pre-created by CDK)
#   ${TES_ROLE_NAME}      - Token Exchange Service role name
#   ${TES_ROLE_ALIAS}     - Token Exchange Service role alias name
set -ex

echo "=== Installing dependencies ==="
dnf update -y
dnf install -y java-11-amazon-corretto-headless python3.11 unzip

echo "=== Creating Greengrass user ==="
useradd --system --create-home ggc_user || true
groupadd --system ggc_group || true

# Configure sudo for Greengrass — Nucleus needs to run component lifecycle
# scripts as ggc_user:ggc_group via sudo
echo "root    ALL=(ALL:ALL) ALL" >> /etc/sudoers

sudo -u ggc_user python3.11 -m venv /home/ggc_user/.venv

echo "=== Downloading Greengrass Nucleus ==="
cd /tmp
curl -s https://d2s8p88vqu9w66.cloudfront.net/releases/greengrass-nucleus-latest.zip -o greengrass-nucleus-latest.zip
unzip -o greengrass-nucleus-latest.zip -d GreengrassInstaller

echo "=== Writing Nucleus config ==="
cat > /tmp/gg-init-config.yaml << 'EOF'
services:
  aws.greengrass.Nucleus:
    configuration:
      interpolateComponentConfiguration: true
EOF

echo "=== Provisioning Greengrass ==="
java -Droot="/greengrass/v2" -Dlog.store=FILE \
  -jar /tmp/GreengrassInstaller/lib/Greengrass.jar \
  --aws-region ${AWS_REGION} \
  --thing-name ${THING_NAME} \
  --thing-group-name ${THING_GROUP} \
  --thing-policy-name ${IOT_POLICY_NAME} \
  --tes-role-name ${TES_ROLE_NAME} \
  --tes-role-alias-name ${TES_ROLE_ALIAS} \
  --component-default-user ggc_user:ggc_group \
  --provision true \
  --setup-system-service true \
  --deploy-dev-tools true \
  --init-config /tmp/gg-init-config.yaml

echo "=== Greengrass provisioning complete ==="
