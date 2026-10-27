# SageMaker real-time endpoint and batch Processing Job image.
#
# Installs the slim pdm library (pdm-project[serve]) at the exact versions in
# uv.lock, i.e. the same Python, AutoGluon and scikit-learn the model was
# trained with. Build from the project root, the directory holding
# pyproject.toml, uv.lock and pdm/ (the skill repo, or a project set up by
# scripts/setup.sh with sagemaker_container/ copied in):
#
#   docker build --platform linux/amd64 -f sagemaker_container/Dockerfile -t pdm-inference .
#
# sagemaker_container/Dockerfile.dockerignore limits the build context to the
# files below, so data/ and .venv/ are never sent to the Docker daemon.

# Must match .python-version (AutoGluon refuses to load across Python minors).
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# uv is only used to export and install the locked requirements.
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /usr/local/bin/uv

WORKDIR /opt/pdm
COPY pyproject.toml uv.lock ./
# Locked, hash-checked dependencies of the serving install — no training-only
# packages (shap, optuna, imbalanced-learn, lifelines, ...).
RUN uv export --frozen --no-dev --no-emit-project --extra serve --quiet -o requirements.txt \
    && uv pip install --system --no-cache --require-hashes -r requirements.txt

# The pdm library itself; its dependencies are already installed above.
COPY pdm ./pdm
RUN uv pip install --system --no-cache --no-deps . && rm -rf build ./*.egg-info

COPY sagemaker_container/serve.py sagemaker_container/inference.py /opt/program/

ENV PYTHONUNBUFFERED=TRUE
ENV PATH="/opt/program:${PATH}"

WORKDIR /opt/program

RUN useradd -r -s /bin/false appuser && chown -R appuser /opt/program
USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/ping')" || exit 1

ENTRYPOINT ["python", "serve.py"]
