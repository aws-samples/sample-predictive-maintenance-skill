FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    typing_extensions \
    "autogluon.tabular[lightgbm]==1.5.0" \
    autogluon.core==1.5.0 \
    flask \
    gunicorn \
    boto3 \
    tqdm \
    pyarrow

COPY serve.py /opt/program/serve.py
COPY inference.py /opt/program/inference.py

ENV PYTHONUNBUFFERED=TRUE
ENV PATH="/opt/program:${PATH}"

WORKDIR /opt/program

RUN useradd -r -s /bin/false appuser && chown -R appuser /opt/program
USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/ping')" || exit 1

ENTRYPOINT ["python", "serve.py"]
