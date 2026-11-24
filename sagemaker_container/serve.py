"""Minimal SageMaker-compatible Flask server.

/ping returns 200 immediately (no model loading — avoids health check timeout).
/invocations lazy-loads the model on first call, then serves predictions.
"""
import json
import sys
sys.path.insert(0, "/opt/ml/model/code")
sys.path.insert(0, "/opt/program")

from flask import Flask, request, Response
from inference import model_fn, input_fn, predict_fn, output_fn
from pdm.deployment.serving_io import FeatureSchemaError

app = Flask(__name__)
model = None


@app.route("/ping", methods=["GET"])
def ping():
    return Response(response="", status=200)


@app.route("/invocations", methods=["POST"])
def invocations():
    global model
    # Validate user input *before* cold-loading a potentially large model.
    # Prediction/model failures remain server errors; malformed requests are 400.
    try:
        data = input_fn(request.get_data(), request.content_type)
    except ValueError as exc:
        return Response(
            response=json.dumps({"error": str(exc)}),
            status=400,
            mimetype="application/json",
        )
    if model is None:
        model = model_fn("/opt/ml/model")
    try:
        result = predict_fn(data, model)
    except FeatureSchemaError as exc:
        return Response(
            response=json.dumps({"error": str(exc)}),
            status=400,
            mimetype="application/json",
        )
    return Response(
        response=output_fn(result, "application/json"),
        status=200,
        mimetype="application/json",
    )


if __name__ == "__main__":
    import os
    # SageMaker requires 0.0.0.0 to receive traffic from the load balancer
    host = os.environ.get("SAGEMAKER_BIND_TO_PORT", "0.0.0.0")  # nosec B104 - required by SageMaker
    app.run(host=host, port=8080)
