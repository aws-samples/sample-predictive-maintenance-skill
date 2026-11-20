"""AWS Solutions usage tracking (Solution ID SO0356).

Checks that the solution token is in the User-Agent of AWS SDK calls, that the
copies of the ID/version stay consistent, and that CDK stack descriptions follow
the ``(SOxxxx[suffix]) Name, vX.Y.Z`` convention.
"""
import importlib.util
import re
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SOLUTION_ID = "SO0356"
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
SOLUTION_UA = f"AWSSOLUTION/{SOLUTION_ID}/v{VERSION}"

# Captures the User-Agent of a fully built request, then aborts it: no
# credentials, no network, no billable call.
_CAPTURE = textwrap.dedent(
    """
    import os, boto3
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "x")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "x")

    class Stop(Exception):
        pass

    def captured_user_agent(client, service, call):
        out = {}
        def grab(request, **kwargs):
            out["ua"] = request.headers["User-Agent"].decode()
            raise Stop()
        client.meta.events.register(f"before-send.{service}.*", grab)
        try:
            call()
        except Stop:
            pass
        finally:
            client.meta.events.unregister(f"before-send.{service}.*", grab)
        return out["ua"]

    s3 = boto3.client("s3", region_name="us-east-1")
    first = captured_user_agent(s3, "s3", lambda: s3.list_buckets())
    second = captured_user_agent(s3, "s3", lambda: s3.list_buckets())
    sm = boto3.session.Session(region_name="us-east-1").client("sagemaker")
    third = captured_user_agent(sm, "sagemaker", lambda: sm.list_training_jobs())
    print("\\n".join([first, second, third]))
    """
)


def _run(preamble: str, env_extra: dict | None = None) -> list[str]:
    import os

    env = {k: v for k, v in os.environ.items() if k != "USER_AGENT_STRING"}
    env.update(env_extra or {})
    result = subprocess.run(
        [sys.executable, "-c", preamble + _CAPTURE],
        capture_output=True, text=True, env=env, cwd=ROOT, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip().splitlines()[-3:]


def _count(ua: str) -> int:
    return ua.split(" ").count(SOLUTION_UA)


@pytest.fixture(scope="module")
def _botocore():
    pytest.importorskip("boto3")


def test_pdm_hook_tags_every_client(_botocore):
    """`import pdm` tags S3 and SageMaker clients, exactly once, also on repeat calls."""
    for ua in _run("import pdm\n"):
        assert _count(ua) == 1, ua


def test_pdm_hook_honours_user_agent_string_env(_botocore):
    for ua in _run("import pdm\n", {"USER_AGENT_STRING": SOLUTION_UA}):
        assert _count(ua) == 1, ua


def test_lambda_hook_tags_every_client(_botocore):
    """The Lambda copy of the hook tags clients when USER_AGENT_STRING is set."""
    pre = f"import sys; sys.path.insert(0, {str(ROOT / 'infrastructure/batch/lambda')!r})\nimport solution_user_agent\n"
    for ua in _run(pre, {"USER_AGENT_STRING": SOLUTION_UA}):
        assert _count(ua) == 1, ua


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_solution_identifiers_are_consistent():
    batch = _load(ROOT / "infrastructure/batch/solution.py", "batch_solution")
    edge = _load(ROOT / "infrastructure/edge/solution.py", "edge_solution")
    hook_src = (ROOT / "pdm/solution_user_agent.py").read_text()

    assert batch.SOLUTION_ID == edge.SOLUTION_ID == SOLUTION_ID
    assert batch.SOLUTION_VERSION == edge.SOLUTION_VERSION == VERSION
    assert batch.SOLUTION_USER_AGENT == SOLUTION_UA
    assert f'_DEFAULT_SOLUTION_UA = "{SOLUTION_UA}"' in hook_src
    assert re.fullmatch(r"\d+\.\d+\.\d+", VERSION)


def test_stack_descriptions_follow_convention():
    batch = _load(ROOT / "infrastructure/batch/solution.py", "batch_solution")
    edge = _load(ROOT / "infrastructure/edge/solution.py", "edge_solution")
    pattern = re.compile(rf"^\({SOLUTION_ID}[A-Z]?\) Predictive Maintenance Skill - .+, v{re.escape(VERSION)}$")

    main = batch.stack_description("Batch Inference")
    assert main.startswith(f"({SOLUTION_ID}) ")  # only the main template uses the bare ID
    assert pattern.match(main)
    for component in ("Edge Network", "Edge Resources", "Edge Demo Device"):
        desc = edge.stack_description(component)
        assert pattern.match(desc), desc
        assert not desc.startswith(f"({SOLUTION_ID}) ")


@pytest.mark.parametrize("path", [
    "infrastructure/batch/batch_inference_stack.py",
    "infrastructure/edge/network_stack.py",
    "infrastructure/edge/resources_stack.py",
    "infrastructure/edge/demo_device_stack.py",
])
def test_every_stack_sets_a_description(path):
    assert "stack_description(" in (ROOT / path).read_text(), path
