"""Pinned requirements for SageMaker training jobs.

AutoGluon refuses to load a predictor saved by a different AutoGluon version
(``TabularPredictor.load(require_version_match=True)``). Remote training must
therefore install exactly the versions of the local environment, which come
from the project's ``uv.lock`` and are also what the inference container and
edge component install.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

# (distribution name, requirement prefix, fallback specifier if not installed locally)
_TRAINING_PACKAGES = [
    ("autogluon.tabular", "autogluon.tabular[lightgbm,catboost,xgboost]", None),
    ("lifelines", "lifelines", ">=0.29"),
    ("optuna", "optuna", ">=3.0"),
]
# ray is only used for AutoGluon's parallel fold fitting. Take its version range
# from the installed AutoGluon's own "ray" extra so the two never disagree.
def _ray_requirement() -> str | None:
    from importlib.metadata import requires

    try:
        reqs = requires("autogluon.core") or []
    except PackageNotFoundError:
        return None
    for req in reqs:
        spec, _, markers = req.partition(";")
        if spec.strip().startswith("ray") and 'extra == "ray"' in markers:
            return spec.strip()
    return None


def training_requirements() -> str:
    """Return requirements.txt content pinned to the locally installed versions."""
    lines = []
    for dist, requirement, fallback in _TRAINING_PACKAGES:
        try:
            lines.append(f"{requirement}=={version(dist)}")
        except PackageNotFoundError:
            if fallback is None:
                raise RuntimeError(
                    f"{dist} is not installed locally; run `uv sync` so remote training "
                    "can pin the same version the model will be served with."
                ) from None
            lines.append(f"{requirement}{fallback}")
    ray = _ray_requirement()
    if ray:
        lines.append(ray)
    return "\n".join(lines) + "\n"
