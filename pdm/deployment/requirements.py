"""Locked requirements for deployment targets of the slim pdm install.

Deployment targets install the base ``pdm-project`` dependencies plus only the
extras a model needs, at the exact versions in ``uv.lock`` (AutoGluon refuses to
load predictors saved by another version, and scikit-learn pickles are likewise
version-sensitive). Requirements are exported with hashes, so ``pip install -r``
runs in hash-checking mode.

CLI::

    # Greengrass component for a trained model (extras chosen from metadata.json)
    python -m pdm.deployment.requirements --target edge --model-dir ./model -o requirements.txt

    # Explicit extras
    python -m pdm.deployment.requirements --extras serve -o requirements.txt
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

# metadata.json "formulation" -> extras needed to load and predict. Covers both
# PDMModel classes ("multilabel", "rul") and the train.py CLI
# ("multi-label classification", "regression").
FORMULATION_EXTRAS: dict[str, list[str]] = {
    "anomaly_detection": [],
    "classification": ["tabular"],
    "multilabel": ["tabular"],
    "multi-label classification": ["tabular"],
    "rul": ["tabular"],
    "regression": ["tabular"],
    "survival": ["survival"],
}

TARGET_EXTRAS: dict[str, list[str]] = {
    "edge": ["edge"],
    "serve": ["serve"],
}


def extras_for(target: str, formulation: str | None = None) -> list[str]:
    """Extras to install for a deployment target serving ``formulation``."""
    if target not in TARGET_EXTRAS:
        raise ValueError(f"Unknown target {target!r}; choose one of {sorted(TARGET_EXTRAS)}")
    extras = list(TARGET_EXTRAS[target])
    if formulation is not None:
        if formulation not in FORMULATION_EXTRAS:
            raise ValueError(
                f"Unknown formulation {formulation!r}; expected one of {sorted(FORMULATION_EXTRAS)}"
            )
        extras += [e for e in FORMULATION_EXTRAS[formulation] if e not in extras]
    return extras


def model_formulation(model_dir: str | Path) -> str:
    """Read the formulation from a model directory's metadata.json."""
    return json.loads((Path(model_dir) / "metadata.json").read_text())["formulation"]


def export_requirements(extras: list[str], output: str | Path, project_dir: str | Path = ".") -> Path:
    """Write locked, hashed requirements for the base install plus ``extras``.

    Uses ``uv export --frozen``: fails instead of re-resolving if ``uv.lock`` is
    out of date with ``pyproject.toml``.
    """
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to export locked requirements (https://docs.astral.sh/uv/)")
    output = Path(output).resolve()
    cmd = [uv, "export", "--frozen", "--no-dev", "--no-emit-project", "--quiet", "-o", str(output)]
    for extra in extras:
        cmd += ["--extra", extra]
    subprocess.run(cmd, cwd=project_dir, check=True)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--target", choices=sorted(TARGET_EXTRAS))
    group.add_argument("--extras", help="Comma-separated extras, e.g. 'edge,tabular'")
    parser.add_argument("--model-dir", type=Path, help="Model directory; adds the extras its formulation needs")
    parser.add_argument("--project-dir", type=Path, default=Path("."), help="Directory with pyproject.toml and uv.lock")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.target:
        formulation = model_formulation(args.model_dir) if args.model_dir else None
        extras = extras_for(args.target, formulation)
    else:
        extras = [e.strip() for e in args.extras.split(",") if e.strip()]
    path = export_requirements(extras, args.output, project_dir=args.project_dir)
    print(f"Wrote {path} (extras: {', '.join(extras) or 'none'})")


if __name__ == "__main__":
    main()
