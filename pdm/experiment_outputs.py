"""Create collision-resistant output directories for independent experiment runs.

The hypothesis directory can be reused; each model/summary run directory cannot.
This avoids silently overwriting results when a parameter sweep is rerun.
"""

import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def reserve_experiment_run_dir(experiment_dir: str | Path, variant: str) -> Path:
    """Atomically reserve a unique directory beneath an experiment's root.

    `variant` is a short, filesystem-safe description of the configuration.
    The UTC stamp makes runs readable; mkdtemp guarantees uniqueness even when
    two processes start in the same second. Existing runs are never removed.
    """
    if not isinstance(variant, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", variant
    ):
        raise ValueError("variant must be 1-80 alphanumeric, '_' or '-' characters")

    root = Path(experiment_dir)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(tempfile.mkdtemp(prefix=f"{variant}_{stamp}_", dir=root))
