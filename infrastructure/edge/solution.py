"""Solution identifiers for the edge stacks.

Keep in sync with ``infrastructure/batch/solution.py`` (enforced by
``tests/test_solution_tracking.py``). Edge stacks are not the main deployment,
so they use the ``E`` suffix and are not counted as separate deployments.
"""
SOLUTION_ID = "SO0356"
SOLUTION_VERSION = "0.1.0"
SOLUTION_NAME = "Predictive Maintenance Skill"
EDGE_SUFFIX = "E"


def stack_description(component: str) -> str:
    return f"({SOLUTION_ID}{EDGE_SUFFIX}) {SOLUTION_NAME} - {component}, v{SOLUTION_VERSION}"
