"""Single source of truth for this solution's AWS usage-tracking identifier.

``SOLUTION_USER_AGENT`` is injected into the Lambda and SageMaker Processing Job
environments as the ``USER_AGENT_STRING`` variable and appended to the
User-Agent of every AWS SDK call so re-use of the solution can be measured.
"""
SOLUTION_ID = "SO0356"
SOLUTION_VERSION = "0.1.0"
SOLUTION_USER_AGENT = f"AWSSOLUTION/{SOLUTION_ID}/v{SOLUTION_VERSION}"
SOLUTION_NAME = "Predictive Maintenance Skill"


def stack_description(component: str, suffix: str = "") -> str:
    """CloudFormation template Description: ``(<ID>[suffix]) <Name> - <component>, v<version>``.

    Only the main deployment template uses the bare Solution ID. Every other
    template passes a ``suffix`` so it is not counted as a separate deployment.
    """
    return f"({SOLUTION_ID}{suffix}) {SOLUTION_NAME} - {component}, v{SOLUTION_VERSION}"
