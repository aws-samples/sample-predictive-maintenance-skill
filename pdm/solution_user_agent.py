"""Register a process-wide botocore hook that appends the AWS Solutions usage
tracking token to the User-Agent of every AWS SDK (boto3/botocore) call this
process makes.

Import this module before any other module that talks to AWS.

botocore is part of the ``aws`` extra, not the slim inference core. Without
it (e.g. on an edge device) the process makes no AWS SDK calls, so there is
nothing to tag and the hook is skipped.
"""
import os

try:
    import botocore.session
except ImportError:  # slim install without the aws extra
    botocore = None

_DEFAULT_SOLUTION_UA = "AWSSOLUTION/SO0356/v0.1.0"
SOLUTION_UA = os.environ.get("USER_AGENT_STRING") or _DEFAULT_SOLUTION_UA

if botocore is not None:
    _orig_init = botocore.session.Session.__init__

    def _init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        extra = self.user_agent_extra or ""
        if SOLUTION_UA and SOLUTION_UA not in extra.split():
            self.user_agent_extra = f"{extra} {SOLUTION_UA}".strip()

    botocore.session.Session.__init__ = _init
