# NVIDIA Attribution Library

This package contains the rules engine used by the NVIDIA Attribution Service.
It is bundled as a local dependency of the Service and is not normally run
directly by end users.

During private preview, the bundled rules are not a supported customer
extension point. See the bundle root `README.md` for supported workflows.

## Slurm cancellation policy

Services can identify scheduler-owned accounts and translate recognized Slurm
`AdminComment` prefixes into customer-facing symptoms. Unrecognized comments
from a configured account fall back to `Cancelled by cluster operators`; other
accounts retain the normal `Cancelled by user` behavior.

```python
from attribution_lib.cancellation import (
    SlurmCancellationCommentRule,
    SlurmCancellationPolicy,
)
from attribution_lib.core import setup_rules

policy = SlurmCancellationPolicy(
    operator_accounts=frozenset({"scheduler-policy-bot"}),
    admin_comment_rules=(
        SlurmCancellationCommentRule(
            admin_comment_prefix="Cancelled by quota policy:",
            symptom="Cancelled by quota policy",
        ),
    ),
)
setup_rules(slurm_cancellation_policy=policy)
```

Pass the raw scheduler identity to `JobInfoBuilder.cancelled_by()` and the raw
Slurm administrator comment to `JobInfoBuilder.admin_comment()`. A comment rule
matches only for a configured operator account and only when text follows the
configured prefix.
