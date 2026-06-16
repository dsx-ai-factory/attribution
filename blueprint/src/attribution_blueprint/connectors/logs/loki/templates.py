# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from attribution_blueprint.types import Source

# Default LogQL templates per Source. One Source can carry multiple queries
# (concatenated into a single Observation). Templates substitute customer-
# supplied label values (query_kwargs) at render time.
#
# LogQL uses {} for label selectors, which collide with Python's format()
# field syntax — literal { and } must be doubled. Substitution placeholders
# stay as single braces, e.g. {service_name}.
#
# OVERRIDING THESE
# ----------------
# These are samples from one reference fleet, not a contract. Customers
# running against their own Loki should expect to override.
#
#   TOML / CLI / env: supply `query` (single template) or `queries` (list)
#                     under `[log.<source>.loki]`. Customer-supplied queries
#                     fully replace this module's defaults for that source —
#                     `deep_merge` clobbers list-valued keys (see the
#                     "Precedence and merge" section of ARCHITECTURE.md).
#
#   Programmatic:     pass `templates=` to LokiConnector(...). Importing
#                     `BARE_TEMPLATES` from this module gives a minimal
#                     selector-only set as a starting point that returns
#                     every line matching the customer's labels — useful
#                     for exploration before locking down filters.
#
# Bundled SYSLOG query templates.
TEMPLATES: dict[Source, list[str]] = {
    Source.SYSLOG: [
        '{{service_name="{service_name}", level!=`INFO`, appName=`kernel`}} '
        "|~ `LustreError|connection2:0: ping timeout|detected conn error|Xid|"
        "GspRmAlloc failed|NV_ERR_|NV_WARN_|NMI watchdog|general protection|"
        "frame pointer|Failed to collect nvlink status info!|"
        "Failed to update Rx Detect Link mask!|"
        "Abrupt nvidia-imex daemon shutdown detected, robust channel recovery invoked!|"
        "Unable to handle kernel|not responding, still trying|"
        "Out of memory: Killed process`",
        '{{service_name="{service_name}", appName=`systemd`}} |~ `Stopping nvidia-imex.service`',
        '{{service_name="{service_name}", level!=`INFO`, appName=`systemd-logind`}} |~ `power`',
    ],
    Source.APPLOG: [
        '{{service_name="{service_name}", app="{app}"}} | slurm_job_id = `{job_id}`',
    ],
}


# Minimal selector-only templates. Returns every line matching the customer-
# supplied label values, with no `|~` filter narrowing. Useful as a starting
# point when onboarding an unfamiliar Loki — see what the stream actually
# contains, then add filters. Customers should expect to tighten these
# (or switch back to a tuned bundled set) before production use, since
# stream-wide queries can be expensive.
#
# Selector keys: SYSLOG uses `{service_name}`, APPLOG uses both
# `{service_name}` and `{app}` — match what the bundled set above expects so
# `BARE_TEMPLATES` is a drop-in for the same `query_kwargs`.
BARE_TEMPLATES: dict[Source, list[str]] = {
    Source.SYSLOG: ['{{service_name="{service_name}"}}'],
    Source.APPLOG: ['{{service_name="{service_name}", app="{app}"}}'],
}
