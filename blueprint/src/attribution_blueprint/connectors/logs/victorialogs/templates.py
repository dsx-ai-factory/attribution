# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from attribution_blueprint.types import Source

TEMPLATES: dict[Source, list[str]] = {
    Source.SYSLOG: [
        'service_name:="{service_name}" -level:="INFO" appName:="kernel" '
        '_msg:~"LustreError|connection2:0: ping timeout|detected conn error|Xid|'
        "GspRmAlloc failed|NV_ERR_|NV_WARN_|NMI watchdog|general protection|"
        "frame pointer|Failed to collect nvlink status info!|"
        "Failed to update Rx Detect Link mask!|"
        "Abrupt nvidia-imex daemon shutdown detected, robust channel recovery invoked!|"
        "Unable to handle kernel|not responding, still trying|"
        'Out of memory: Killed process"',
        'service_name:="{service_name}" appName:="systemd" _msg:~"Stopping nvidia-imex.service"',
        'service_name:="{service_name}" -level:="INFO" appName:="systemd-logind" _msg:~"power"',
    ],
    Source.APPLOG: [
        'service_name:="{service_name}" app:="{app}" slurm_job_id:="{job_id}"',
    ],
}

BARE_TEMPLATES: dict[Source, list[str]] = {
    Source.SYSLOG: ['service_name:="{service_name}"'],
    Source.APPLOG: ['service_name:="{service_name}" app:="{app}"'],
}
