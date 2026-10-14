# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading


class HealthState:
    def __init__(self):
        self.initial_list_complete = False
        self.queue_saturated = False
        self.persistence_failed = False
        self.recording_failed = False
        self._lock = threading.Lock()

    def update(
        self,
        *,
        initial_list_complete: bool | None = None,
        queue_saturated: bool | None = None,
        persistence_failed: bool | None = None,
        recording_failed: bool | None = None,
    ) -> None:
        with self._lock:
            if initial_list_complete is not None:
                self.initial_list_complete = initial_list_complete
            if queue_saturated is not None:
                self.queue_saturated = queue_saturated
            if persistence_failed is not None:
                self.persistence_failed = persistence_failed
            if recording_failed is not None:
                self.recording_failed = recording_failed

    def ready(self) -> bool:
        with self._lock:
            return (
                self.initial_list_complete
                and not self.queue_saturated
                and not self.persistence_failed
                and not self.recording_failed
            )
