# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager

from attribution_lib.declarative.context import _using_rule_loader
from attribution_lib.rules_engine import RuleLoader


@contextmanager
def _with_loader():
    """Provide a loader that is active for rule registration."""
    loader = RuleLoader()
    with _using_rule_loader(loader):
        yield loader
