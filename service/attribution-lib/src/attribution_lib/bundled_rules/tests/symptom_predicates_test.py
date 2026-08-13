# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Characterize the conditional-discard predicates and their resolution.

The predicate logic lives in bundled_rules; the resolution mechanism is an
engine-scoped registry populated by ``@symptom_predicate`` at rule-load time. These
pin both.
"""

from attribution_lib.bundled_rules.symptom_predicates import (
    contains_ibv_modify_qp_failure,
    job_elapsed_below_eap_observation_window,
    job_exceeded_time_limit,
)
from attribution_lib.core import get_bundled_rules_path
from attribution_lib.rules_engine import RuleEngine
from attribution_lib.sources import SymptomSet
from test_support.helpers import create_source_set


def _symptoms(dummy_ctx, *names):
    symptoms: SymptomSet[str] = SymptomSet()
    for name in names:
        symptoms.add(create_source_set(dummy_ctx, name))
    return symptoms


def test_contains_ibv_modify_qp_failure(dummy_ctx):
    assert contains_ibv_modify_qp_failure(_symptoms(dummy_ctx, "Call to ibv_modify_qp failed"), None) is True
    assert contains_ibv_modify_qp_failure(_symptoms(dummy_ctx, "some other symptom"), None) is False


def test_job_elapsed_below_eap_observation_window(simple_job_factory):
    short = simple_job_factory(start_timestamp=1700000000, end_timestamp=1700000000 + 3600)
    assert job_elapsed_below_eap_observation_window(SymptomSet(), short) is True

    long_run = simple_job_factory(start_timestamp=1700000000, end_timestamp=1700000000 + 3 * 3600)
    assert job_elapsed_below_eap_observation_window(SymptomSet(), long_run) is False

    assert job_elapsed_below_eap_observation_window(SymptomSet(), None) is False


def test_job_exceeded_time_limit(simple_job_factory):
    # Default window is ~167 minutes (10000s), so a 10-minute limit is exceeded.
    exceeded = simple_job_factory(time_limit=10)
    assert job_exceeded_time_limit(SymptomSet(), exceeded) is True

    within = simple_job_factory(time_limit=6000)
    assert job_exceeded_time_limit(SymptomSet(), within) is False

    assert job_exceeded_time_limit(SymptomSet(), None) is False


def test_predicates_register_onto_engine_on_load():
    """Loading the predicate rule file registers each predicate onto that engine.

    Registration is engine-scoped (via @symptom_predicate against the engine loading
    the file), so a fresh engine starts empty and gets exactly this file's predicates.
    """
    engine = RuleEngine()
    engine.load(str(get_bundled_rules_path() / "symptom_predicates.py"))
    for name in (
        "job_elapsed_below_eap_observation_window",
        "job_exceeded_time_limit",
        "contains_ibv_modify_qp_failure",
    ):
        assert engine.symptom_predicates.get(name) is not None, name
    assert engine.symptom_predicates.get("not_a_predicate") is None
