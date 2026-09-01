# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import pytest

import attribution_lib
from attribution_lib.attributor import Attributor
from attribution_lib.cancellation import (
    SlurmCancellationCommentRule,
    SlurmCancellationDecision,
    SlurmCancellationPolicy,
)
from attribution_lib.core import setup_rules
from attribution_lib.events import EventManager

OPERATOR_ACCOUNT = "scheduler-policy-bot"
QUOTA_COMMENT = "Cancelled by quota policy: allocation exceeded the configured quota"
QUOTA_SYMPTOM = "Cancelled by quota policy"
IDLE_COMMENT = "Cancelled by idle policy: allocation was idle for 30 minutes"
IDLE_SYMPTOM = "Cancelled by idle policy"


@pytest.fixture
def cancellation_policy():
    policy = SlurmCancellationPolicy(
        operator_accounts=frozenset({OPERATOR_ACCOUNT}),
        admin_comment_rules=(
            SlurmCancellationCommentRule(
                admin_comment_prefix="Cancelled by quota policy:",
                symptom=QUOTA_SYMPTOM,
            ),
            SlurmCancellationCommentRule(
                admin_comment_prefix="Cancelled by idle policy:",
                symptom=IDLE_SYMPTOM,
            ),
        ),
    )
    setup_rules(slurm_cancellation_policy=policy, force=True)
    yield policy
    setup_rules(force=True)


def _attribute(job_info):
    event_manager = EventManager(rules_engine=attribution_lib.get_rules_engine())
    try:
        event_stream = event_manager.create_stream(str(job_info.id))
        return Attributor.new(event_stream, job_info).attribute()
    finally:
        event_manager.close()


def test_policy_requires_a_configured_operator_and_nonempty_comment_detail():
    rule = SlurmCancellationCommentRule(
        admin_comment_prefix="Cancelled by policy:",
        symptom="Cancelled by configured policy",
    )
    policy = SlurmCancellationPolicy(
        operator_accounts=frozenset({OPERATOR_ACCOUNT}),
        admin_comment_rules=(rule,),
    )

    assert policy.classify(cancelled_by="workload-owner", admin_comment="Cancelled by policy: details") is None
    assert policy.classify(cancelled_by=OPERATOR_ACCOUNT, admin_comment="Cancelled by policy:") is None
    assert policy.classify(cancelled_by=OPERATOR_ACCOUNT, admin_comment=None) is None
    assert policy.classify(
        cancelled_by=OPERATOR_ACCOUNT,
        admin_comment="  Cancelled by policy: details  ",
    ) == SlurmCancellationDecision(rule=rule, evidence="Cancelled by policy: details")


def test_policy_uses_the_most_specific_matching_prefix():
    broad = SlurmCancellationCommentRule(
        admin_comment_prefix="Cancelled by policy:",
        symptom="Cancelled by broad policy",
    )
    specific = SlurmCancellationCommentRule(
        admin_comment_prefix="Cancelled by policy: quota:",
        symptom="Cancelled by quota policy",
    )
    policy = SlurmCancellationPolicy(
        operator_accounts=frozenset({OPERATOR_ACCOUNT}),
        admin_comment_rules=(broad, specific),
    )

    decision = policy.classify(
        cancelled_by=OPERATOR_ACCOUNT,
        admin_comment="Cancelled by policy: quota: limit exceeded",
    )

    assert decision is not None
    assert decision.rule == specific


def test_policy_rejects_duplicate_prefixes():
    first = SlurmCancellationCommentRule("Cancelled by policy:", "Policy A")
    second = SlurmCancellationCommentRule("Cancelled by policy:", "Policy B")

    with pytest.raises(ValueError, match="duplicate prefixes"):
        SlurmCancellationPolicy(admin_comment_rules=(first, second))


def test_policy_allows_multiple_prefixes_for_the_same_outcome_symptom():
    first = SlurmCancellationCommentRule("Cancelled by policy v1:", "Cancelled by policy")
    second = SlurmCancellationCommentRule("Cancelled by policy v2:", "Cancelled by policy")

    policy = SlurmCancellationPolicy(admin_comment_rules=(first, second))

    assert policy.symptom_declarations() == (first.symptom_declaration(),)


def test_policy_rejects_conflicting_outcome_metadata_for_one_symptom():
    first = SlurmCancellationCommentRule("Cancelled by policy v1:", "Cancelled by policy")
    second = SlurmCancellationCommentRule(
        "Cancelled by policy v2:",
        "Cancelled by policy",
        outcome="Fail InfraSW",
    )

    with pytest.raises(ValueError, match="must use the same outcome metadata"):
        SlurmCancellationPolicy(admin_comment_rules=(first, second))


@pytest.mark.parametrize(
    ("admin_comment", "expected_symptom"),
    [
        (QUOTA_COMMENT, QUOTA_SYMPTOM),
        (IDLE_COMMENT, IDLE_SYMPTOM),
    ],
)
def test_configured_admin_comment_is_attributed_inside_the_rule_engine(
    simple_job_factory,
    cancellation_policy,
    admin_comment,
    expected_symptom,
):
    job_info = simple_job_factory(
        state="CANCELLED by 4242",
        cancelled_by=OPERATOR_ACCOUNT,
        admin_comment=admin_comment,
        exit_code_signal=None,
    )

    result = _attribute(job_info)
    outcome = result.outcome_attr.value()

    assert outcome.outcome == "Fail User"
    assert outcome.root_cause_symptom == expected_symptom
    assert expected_symptom in result.symptoms
    assert "Cancelled by cluster operators" not in result.symptoms
    source = result.symptoms[expected_symptom].one_source()
    assert source is not None
    assert source.datum == admin_comment
    assert source.rule == "Slurm AdminComment cancellation policy"


@pytest.mark.parametrize("admin_comment", [None, "", "unrecognized scheduler comment"])
def test_configured_operator_without_comment_decision_uses_operator_fallback(
    simple_job_factory,
    cancellation_policy,
    admin_comment,
):
    job_info = simple_job_factory(
        state="CANCELLED by 4242",
        cancelled_by=OPERATOR_ACCOUNT,
        admin_comment=admin_comment,
        exit_code_signal=None,
    )

    result = _attribute(job_info)

    assert result.outcome_attr.value().outcome == "Cancelled by cluster operators"
    assert "Cancelled by cluster operators" in result.symptoms
    assert QUOTA_SYMPTOM not in result.symptoms
    assert IDLE_SYMPTOM not in result.symptoms


def test_unconfigured_canceller_remains_a_user_cancellation(simple_job_factory, cancellation_policy):
    job_info = simple_job_factory(
        state="CANCELLED by 4242",
        cancelled_by="workload-owner",
        admin_comment=QUOTA_COMMENT,
        exit_code_signal=None,
    )

    result = _attribute(job_info)

    assert result.outcome_attr.value().outcome == "Cancelled by User"
    assert "Cancelled by user" in result.symptoms
    assert QUOTA_SYMPTOM not in result.symptoms
