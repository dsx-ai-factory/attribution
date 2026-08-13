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

from attribution_lib.hooks import MultiHookManager, SingleHookIllegalReplacement, SingleHookManager, SingleHookNoDefault


@pytest.fixture
def multihooks():
    return MultiHookManager()


@pytest.fixture
def singlehooks():
    return SingleHookManager()


def test_multi_hook_symptom_processing(multihooks):
    # Clear for isolation
    multihooks._specs.clear()

    @multihooks.SymptomProcessing.hook(priority=10)
    def svc_plugin(symptoms, job_info):
        if hasattr(job_info, "cancelled_by") and job_info.cancelled_by == "user_ab8e17ec":
            symptoms.add("Cancelled by service")
            symptoms.discard("Cancelled by user")

    @multihooks.SymptomProcessing.hook(priority=5)
    def timeout_plugin(symptoms, job_info):
        if hasattr(job_info, "state") and job_info.state == "TIMEOUT":
            symptoms.add("Cancelled by Timeout")

    class JobInfo:
        def __init__(self, cancelled_by=None, state=None):
            self.cancelled_by = cancelled_by
            self.state = state

    # Test MultiHook
    job = JobInfo(cancelled_by="user_ab8e17ec", state="TIMEOUT")
    symptoms = {"terminate called", "Cancelled by user"}

    multihooks.SymptomProcessing(symptoms, job)

    assert "Cancelled by service" in symptoms
    assert "Cancelled by Timeout" in symptoms
    assert "Cancelled by user" not in symptoms


def test_single_hook_replacement_control(singlehooks):
    singlehooks._specs.clear()

    # Set up SingleHook with default
    @singlehooks.TestReplace.set_default
    def default_func(x):
        return x * 2

    # First hook
    @singlehooks.TestReplace.hook()
    def first_hook(x):
        return x * 3

    result1 = singlehooks.TestReplace(5)
    assert result1 == 15

    # Try to replace with replace_ok=False - should fail
    with pytest.raises(SingleHookIllegalReplacement):

        @singlehooks.TestReplace.hook()
        def blocked_hook(x):
            return x * 4

    # Replace with explicit permission - should work
    @singlehooks.TestReplace.hook(replace_ok=True)
    def replacement_hook(x):
        return x * 5

    result3 = singlehooks.TestReplace(5)
    assert result3 == 25  # Now using replacement_hook


def test_single_hook_clear_functionality(singlehooks):
    singlehooks._specs.clear()

    @singlehooks.TestClear.set_default
    def default_format(text):
        return f"Default: {text}"

    # Test default
    result1 = singlehooks.TestClear("hello")
    assert result1 == "Default: hello"

    # Add custom hook
    @singlehooks.TestClear.hook()
    def custom_format(text):
        return f"Custom: {text}"

    result2 = singlehooks.TestClear("hello")
    assert result2 == "Custom: hello"

    # Clear hook - should revert to default
    singlehooks.TestClear.clear_hook()
    result3 = singlehooks.TestClear("hello")
    assert result3 == "Default: hello"


def test_multi_hook_priorities(multihooks):
    multihooks._specs.clear()

    execution_order = []

    @multihooks.TestPriority.hook(priority=1)
    def low_priority(data):
        execution_order.append("low")
        data.append("low")

    @multihooks.TestPriority.hook(priority=10)
    def high_priority(data):
        execution_order.append("high")
        data.append("high")

    @multihooks.TestPriority.hook(priority=5)
    def medium_priority(data):
        execution_order.append("medium")
        data.append("medium")

    test_data = []
    multihooks.TestPriority(test_data)

    assert execution_order == ["high", "medium", "low"]
    assert test_data == ["high", "medium", "low"]


def test_error_messages(singlehooks):
    singlehooks._specs.clear()

    # Try to run SingleHook without default
    with pytest.raises(SingleHookNoDefault) as exc_info:
        singlehooks.BadExample()

    error_msg = str(exc_info.value)
    assert "SingleHookNoDefault" in error_msg
    assert "BadExample" in error_msg
