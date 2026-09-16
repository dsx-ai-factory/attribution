# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

.PHONY: help test test-unit test-e2e

help:
	@printf '%s\n' \
		'Usage: make <target>' \
		'' \
		'Targets:' \
		'  help       Show this help message.' \
		'  test       Run all test suites.' \
		'  test-unit  Run unit and integration test suites.' \
		'  test-e2e   Run end-to-end tests.'

test:
	bash .ci/run-tests.sh all

test-unit:
	bash .ci/run-tests.sh attribution-lib
	bash .ci/run-tests.sh service-app
	bash .ci/run-tests.sh service-app-client-drift
	bash .ci/run-tests.sh blueprint
	bash .ci/run-tests.sh frontend

test-e2e:
	bash .ci/run-tests.sh end-to-end
