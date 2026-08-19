# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

from _attribution_lib_version import (
    get_commit_sha as get_commit_sha,
    get_version as get_version,
)
from attribution_lib.exceptions import (
    UninitializedRuleEngineError,
    UninitializedSymptomRegistryError,
)
from attribution_lib.prefilters import (
    get_metric_spec as get_metric_spec,
    get_pre_filter as _registered_pre_filter,
)

if TYPE_CHECKING:
    from attribution_lib.observation.inbound import Source as ObservationSource
    from attribution_lib.prefilters import PreFilterSpec
    from attribution_lib.rules_engine import RuleEngine
    from attribution_lib.symptoms.registry_generation import GeneratedSymptomRegistry

ip_to_nodenames: dict[str, str] = {}
ignored_hostname_prefixes: list[str] = []
_rules_engine: Optional["RuleEngine"] = None
_symptom_registry: Optional["GeneratedSymptomRegistry"] = None


def get_rules_engine() -> "RuleEngine":
    """Return the initialized RuleEngine. Raises if initialize() hasn't been called."""
    if _rules_engine is None:
        raise UninitializedRuleEngineError("attribution_lib.initialize() must be called before using the rules engine")
    return _rules_engine


def get_symptom_registry() -> "GeneratedSymptomRegistry":
    """Return the active generated symptom registry built during setup_rules().

    Returns:
        The process-global GeneratedSymptomRegistry compiled from loaded symptom
        declarations.

    Raises:
        UninitializedSymptomRegistryError: If setup_rules() has not been called.
    """
    if _symptom_registry is None:
        raise UninitializedSymptomRegistryError(
            "attribution_lib.setup_rules() must be called before using the symptom registry"
        )
    return _symptom_registry


def get_pre_filter(source: "ObservationSource") -> "PreFilterSpec":
    """Return the pre-filter spec for the given source built during setup_rules().

    Args:
        source: The source to get the pre-filter spec for.

    Returns:
        The pre-filter spec registered for the given source.

    Raises:
        UninitializedRuleEngineError: If setup_rules() has not been called.
    """
    if not initialized:
        raise UninitializedRuleEngineError("attribution_lib.setup_rules() must be called before querying pre-filters")
    return _registered_pre_filter(source)


def _identity(x: str) -> str:
    return x


hostname_mechanism: Callable[[str], str] = _identity

NodeDomainResolver = Callable[[str], Optional[str]]
node_domain_resolver: Optional[NodeDomainResolver] = None

initialized: bool = False


def __getattr__(name: str) -> str:
    if name == "__version__":
        return get_version()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
