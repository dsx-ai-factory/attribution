# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Shared exception type for the declarative rule loader.

Lives in its own module so both ``emit`` and ``expressions`` can import it
without creating a circular dependency (``emit`` already imports from
``expressions``).
"""

import difflib
from collections.abc import Iterable

from attribution_lib.exceptions import RuleIngestionError


class LoadError(RuleIngestionError):
    """A declarative rule file is malformed or uses an unsupported construct."""


class RegistrationError(LoadError):
    """Structural error caught during the per-stanza registration pass.

    Raised for problems detectable from the stanza alone: missing or conflicting
    ``emit``/``processor`` fields, an unrecognized key, reserved capture-group names,
    an unknown ``processor`` reference, duplicate ``const`` names, or colliding table
    names.
    """


class UnknownKeyError(RegistrationError):
    """A declaration carries a key the grammar does not define at that level."""

    def __init__(self, key: str, known: Iterable[str], where: str):
        options = sorted(known)
        close = difflib.get_close_matches(key, options, n=1)
        hint = f"did you mean {close[0]!r}?" if len(close) > 0 else f"known keys are {options}"
        super().__init__(f"{where}: unknown key {key!r}; {hint}")


class FinalizeError(LoadError):
    """Validation error caught during the post-registration finalize pass.

    Raised after all stanzas are collected, when the engine validates and compiles
    them: bad ``match_re``, unresolved captures/helpers/consts/names, failed
    coverage checks, or compilation errors.
    """
