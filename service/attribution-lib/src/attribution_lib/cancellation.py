# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Configurable Slurm cancellation attribution policy.

Consumers provide their scheduler operator identities and optional
``AdminComment`` rules. The attribution engine owns applying those rules so
symptoms, outcomes, evidence, and decision traces remain internally consistent.
"""

from dataclasses import dataclass
from typing import Optional

from attribution_lib.symptoms.declarations import OutcomePriority, SymptomDeclaration


@dataclass(frozen=True, slots=True)
class SlurmCancellationCommentRule:
    """Map one Slurm ``AdminComment`` prefix to an attribution symptom."""

    admin_comment_prefix: str
    symptom: str
    outcome: str = "Fail User"
    hw_cause: str = "Not Applicable"
    priority: OutcomePriority = OutcomePriority.FALLBACK
    doc: str = "cancelled-by-user"

    def __post_init__(self) -> None:
        normalized_fields = {
            "admin_comment_prefix": self.admin_comment_prefix.strip(),
            "symptom": self.symptom.strip(),
            "outcome": self.outcome.strip(),
            "hw_cause": self.hw_cause.strip(),
            "doc": self.doc.strip(),
        }
        for field_name, value in normalized_fields.items():
            if value == "":
                raise ValueError(f"{field_name} must not be empty")
            object.__setattr__(self, field_name, value)

    def symptom_declaration(self) -> SymptomDeclaration:
        """Return registry metadata for the configured symptom."""
        return SymptomDeclaration(
            prefix=self.symptom,
            outcome=self.outcome,
            hw_cause=self.hw_cause,
            priority=self.priority,
            doc=self.doc,
        )


@dataclass(frozen=True, slots=True)
class SlurmCancellationDecision:
    """A policy match with normalized comment evidence."""

    rule: SlurmCancellationCommentRule
    evidence: str


@dataclass(frozen=True, slots=True)
class SlurmCancellationPolicy:
    """Classify scheduler-owned cancellations without embedding site identities."""

    operator_accounts: frozenset[str] = frozenset()
    admin_comment_rules: tuple[SlurmCancellationCommentRule, ...] = ()

    def __post_init__(self) -> None:
        normalized_accounts = frozenset(account.strip() for account in self.operator_accounts)
        if "" in normalized_accounts:
            raise ValueError("operator_accounts must not contain empty values")

        rules = tuple(self.admin_comment_rules)
        prefixes = [rule.admin_comment_prefix for rule in rules]
        if len(prefixes) != len(set(prefixes)):
            raise ValueError("admin_comment_rules must not contain duplicate prefixes")

        declarations_by_symptom: dict[str, SymptomDeclaration] = {}
        for rule in rules:
            declaration = rule.symptom_declaration()
            existing = declarations_by_symptom.setdefault(rule.symptom, declaration)
            if existing != declaration:
                raise ValueError(f"admin_comment_rules for symptom {rule.symptom!r} must use the same outcome metadata")

        # Most-specific prefix wins when providers intentionally configure a
        # broad fallback and a narrower policy beneath it.
        rules = tuple(sorted(rules, key=lambda rule: (-len(rule.admin_comment_prefix), rule.admin_comment_prefix)))
        object.__setattr__(self, "operator_accounts", normalized_accounts)
        object.__setattr__(self, "admin_comment_rules", rules)

    @property
    def configured_symptoms(self) -> frozenset[str]:
        """Return every customer-facing symptom this policy can emit."""
        return frozenset(rule.symptom for rule in self.admin_comment_rules)

    def is_operator(self, cancelled_by: Optional[str]) -> bool:
        """Return whether ``cancelled_by`` is a configured operator identity."""
        return cancelled_by is not None and cancelled_by in self.operator_accounts

    def classify(
        self,
        *,
        cancelled_by: Optional[str],
        admin_comment: Optional[str],
    ) -> Optional[SlurmCancellationDecision]:
        """Return the most-specific recognized AdminComment decision, if any."""
        if not self.is_operator(cancelled_by) or not isinstance(admin_comment, str):
            return None

        normalized = admin_comment.strip()
        for rule in self.admin_comment_rules:
            if not normalized.startswith(rule.admin_comment_prefix):
                continue
            if normalized.removeprefix(rule.admin_comment_prefix).strip() == "":
                continue
            return SlurmCancellationDecision(rule=rule, evidence=normalized)
        return None

    def symptom_declarations(self) -> tuple[SymptomDeclaration, ...]:
        """Return registry declarations for all configured comment rules."""
        return tuple({rule.symptom: rule.symptom_declaration() for rule in self.admin_comment_rules}.values())


DEFAULT_SLURM_CANCELLATION_POLICY = SlurmCancellationPolicy(
    operator_accounts=frozenset({"site-automation-account"}),
)
