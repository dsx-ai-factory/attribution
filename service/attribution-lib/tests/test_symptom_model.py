# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tests for structured symptom families and semantic selectors."""

from types import MappingProxyType
from typing import Any

import pytest

from attribution_lib.serialization import serialize
from attribution_lib.symptoms import BaseSymptom, SymptomFamily, SymptomSelector


def test_symptom_family_defensively_freezes_member_properties():
    """Family members do not retain mutable caller-owned mappings."""
    properties = {"code": 149}
    family = SymptomFamily(name="xid", properties=properties)

    properties["code"] = 43

    assert family.properties == {"code": 149}
    assert isinstance(family.properties, MappingProxyType)
    with pytest.raises(TypeError):
        family.properties["code"] = 43


def test_symptom_family_equality_and_hash_ignore_mapping_order():
    """Equivalent family values remain interchangeable in hashed indexes."""
    first = SymptomFamily(name="xid", properties={"code": 149, "subcode": 21})
    second = SymptomFamily(name="xid", properties={"subcode": 21, "code": 149})

    assert first == second
    assert hash(first) == hash(second)
    assert {first, second} == {first}


def test_symptom_family_defensively_freezes_property_schema():
    """Family declarations do not retain mutable caller-owned schemas."""
    property_types = {"code": int, "subcode": int | None}
    family = SymptomFamily(name="xid", property_types=property_types)

    property_types["code"] = str

    assert family.property_types == {"code": int, "subcode": int | None}
    assert family.properties == {}
    assert isinstance(family.property_types, MappingProxyType)


def test_symptom_family_member_serializes_immutable_properties_as_mapping():
    """Event serialization preserves structured member properties as data."""
    family = SymptomFamily(name="xid", properties={"code": 149})

    assert serialize(family)["properties"] == {"code": 149}


def test_symptom_family_rejects_invalid_identity():
    """Malformed family identities fail at their construction boundary."""
    with pytest.raises(TypeError, match="name must be a string"):
        SymptomFamily(name=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="name must be non-empty"):
        SymptomFamily(name=" ")
    with pytest.raises(ValueError, match="cannot define both"):
        SymptomFamily(name="xid", property_types={"code": int}, properties={"code": 149})
    with pytest.raises(TypeError, match="must be hashable"):
        SymptomFamily(name="xid", properties={"code": []})
    for invalid_type in (list[int] | None, Any):
        with pytest.raises(TypeError, match="runtime-checkable type"):
            SymptomFamily(name="xid", property_types={"code": invalid_type})


@pytest.fixture
def xid_149() -> BaseSymptom:
    """Return one structured XID family member."""
    return BaseSymptom(
        symptom_id="XID-149.21-NETIR",
        family=SymptomFamily(name="xid", properties={"code": 149, "subcode": 21}),
    )


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        (SymptomSelector(symptom_id="XID-149.21-NETIR"), True),
        (SymptomSelector(symptom_id="XID-43"), False),
        (SymptomSelector(family_name="xid"), True),
        (SymptomSelector(family_name="gpu"), False),
        (SymptomSelector(family_name="xid", family_properties={"code": 149}), True),
        (SymptomSelector(family_name="xid", family_properties={"code": 43}), False),
        (
            SymptomSelector(family_name="xid", family_properties={"code": frozenset({43, 149})}),
            True,
        ),
        (SymptomSelector(family_name="xid", family_properties={"severity": "fatal"}), False),
        (
            SymptomSelector(
                symptom_id="XID-149.21-NETIR",
                family_name="xid",
                family_properties={"code": 149},
            ),
            True,
        ),
    ],
)
def test_selector_member_evaluates_structured_identity(
    xid_149: BaseSymptom,
    selector: SymptomSelector,
    expected: bool,
):
    """Selector membership requires every supplied identity criterion."""
    assert selector.member(xid_149) is expected


def test_selector_family_criteria_do_not_match_unstructured_symptom():
    """A raw-ID-only symptom cannot satisfy a family selector."""
    symptom = BaseSymptom(symptom_id="XID-149.21-NETIR")

    assert SymptomSelector(family_name="xid").member(symptom) is False


def test_selector_defensively_freezes_properties_and_is_hashable():
    """Selectors are safe immutable values for later registry indexes."""
    properties = {"code": frozenset({43, 149})}
    selector = SymptomSelector(family_name="xid", family_properties=properties)
    equivalent = SymptomSelector(family_name="xid", family_properties=dict(properties))

    properties["code"] = frozenset({31})

    assert selector.family_properties == {"code": frozenset({43, 149})}
    assert hash(selector) == hash(equivalent)
    with pytest.raises(TypeError):
        selector.family_properties["code"] = frozenset({31})


def test_selector_rejects_invalid_criteria():
    """Selectors reject criteria that cannot identify or match a symptom."""
    with pytest.raises(TypeError, match="symptom_id must be a string"):
        SymptomSelector(symptom_id=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires symptom_id or family_name"):
        SymptomSelector()
    with pytest.raises(ValueError, match="symptom_id must be non-empty"):
        SymptomSelector(symptom_id="")
    with pytest.raises(ValueError, match="family_name must be non-empty"):
        SymptomSelector(family_name=" ")
    with pytest.raises(ValueError, match="require family_name"):
        SymptomSelector(symptom_id="XID-149", family_properties={"code": 149})
    with pytest.raises(TypeError, match="must be hashable"):
        SymptomSelector(family_name="xid", family_properties={"code": []})


def test_base_symptom_serialization_preserves_legacy_shape_without_family():
    """Adding optional family metadata does not change existing payloads."""
    symptom = BaseSymptom(symptom_id="XID-149")

    assert symptom.to_dict() == {
        "symptom_id": "XID-149",
        "sources": [],
        "attributes": {},
        "cascade_discards": [],
    }


def test_base_symptom_serializes_resolved_family_properties():
    """Structured runtime identity is serialized without declaration schemas."""
    symptom = BaseSymptom(
        symptom_id="XID-149.21-NETIR",
        family=SymptomFamily(name="xid", properties={"code": 149, "subcode": 21}),
    )

    assert symptom.to_dict()["family"] == {
        "name": "xid",
        "properties": {"code": 149, "subcode": 21},
    }


def test_base_symptom_rejects_family_schema():
    """Runtime symptoms carry resolved member identity, not declaration schemas."""
    with pytest.raises(ValueError, match="member properties"):
        BaseSymptom(symptom_id="XID-", family=SymptomFamily(name="xid", property_types={"code": int}))


def test_base_symptom_string_identity_remains_compatibility_authority():
    """Family metadata does not alter existing equality or hash behavior."""
    structured = BaseSymptom(
        symptom_id="XID-149",
        family=SymptomFamily(name="xid", properties={"code": 149}),
    )
    unstructured = BaseSymptom(symptom_id="XID-149")

    assert structured == unstructured
    assert structured == "XID-149"
    assert hash(structured) == hash(unstructured)
