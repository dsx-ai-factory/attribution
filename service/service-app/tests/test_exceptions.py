# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from typing import ClassVar

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from app.exceptions import (
    AttributionErrorModel,
    AttributionHTTPException,
    DuplicateAttributorError,
    DuplicateObservationError,
    EventStorageOverflowError,
    ResponseModel,
    UnknownAttributorIdError,
    UnknownObservationFormatError,
)


def test_response_model_registry():
    """Test that ResponseModel properly registers subclasses."""

    class TestModel(ResponseModel):
        pass

    assert TestModel.__name__ in ResponseModel._registry
    assert ResponseModel._registry[TestModel.__name__] is TestModel


def test_response_model_union():
    """Test that ResponseModel can generate a proper union type."""

    class ModelA(ResponseModel):
        field_a: str

    class ModelB(ResponseModel):
        field_b: int

    union_type = ResponseModel.get_union()
    assert union_type is not None


def test_response_model_from_dict():
    """Test that ResponseModel can create instances from dictionaries."""

    class TestModel(ResponseModel):
        test_field: str

    data = {"test_field": "test_value", "model_type": "TestModel"}
    instance = ResponseModel.from_dict(data)
    assert isinstance(instance, TestModel)
    assert instance.test_field == "test_value"


def test_attribution_error_model_base():
    """Test the base AttributionErrorModel functionality."""

    class TestError(AttributionErrorModel):
        code: ClassVar[int] = 400
        msg: ClassVar[str] = "Test error"

    error = TestError()
    assert error.code == 400
    assert error.msg == "Test error"
    assert error.detail() == "Test error"

    response = TestError.response()
    assert 400 in response
    assert response[400]["description"] == "Test error"


def test_duplicate_observation_error():
    """Test DuplicateObservationError functionality."""
    obs_id = 123
    error = DuplicateObservationError.new(obs_id)

    assert error.code == 409
    assert error.obs_id == obs_id
    assert error.detail() == f"Duplicate observation {obs_id} rejected."


def test_unknown_attributor_id_error():
    """Test UnknownAttributorIdError functionality."""
    att_id = "test_att_123"
    error = UnknownAttributorIdError.new(att_id)

    assert error.code == 404
    assert error.att_id == att_id
    assert error.detail() == f"Attribution id {att_id} unknown."


def test_duplicate_attributor_error():
    """Test DuplicateAttributorError functionality."""
    att_id = "test_att_123"
    error = DuplicateAttributorError.new(att_id)

    assert error.code == 409
    assert error.att_id == att_id
    assert error.detail() == f"Duplicate attribution {att_id} rejected."


def test_unknown_observation_format_error():
    """Test UnknownObservationFormatError functionality."""
    format = "invalid_format"
    error = UnknownObservationFormatError.new(format)

    assert error.code == 400
    assert error.format == format
    assert error.detail() == f"Unknown observation format {format}."


def test_event_storage_overflow_error():
    """Test EventStorageOverflowError functionality."""
    att_id = "test_att_123"
    error = EventStorageOverflowError.new(att_id)

    assert error.code == 507
    assert error.att_id == att_id
    assert error.detail() == f"Attributor {att_id} exceeded its memory budget."


def test_attribution_http_exception():
    """Test AttributionHTTPException functionality."""
    error = UnknownAttributorIdError.new("test_att_123")
    http_exception = AttributionHTTPException(error)

    assert isinstance(http_exception, HTTPException)
    assert http_exception.status_code == 404
    assert http_exception.detail["att_id"] == "test_att_123"
    assert http_exception.detail["model_type"] == "UnknownAttributorIdError"

    json_response = http_exception.to_json_response()
    assert isinstance(json_response, JSONResponse)
    assert json_response.status_code == 404
    body = json.loads(json_response.body)
    assert body["att_id"] == "test_att_123"
    assert body["model_type"] == "UnknownAttributorIdError"
