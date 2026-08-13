# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from unittest.mock import Mock

import pytest

from attribution_blueprint.webui import cli as web_cli


def test_main_reports_missing_assets_without_traceback(monkeypatch, capsys):
    """Missing web asset messages should be concise CLI errors.

    Args:
        monkeypatch: Pytest fixture for replacing app creation and server startup.
        capsys: Pytest fixture for capturing stdout and stderr.
    """

    create_app = Mock(side_effect=web_cli.WebAssetsMissingError("React assets missing: /tmp/index.html not found"))
    run = Mock()
    monkeypatch.setattr(web_cli, "create_app", create_app)
    monkeypatch.setattr(web_cli.uvicorn, "run", run)

    rc = web_cli.main(["--db", "test.db"])

    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert "React assets missing" in captured.err
    assert "Traceback" not in captured.err
    create_app.assert_called_once_with("test.db")
    run.assert_not_called()


def test_main_reports_schema_error_without_traceback(monkeypatch, capsys):
    """Invalid SQLite schema messages should be concise CLI errors.

    Args:
        monkeypatch: Pytest fixture for replacing app creation and server startup.
        capsys: Pytest fixture for capturing stdout and stderr.
    """

    message = "SQLite attribution database is missing required column(s): sub_domain, terminal_state."
    create_app = Mock(side_effect=web_cli.DatabaseSchemaError(message))
    run = Mock()
    monkeypatch.setattr(web_cli, "create_app", create_app)
    monkeypatch.setattr(web_cli.uvicorn, "run", run)

    rc = web_cli.main(["--db", "test.db"])

    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert message in captured.err
    assert "Traceback" not in captured.err
    create_app.assert_called_once_with("test.db")
    run.assert_not_called()


def test_main_lets_unexpected_runtime_error_raise(monkeypatch):
    """Unexpected RuntimeError should not be reported as a startup usage error.

    Args:
        monkeypatch: Pytest fixture for replacing app creation.
    """

    create_app = Mock(side_effect=RuntimeError("unexpected app setup failure"))
    monkeypatch.setattr(web_cli, "create_app", create_app)

    with pytest.raises(RuntimeError, match="unexpected app setup failure"):
        web_cli.main(["--db", "test.db"])
