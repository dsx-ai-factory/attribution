# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spill directories of unclosed EventManagers are removed when interpreters and pool workers exit."""

import json
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROBE = """
import gc
import json
import multiprocessing
import sys

LEAKED = []


def build_manager(_):
    '''Create an EventManager with a stream in this worker and never close it.

    Args:
        _: Unused pool input.

    Returns:
        The spill directory path of the manager.
    '''
    from attribution_lib.events import EventManager
    from attribution_lib.rules_engine import RuleLoader

    # A manager and its stream reference each other, so only a garbage
    # collection pass can run __del__. Disable it to model a worker that
    # exits before the next pass.
    gc.disable()
    manager = EventManager(RuleLoader().build())
    manager.create_stream("1")
    LEAKED.append(manager)
    return manager._spill_dir


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "interpreter":
        paths = [build_manager(None)]
    else:
        pool = multiprocessing.get_context(mode).Pool(1, maxtasksperchild=1)
        paths = pool.map(build_manager, range(2), chunksize=1)
        pool.close()
        pool.join()
    print(json.dumps(paths))
"""


@pytest.mark.parametrize("mode", ["interpreter", *multiprocessing.get_all_start_methods()])
def test_spill_directories_of_unclosed_managers_are_removed_after_exit(tmp_path_factory, mode):
    """Normal shutdown and worker recycling must remove each unclosed manager's spill directory.

    Args:
        tmp_path_factory: Factory for isolated temporary directories.
        mode: Interpreter shutdown or a multiprocessing start method.
    """
    # Keep forkserver socket paths below the Unix-domain length limit.
    tmp_path = tmp_path_factory.mktemp("spill")
    script = tmp_path / "spill_cleanup_probe.py"
    script.write_text(PROBE)
    result = subprocess.run(
        [sys.executable, str(script), mode],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "TMPDIR": str(tmp_path)},
    )
    assert result.returncode == 0, result.stderr
    paths = [Path(path) for path in json.loads(result.stdout.splitlines()[-1])]
    assert len(set(paths)) == (1 if mode == "interpreter" else 2)
    for path in paths:
        assert path.parent == tmp_path
        assert not path.exists()
    assert list(tmp_path.glob("attribution_lib_spill_*")) == []


def test_finalizer_logs_a_directory_it_cannot_remove(tmp_path, monkeypatch):
    """A removal failure in the exit finalizer is logged with the directory path, not raised.

    Args:
        tmp_path: Isolated temporary directory.
        monkeypatch: Fixture that replaces `shutil.rmtree`.
    """
    from loguru import logger

    from attribution_lib import events

    def fail_rmtree(path):
        """Fail as a non-writable spill root does.

        Args:
            path: Directory to remove.
        """
        raise PermissionError(13, "Permission denied", path)

    path, finalizer = events._mkdtemp_removed_at_exit("probe_", root=str(tmp_path))
    monkeypatch.setattr(events.shutil, "rmtree", fail_rmtree)
    messages = []
    sink = logger.add(messages.append, level="ERROR")
    try:
        finalizer()
    finally:
        logger.remove(sink)
    assert len(messages) == 1
    assert path in messages[0]
    assert "PermissionError" in messages[0]
