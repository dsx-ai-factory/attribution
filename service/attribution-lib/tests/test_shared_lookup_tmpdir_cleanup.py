# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared lookup directories are removed when interpreters and pool workers exit."""

import json
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROBE = """
import json
import multiprocessing
import sys


def build_lookup(_):
    '''Create a lookup directory in this worker.

    Args:
        _: Unused pool input.

    Returns:
        The shared lookup directory path.
    '''
    from attribution_lib.events import _get_shared_lookup_tmpdir
    return _get_shared_lookup_tmpdir()


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "interpreter":
        paths = [build_lookup(None)]
    else:
        pool = multiprocessing.get_context(mode).Pool(1, maxtasksperchild=1)
        paths = pool.map(build_lookup, range(2), chunksize=1)
        pool.close()
        pool.join()
    print(json.dumps(paths))
"""


@pytest.mark.parametrize("mode", ["interpreter", *multiprocessing.get_all_start_methods()])
def test_shared_lookup_directories_are_removed_after_exit(tmp_path_factory, mode):
    """Normal shutdown and worker recycling must remove each owned directory.

    Args:
        tmp_path_factory: Factory for isolated temporary directories.
        mode: Interpreter shutdown or a multiprocessing start method.
    """
    # Keep forkserver socket paths below the Unix-domain length limit.
    tmp_path = tmp_path_factory.mktemp("lookup")
    script = tmp_path / "lookup_cleanup_probe.py"
    script.write_text(PROBE)
    result = subprocess.run(
        [sys.executable, str(script), mode],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "TMPDIR": str(tmp_path)},
    )
    assert result.returncode == 0, result.stderr
    paths = [Path(path) for path in json.loads(result.stdout)]
    assert len(set(paths)) == (1 if mode == "interpreter" else 2)
    for path in paths:
        assert path.parent == tmp_path
        assert not path.exists()
    assert list(tmp_path.glob("attribution_lib_shared_lookup_*")) == []
