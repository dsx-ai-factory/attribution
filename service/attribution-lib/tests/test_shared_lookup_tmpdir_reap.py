"""A shared lookup directory must not outlive every process that can reach it.

`attribution_lib.events` gives each process one `attribution_lib_shared_lookup_*`
directory and removes it from an `atexit` handler. `atexit` only runs on a clean
interpreter shutdown, so a process killed with SIGKILL leaves its directory behind
with nothing left that knows the owner is gone, and so does one that exits through
`os._exit()`, which is how CPython ends a `multiprocessing` worker under `fork` and
`forkserver` before 3.13. Measured at 192 directories an hour on
one host, 3.4 MB each, from a pool whose workers run Python 3.11 under `fork`.

Reaping has to stay narrower than "the owning PID is dead", because a forked child
inherits the directory and may still be ATTACHed to files inside it after the owner
exits. These tests pin both halves: an abandoned directory is reclaimed, and a
directory anything can still reach is not.
"""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

PREFIX = "attribution_lib_shared_lookup_"

# Exercised in a child interpreter rather than in-process: the defect is about how
# a process dies, and the reap runs once per process on first use.
PROBE = """
import os, sys
import attribution_lib.events as ev
from attribution_lib.events import _get_shared_lookup_tmpdir

mode = sys.argv[1]

if mode == "reap-in-lock-gap":
    # Drive the real _get_shared_lookup_tmpdir() and run a reap inside the
    # gap between creating the lock file and locking it, which is the only
    # moment a reaper could find a live directory unguarded. Hooked on
    # os.open rather than on a fixed filename so it lands in that gap
    # whatever name the lock file is built under. O_CREAT tells the owner
    # creating the file apart from a reaper opening one, which never passes
    # it, so the reap this triggers is not re-entered.
    parent = os.environ["TMPDIR"]
    real_open = os.open
    fired = []

    def hooked_open(path, flags, *args, **kwargs):
        # Proxies os.open faithfully, kwargs included: the reap this triggers
        # runs shutil.rmtree, which calls os.open with dir_fd. Dropping that
        # would raise TypeError and make the test fail for the wrong reason.
        fd = real_open(path, flags, *args, **kwargs)
        creating = bool(flags & os.O_CREAT)
        if creating and os.path.basename(str(path)).startswith(ev._SHARED_LOOKUP_LOCK_NAME) and fired == []:
            fired.append(path)
            ev._reap_orphaned_shared_lookup_tmpdirs(parent)
        return fd

    os.open = hooked_open
    try:
        owned = _get_shared_lookup_tmpdir()
    finally:
        os.open = real_open
    print(owned, flush=True)
    print("interposed" if fired != [] else "not-interposed", flush=True)
    print("survived" if os.path.isdir(owned) else "deleted", flush=True)
    # A reap after the lock is in place must also leave it alone, which fails
    # if the lock file was renamed into position without being held.
    ev._reap_orphaned_shared_lookup_tmpdirs(parent)
    print("locked" if os.path.isdir(owned) else "unlocked", flush=True)
    os._exit(0)

if mode == "rollback-on-init-failure":
    # Fail the last step of building the directory. Without a rollback the
    # directory is stranded: no lock file under the name a reaper looks up,
    # and no published cleanup state, so nothing can ever remove it. Must run
    # before the unconditional build below, or the directory already exists
    # and _get_shared_lookup_tmpdir just returns the cached path.
    real_rename = os.rename

    def failing_rename(*args, **kwargs):
        raise OSError("rename failed")

    parent = os.environ["TMPDIR"]
    os.rename = failing_rename
    try:
        _get_shared_lookup_tmpdir()
        print("no-error", flush=True)
    except OSError:
        print("raised", flush=True)
    finally:
        os.rename = real_rename
    left = [n for n in os.listdir(parent) if n.startswith(ev._SHARED_LOOKUP_PREFIX)]
    print(len(left), flush=True)
    os._exit(0)

owned = _get_shared_lookup_tmpdir()
print(owned, flush=True)

if mode == "fork-child-exit":
    # The parent owns the directory. The child inherits the global and the
    # atexit registration, and must not remove what it did not create.
    pid = os.fork()
    if pid == 0:
        sys.exit(0)  # Runs the inherited atexit handler.
    os.waitpid(pid, 0)
    print("survived" if os.path.isdir(owned) else "deleted", flush=True)
    os._exit(0)  # Leave the directory behind for the test to inspect.

if mode == "fork-child-outlives":
    # The owning PID dies first and the child keeps using the directory.
    pid = os.fork()
    if pid == 0:
        print(os.getpid(), flush=True)
        sys.stdin.read()
        os._exit(0)
    os._exit(0)

if mode == "fork-child-outlives-clean-exit":
    # The owner forks and then exits CLEANLY, so its atexit handler runs
    # while the child is still holding the lock it inherited.
    pid = os.fork()
    if pid == 0:
        print(os.getpid(), flush=True)
        sys.stdin.read()
        os._exit(0)
    # Falls off the end of the script, so the interpreter shuts down normally.

if mode == "wait":
    sys.stdin.read()
"""


def run_probe(tmp_path, mode):
    """Run the probe to completion with `tmp_path` as its temp root.

    Args:
        tmp_path: Directory used as TMPDIR, so the probe's shared lookup
            directory and its reaps stay inside this test.
        mode: Probe mode, selecting what the probe does after it has built
            its directory.

    Returns:
        The probe's whitespace-separated stdout tokens, one per line printed.
    """
    result = subprocess.run(
        [sys.executable, "-c", PROBE, mode],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "TMPDIR": str(tmp_path)},
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.split()


def start_probe(tmp_path, mode):
    """Start the probe and return it once it has reported its directory.

    Args:
        tmp_path: Directory used as TMPDIR, as in `run_probe`.
        mode: Probe mode. Use one that keeps the process alive, or that
            forks, so there is something left to observe.

    Returns:
        The still-running `Popen` and the shared lookup directory it built.
    """
    process = subprocess.Popen(
        [sys.executable, "-c", PROBE, mode],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "TMPDIR": str(tmp_path)},
    )
    owned = process.stdout.readline().strip()
    assert owned.startswith(str(tmp_path)), process.stderr.read()
    return process, Path(owned)


def lookup_dirs(tmp_path):
    """List the shared lookup directories left under `tmp_path`.

    Args:
        tmp_path: Directory the probes used as TMPDIR.

    Returns:
        Their paths, sorted.
    """
    return sorted(path for path in tmp_path.iterdir() if path.name.startswith(PREFIX))


def test_clean_exit_still_removes_its_own_directory(tmp_path):
    """The owner removes its own directory when nothing else holds it."""
    (owned,) = run_probe(tmp_path, "exit")

    assert not Path(owned).exists()
    assert lookup_dirs(tmp_path) == []


def test_sigkilled_owner_leaves_a_directory_that_a_later_process_reclaims(tmp_path):
    """A directory abandoned by SIGKILL is reclaimed by the next process."""
    process, abandoned = start_probe(tmp_path, "wait")
    process.send_signal(signal.SIGKILL)
    process.wait(timeout=60)
    # atexit cannot run on SIGKILL, so the directory has to still be here.
    assert abandoned.is_dir()

    (successor,) = run_probe(tmp_path, "exit")

    assert not abandoned.exists(), "an abandoned directory must be reclaimed"
    assert not Path(successor).exists()
    assert lookup_dirs(tmp_path) == []


def test_reap_leaves_a_directory_whose_owner_is_still_running(tmp_path):
    """Another process's reap must not touch a live owner's directory."""
    process, live = start_probe(tmp_path, "wait")
    try:
        run_probe(tmp_path, "exit")

        assert live.is_dir(), "a live owner's directory must survive another process's reap"
        assert lookup_dirs(tmp_path) == [live]
    finally:
        process.send_signal(signal.SIGKILL)
        process.wait(timeout=60)


def test_forked_child_exit_leaves_the_owning_process_directory(tmp_path):
    """A forked child's own exit must not delete what it did not create."""
    owned, verdict = run_probe(tmp_path, "fork-child-exit")

    assert verdict == "survived", "a forked child must not delete the directory it inherited"
    assert Path(owned).is_dir()


def test_owner_clean_exit_leaves_a_directory_its_forked_child_still_holds(tmp_path):
    """The owner's own cleanup must respect a lock its child inherited.

    Being the owning PID is not enough to delete the directory. `flock`
    lives on the open file description, so a forked child keeps the lock
    held after the owner exits, and may still have EventManagers ATTACHed to
    files inside. This is the case the lock exists to protect, and the
    reaper already honours it; the owner's `atexit` path has to honour it
    too, or one process removes what another is using.
    """
    process, inherited = start_probe(tmp_path, "fork-child-outlives-clean-exit")
    child_pid = int(process.stdout.readline().strip())
    # The owner exits cleanly here, so its atexit handler runs in full.
    assert process.wait(timeout=60) == 0
    try:
        assert inherited.is_dir(), "a cleanly exiting owner must not remove what its child still holds"
        assert lookup_dirs(tmp_path) == [inherited]
    finally:
        os.kill(child_pid, signal.SIGKILL)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
def test_reap_leaves_a_directory_held_only_by_an_outliving_forked_child(tmp_path):
    """The owning PID is gone but a forked child still holds the directory.

    This is the case a PID-keyed reaper gets wrong: the owner is dead by
    construction, yet the directory is still in use.
    """
    process, inherited = start_probe(tmp_path, "fork-child-outlives")
    child_pid = int(process.stdout.readline().strip())
    process.wait(timeout=60)  # The owning PID is now gone; the child is not.
    try:
        assert inherited.is_dir()

        run_probe(tmp_path, "exit")

        assert inherited.is_dir(), "a directory an outliving child still holds must survive"
    finally:
        os.kill(child_pid, signal.SIGKILL)


def test_reap_during_the_lock_gap_leaves_the_directory_being_built(tmp_path):
    """A reaper running while an owner is taking its lock must not reap it.

    Creating the lock file and locking it are two syscalls. If the lock file
    is reachable under the name a reaper opens while it is still unheld, the
    reaper takes the lock, removes the directory, and the owner's own lock
    then succeeds on an unlinked inode, so it never learns the directory is
    gone. One daemon almost never gets preempted between those two calls;
    sixteen pool workers starting together on a many-core host do.
    """
    owned, interposed, verdict, held = run_probe(tmp_path, "reap-in-lock-gap")

    assert interposed == "interposed", "the reap never landed in the gap, so this proves nothing"
    assert verdict == "survived", "a reaper in the lock gap must not remove a directory being built"
    assert held == "locked", "the lock must be held before it is reachable under the name a reaper opens"
    assert Path(owned).is_dir()


def test_failed_initialisation_leaves_nothing_behind(tmp_path):
    """A directory half-built when initialisation fails must be rolled back.

    It would otherwise leak in exactly the shape this change exists to fix:
    no lock file under the name a reaper looks up, so every future reap
    ignores it, and no published cleanup state, so `atexit` cannot remove it
    either. Nothing else can, so the failing path has to undo its own work.
    """
    outcome, remaining = run_probe(tmp_path, "rollback-on-init-failure")

    assert outcome == "raised", "the failure must reach the caller, not be swallowed"
    assert remaining == "0", "a directory stranded by a failed build can never be reaped"
    assert lookup_dirs(tmp_path) == []


def test_reap_leaves_a_directory_it_cannot_prove_is_abandoned(tmp_path):
    """A directory with no lock file predates this code and is left alone."""
    legacy = tmp_path / f"{PREFIX}legacy"
    legacy.mkdir()
    (legacy / "lookup_1.duckdb").write_bytes(b"")

    run_probe(tmp_path, "exit")

    assert legacy.is_dir()
    assert lookup_dirs(tmp_path) == [legacy]
