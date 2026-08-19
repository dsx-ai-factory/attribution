import json
import re
import tomllib
from collections import Counter
from datetime import datetime
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[1] / "examples" / "static" / "configs"
SAMPLE_DATA_DIR = Path(__file__).resolve().parents[1] / "docs" / "sample-data" / "static"
SAMPLE_MANIFEST = SAMPLE_DATA_DIR / "sample-jobs.json"
BASELINE_LAST_JOB_NUMBER = 100
BASELINE_JOB_COUNT = 98
REPEAT_SAMPLE_JOB_COUNT = 20
XID31_ROOT_JOB_COUNT = 18
TOTAL_JOB_COUNT = BASELINE_JOB_COUNT + REPEAT_SAMPLE_JOB_COUNT
REPEAT_OFFENDER_NODE = "node-0101"
PRIORITIZED_NODE_FAILURE_WORKLOADS = {"workload-0101", "workload-0111"}
XID31_PATTERN = re.compile(r"Xid \(PCI:[^)]*\): 31|XID-31|XID 31", re.IGNORECASE)
DISALLOWED_SAMPLE_PATTERNS = {
    "Job failed": re.compile(r"\bjob failed\b", re.IGNORECASE),
    "Cancelled by Timeout": re.compile(r"\bcancelled by timeout\b|DUE TO TIME LIMIT", re.IGNORECASE),
}
REQUIRED_HARDWARE_MARKERS = {
    "XID-145-InfraHW": "Xid (PCI:0000:81:00): 145, (0x00000007 0x00000010",
    "XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0": "Xid (PCI:0000:82:00): 149, (0x00400011 0x00000000",
    "XID-149.10-NETIR_LINK_DOWN-PHY_LAYER_RETRANSMIT_TIMEOUT-0": "Xid (PCI:0000:83:00): 149, (0x00a00011 0x00000000",
    "XID-79-GPU_HAS_FALLEN_OFF_THE_BUS": "Xid (PCI:0000:84:00): 79,",
}
REQUIRED_NON_XID_SYMPTOMS = {
    "IB Error-12 Vendor-129 (IBV_WC_RETRY_EXC_ERR)",
    "Lustre Client Evicted",
    "GPU is lost reported by slurm",
    "cudaErrorIllegalAddress",
    "cudaErrorDevicesUnavailable",
    "OOM kill of user task",
    "OOM kill of system service",
    "OOM kill of a system pod",
}


def _workload_sample_text(workload: str) -> str:
    """Return concatenated application and syslog sample text for one workload.

    Args:
        workload: Static workload name such as ``workload-0001``.

    Returns:
        Combined text from committed application and syslog sample logs.
    """

    paths = [*list((SAMPLE_DATA_DIR / "applog" / workload).glob("*.log"))]
    paths.extend((SAMPLE_DATA_DIR / "syslog" / workload).glob("**/*.log"))
    return "\n".join(path.read_text(encoding="utf-8") for path in sorted(paths))


def _sample_manifest_jobs() -> list[dict[str, object]]:
    """Return sample-job manifest entries.

    Returns:
        Static sample-job manifest entries.
    """

    manifest = json.loads(SAMPLE_MANIFEST.read_text(encoding="utf-8"))
    jobs = manifest["jobs"]
    assert isinstance(jobs, list)
    return jobs


def _evidence_timestamp(job: dict[str, object]) -> datetime:
    """Return the timestamp for the manifest evidence marker.

    Args:
        job: Static sample-job manifest entry.

    Returns:
        Timestamp from the evidence line containing the manifest marker.
    """

    evidence_path = Path(__file__).resolve().parents[1] / str(job["evidence_path"])
    evidence_marker = str(job["evidence_marker"])
    for line in evidence_path.read_text(encoding="utf-8").splitlines():
        if evidence_marker in line:
            return datetime.fromisoformat(line.split(maxsplit=1)[0].replace("Z", "+00:00"))
    raise AssertionError(f"missing evidence marker for {job['id']}")


def test_static_quickstart_samples_use_workload_names():
    """Quickstart sample identities should use workload names, not job names."""
    paths = sorted(CONFIG_DIR.glob("workload-*.toml"))
    assert len(paths) == TOTAL_JOB_COUNT
    assert list(CONFIG_DIR.glob("job-*.toml")) == []

    for path in paths:
        config = tomllib.loads(path.read_text(encoding="utf-8"))
        static_config = config["eca"]["static"]
        expected_name = path.stem
        assert static_config["id"] == expected_name
        assert static_config["name"] == expected_name
        assert "/workload-" in static_config["std_out"]
        assert "/workload-" in static_config["std_err"]


def test_static_quickstart_samples_have_varied_elapsed_hours():
    """Quickstart sample workloads should exercise varied GPU-hour values."""
    elapsed_hours: list[float] = []
    for path in sorted(CONFIG_DIR.glob("workload-*.toml")):
        config = tomllib.loads(path.read_text(encoding="utf-8"))
        static_config = config["eca"]["static"]
        start_time = datetime.fromisoformat(static_config["start_time"])
        end_time = datetime.fromisoformat(static_config["end_time"])
        elapsed = (end_time - start_time).total_seconds() / 3600
        assert elapsed > 0
        elapsed_hours.append(round(elapsed, 3))

    assert len(elapsed_hours) == TOTAL_JOB_COUNT
    assert len(set(elapsed_hours)) >= 75
    assert max(elapsed_hours) - min(elapsed_hours) >= 24


def test_static_quickstart_first_workload_uses_placeholder_remote_peers():
    """The first sample should not emit external peer names as responsible nodes."""
    text = (SAMPLE_DATA_DIR / "applog" / "workload-0001" / "application.log").read_text(encoding="utf-8")
    assert "192.0.2.10" not in text
    assert ".example.invalid" not in text


def test_static_quickstart_sample_manifest_matches_configs():
    """The quickstart sample manifest should describe the full config corpus."""
    workloads = sorted(path.stem for path in CONFIG_DIR.glob("workload-*.toml"))
    manifest_jobs = _sample_manifest_jobs()

    assert len(manifest_jobs) == TOTAL_JOB_COUNT
    assert sorted(job["id"] for job in manifest_jobs) == workloads

    for job in manifest_jobs:
        config = tomllib.loads((CONFIG_DIR / f"{job['id']}.toml").read_text(encoding="utf-8"))
        static_config = config["eca"]["static"]
        assert static_config["state"] == job["state"]
        assert static_config["nodes"] == job["nodes"]


def test_static_quickstart_samples_use_multi_node_allocations():
    """Quickstart samples should demonstrate multi-node GPU jobs."""
    baseline_node_counts: list[int] = []
    repeat_node_counts: list[int] = []
    for path in sorted(CONFIG_DIR.glob("workload-*.toml")):
        config = tomllib.loads(path.read_text(encoding="utf-8"))
        nodes = config["eca"]["static"]["nodes"]

        assert config["workload"]["gpus_per_node"] == 8
        if int(path.stem.rsplit("-", maxsplit=1)[1]) <= BASELINE_LAST_JOB_NUMBER:
            baseline_node_counts.append(len(nodes))
            assert len(nodes) > 1
        else:
            repeat_node_counts.append(len(nodes))

    assert set(baseline_node_counts) == {32, 64, 1024}
    assert sorted(repeat_node_counts) == sorted([*list(range(1, 11)), *list(range(1, 11))])


def test_static_quickstart_sample_evidence_is_inside_job_window():
    """Each sample's evidence timestamp should fall inside its static job window."""
    manifest_jobs = _sample_manifest_jobs()

    for job in manifest_jobs:
        config = tomllib.loads((CONFIG_DIR / f"{job['id']}.toml").read_text(encoding="utf-8"))
        static_config = config["eca"]["static"]
        start_time = datetime.fromisoformat(static_config["start_time"])
        end_time = datetime.fromisoformat(static_config["end_time"])
        evidence_time = _evidence_timestamp(job)

        assert start_time <= evidence_time <= end_time


def test_static_quickstart_samples_include_manifest_evidence():
    """Each quickstart sample should contain its declared evidence marker."""
    manifest_jobs = _sample_manifest_jobs()

    for job in manifest_jobs:
        workload = str(job["id"])
        text = _workload_sample_text(workload)
        assert str(job["evidence_marker"]) in text


def test_static_quickstart_prioritizes_slurm_node_failures_over_xid31():
    """Slurm node failures should remain the root cause when XID-31 is also present."""
    jobs_by_id = {str(job["id"]): job for job in _sample_manifest_jobs()}

    for workload in PRIORITIZED_NODE_FAILURE_WORKLOADS:
        job = jobs_by_id[workload]
        assert job["root_cause_symptom"] == "Node failure reported by slurm"
        assert job["expected_outcome"] == "Fail InfraSW"
        assert XID31_PATTERN.search(_workload_sample_text(workload)) is not None


def test_static_quickstart_samples_have_root_cause_cap():
    """The quickstart corpus should not overrepresent one root symptom."""
    manifest_jobs = _sample_manifest_jobs()
    counts = Counter(str(job["root_cause_symptom"]) for job in manifest_jobs)
    repeat_count = counts.pop("XID-31", 0)

    assert counts != Counter()
    assert max(counts.values()) <= 3
    assert repeat_count == XID31_ROOT_JOB_COUNT


def test_static_quickstart_samples_include_xid31_repeat_offender_demo():
    """XID-31 demo jobs should point at one repeat-offender node across users."""
    manifest_jobs = [job for job in _sample_manifest_jobs() if str(job["root_cause_symptom"]) == "XID-31"]
    users: set[str] = set()
    attempt_ids: set[str] = set()
    elapsed_minutes: set[int] = set()
    node_counts: Counter = Counter()

    assert len(manifest_jobs) == XID31_ROOT_JOB_COUNT
    for job in manifest_jobs:
        config = tomllib.loads((CONFIG_DIR / f"{job['id']}.toml").read_text(encoding="utf-8"))
        static_config = config["eca"]["static"]
        nodes = static_config["nodes"]
        responsible_nodes = job["responsible_nodes"]
        start_time = datetime.fromisoformat(static_config["start_time"])
        end_time = datetime.fromisoformat(static_config["end_time"])

        users.add(static_config["user_name"])
        attempt_ids.add(static_config["attempt_id"])
        elapsed_minutes.add(int((end_time - start_time).total_seconds() / 60))
        node_counts[len(nodes)] += 1
        assert nodes[0] == REPEAT_OFFENDER_NODE
        assert job["expected_outcome"] == "Fail MultiCategory"
        assert responsible_nodes == [REPEAT_OFFENDER_NODE]
        assert job["evidence_source"] == "syslog"
        assert f"/{REPEAT_OFFENDER_NODE}/syslog.log" in str(job["evidence_path"])

    assert len(users) == XID31_ROOT_JOB_COUNT
    assert len(attempt_ids) == XID31_ROOT_JOB_COUNT
    assert len(elapsed_minutes) == XID31_ROOT_JOB_COUNT
    assert node_counts == Counter({count: 2 for count in range(2, 11)})


def test_static_quickstart_samples_identify_specific_responsible_nodes():
    """Each quickstart sample should identify one concrete responsible node."""
    manifest_jobs = _sample_manifest_jobs()

    for job in manifest_jobs:
        responsible_nodes = job["responsible_nodes"]
        nodes = job["nodes"]
        assert isinstance(responsible_nodes, list)
        assert isinstance(nodes, list)
        assert len(responsible_nodes) == 1
        assert responsible_nodes[0] in nodes

        evidence_path = Path(__file__).resolve().parents[1] / str(job["evidence_path"])
        evidence_text = evidence_path.read_text(encoding="utf-8")
        responsible_node = str(responsible_nodes[0])
        if job["evidence_source"] == "applog":
            assert f"responsible_node={responsible_node}" in evidence_text
        elif job["evidence_source"] == "syslog":
            assert f"/{responsible_node}/syslog.log" in str(evidence_path)
            assert responsible_node in evidence_text
        else:
            raise AssertionError(job["evidence_source"])


def test_static_quickstart_samples_exclude_generic_failed_and_timeout_outcomes():
    """Generic failed and timeout-cancelled samples should stay out."""
    corpus_text_parts = [SAMPLE_MANIFEST.read_text(encoding="utf-8")]
    for workload in sorted(path.stem for path in CONFIG_DIR.glob("workload-*.toml")):
        corpus_text_parts.append(_workload_sample_text(workload))

    corpus_text = "\n".join(corpus_text_parts)
    for outcome, pattern in DISALLOWED_SAMPLE_PATTERNS.items():
        assert pattern.search(corpus_text) is None, outcome


def test_static_quickstart_baseline_samples_skip_xid31():
    """The baseline quickstart corpus should not include XID-31 samples."""
    baseline_workloads = [
        path.stem
        for path in sorted(CONFIG_DIR.glob("workload-*.toml"))
        if int(path.stem.rsplit("-", maxsplit=1)[1]) <= BASELINE_LAST_JOB_NUMBER
    ]
    corpus_text = "\n".join(_workload_sample_text(workload) for workload in baseline_workloads)

    assert XID31_PATTERN.search(corpus_text) is None


def test_static_quickstart_samples_include_infrahw_xid_coverage():
    """The quickstart corpus should include restored Fail InfraHW samples."""
    manifest_jobs = _sample_manifest_jobs()
    hardware_jobs = [job for job in manifest_jobs if job["expected_outcome"] == "Fail InfraHW"]
    hardware_symptoms = {str(job["root_cause_symptom"]) for job in hardware_jobs}
    corpus_text = "\n".join(_workload_sample_text(str(job["id"])) for job in manifest_jobs)

    assert len(hardware_jobs) >= 30
    for symptom, marker in REQUIRED_HARDWARE_MARKERS.items():
        assert symptom in hardware_symptoms
        assert marker in corpus_text


def test_static_quickstart_samples_include_requested_non_xid_coverage():
    """The quickstart corpus should include the requested non-XID root symptoms."""
    manifest_jobs = _sample_manifest_jobs()
    symptoms = {str(job["root_cause_symptom"]) for job in manifest_jobs}

    assert REQUIRED_NON_XID_SYMPTOMS <= symptoms
