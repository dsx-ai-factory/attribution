# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import attribution_lib
import attribution_lib.core as core
import pytest
from attribution_lib.bundled_rules.common.syslog_helpers import SyslogAttributionParams
from attribution_lib.core import process_check
from attribution_lib.job_info import JobInfoBuilder
from attribution_lib.observation.inbound import (
    Format as ObservationFormat,
    Source as ObservationSource,
    TimeInterval,
)
from attribution_lib.resources.resource_set import AllJobResources
from attribution_lib.sources import Source, SourceSet
from attribution_lib.symptoms.symptom_enums import RegisteredSymptoms
from pytest import param as pparam
from test_support.helpers import (
    UnknownAttr,
    make_node_to_ranks,
    make_syslog_entry,
    match_bad,
    match_interesting,
)

xfail = pytest.mark.xfail
TEST_LOG_ROOT = Path(__file__).parents[4] / "tests" / "app_logs"


@pytest.fixture(autouse=True)
def _preload_test_rules(event_manager):
    """Preload rule names specific to this test file."""
    event_manager.preload_rule_names(
        {
            "Synthetic MFU",
            "Synthetic PyTorch NCCL watchdog",
            "Synthetic dataloader exception",
            "fm-lost-connection-to-gpu",
        }
    )


@pytest.fixture
def sys_tst_bldr(attribution_test_builder):
    """Wrapper around attribution_test_builder with syslog-specific defaults.

    Simplifies syslog tests by providing sensible defaults:
    - source: SYSLOG (always)
    """

    def _build(job_info, data, *, stream_id=None, events_to_add=None, time_interval=None):
        return attribution_test_builder(
            job_info=job_info,
            data=data,
            source=ObservationSource.SYSLOG,
            check_fn=process_check,
            stream_id=stream_id,
            events_to_add=events_to_add,
            time_interval=time_interval,
        )

    return _build


@pytest.fixture(scope="class")
def syslog_jobinfo_builder(timezone):
    """JobInfo fixture for syslog tests."""

    def _build_jobinfo(tflops_start, single_gpu=False):
        start = datetime.fromtimestamp(1741860016, tz=timezone)
        end = datetime.fromtimestamp(1744734448, tz=timezone)
        if tflops_start:
            tflops_time = start
        else:
            tflops_time = end

        all_nodes = [
            "host-08b5fda3",
            "host-359412e9",
            "host-0e040fc6",
            "host-7b29aa8c",
            "host-1b17a383",
            "host-c30cc342",
            "host-db354baf",
            "host-303cc177",
            "host-98f6741a",
            "host-81d6467b",
            "host-659b9b31",
            "host-5d299dd2",
            "host-fdb7a3b8",
            "host-336e171b",
            "host-f56ea954",
            "host-77dd5c5d",
            "host-879513e8",
            "host-df28688e",
            "host-15700e13",
            "host-241bb838",
            "host-e98da538",
            "host-daa9fd6a",
            "host-a7300290",
            "host-fece4e02",
            "host-98b6cefa",
            "host-70288d8a",
            "host-5e5a1aa5",
            "host-cf6f1555",
            "host-21ee1d38",
            "host-8ad70bcf",
            "host-bfaaa49a",
            "host-e35c094c",
            "host-97ff87d7",
            "host-c39adf1d",
            "host-f67f1bec",
            "host-a5a36e22",
            "host-b1c83eeb",
            "host-8ca374c2",
            "host-d25b79d2",
            "host-6140f062",
            "host-7b054672",
            "host-38d9571b",
            "host-dca8700f",
            "host-587aa398",
            "host-ad4a6bfb",
            "host-3bb450e3",
            "host-f26d405e",
            "host-32568b11",
            "host-3d3b61a0",
            "host-718fc892",
            "host-de383c8f",
            "host-0447dadc",
            "host-ae569eac",
            "host-81f06fd5",
            "host-de13b9ca",
            "host-df384431",
            "host-2be2da75",
            "host-b7f1e8c8",
            "host-e029258a",
            "host-56b3f881",
            "host-f5eb2d49",
            "host-756564a4",
            "host-fbf898d3",
            "host-0aa449ea",
            "host-e6c5bdcf",
            "host-4dc2b464",
            "host-70c8e201",
            "host-4104bc45",
            "host-04de1cc2",
            "host-27940975",
            "host-26647abd",
            "host-211aa9e6",
            "host-55c5ca51",
            "host-70c94f9b",
            "host-5e23ad63",
            "host-d39929bf",
            "host-b57a7941",
            "host-be1c71d6",
            "host-3be7e0f9",
            "host-88e24d8a",
            "host-99c47acb",
            "host-fc501ecc",
            "host-cafb4253",
            "host-c66ec529",
            "host-b1d5d524",
            "host-7db413f6",
            "host-b497dde2",
            "host-8b0c8172",
            "host-2af9d726",
            "host-a875fd03",
            "host-ab7519a2",
            "host-36e692e4",
            "host-4b588b4d",
            "host-cc784838",
            "host-47887fe7",
            "host-2fee5f75",
            "host-ff8c4158",
            "host-9cba717b",
            "host-d465621a",
            "host-d08a49fa",
            "host-5b862484",
            "host-1a979463",
            "host-890a21e8",
            "host-ef7af87d",
            "host-d2299be7",
            "host-f9d3d838",
            "host-64962a8f",
            "host-7d8fe5f3",
            "host-956d9d0d",
            "host-964390dc",
            "host-2bcc8030",
            "host-bb14b9cd",
            "host-d6f7fdd3",
            "host-cd6efb46",
            "host-04685df5",
            "host-08c3ac05",
            "host-d00b46da",
            "host-8b89870d",
            "host-7fce7aa9",
            "host-4d39d0ca",
            "host-3ffbbae2",
            "host-a4a913ee",
            "host-f57b4fcb",
            "host-d0771cd4",
            "host-b36e9df2",
            "host-0ce3710d",
            "host-b8887df1",
            "host-136a3306",
            "host-2b2acff7",
            "host-ce325563",
            "host-4afee59f",
            "host-f3b3beb8",
            "host-d4ceae90",
            "host-9a8b24b6",
            "host-79d860f5",
            "host-23112bf8",
            "host-8a6bdd53",
            "host-cc60085d",
            "host-4c09b973",
            "host-7f617295",
            "host-1bc63f0c",
            "host-92582b3d",
            "host-47e9cd17",
            "host-8cab3d7a",
            "host-dbbbca2f",
            "host-ac28afcc",
            "host-a68740d7",
            "host-0595e03f",
            "host-c5e98a79",
            "host-8ded53e3",
            "host-57f4fa8e",
            "host-de02e8ef",
            "host-13ce2eae",
            "host-2e5c8325",
            "host-673439a6",
            "host-0719c837",
            "host-df4f3ee5",
            "host-ff353f7b",
            "host-0e16f33b",
            "host-8782124b",
            "host-c6e40eb3",
            "host-8c088911",
            "host-260c66c2",
            "host-67a3c452",
            "host-80ee57a6",
            "host-b6e94070",
            "host-6756f650",
            "host-2c6e9a62",
            "host-eafe27d1",
            "host-687150f3",
            "host-e943e435",
            "host-24465aa4",
            "host-fca64f39",
            "host-bd944f94",
            "host-e510ac3e",
            "host-3adefa12",
            "host-89ebd3b1",
            "host-36df2c98",
            "host-6f76d09b",
            "host-6ae7c6d8",
            "host-6f13e623",
            "host-db75b6f2",
            "host-59f59ca2",
            "host-047bd7b1",
            "host-bd5b99b3",
            "host-13f686d9",
            "host-b79d1164",
            "host-8660f24b",
            "host-9cc55af6",
            "host-6b3de755",
            "host-44051da1",
            "host-95bd9233",
            "host-ed4748b4",
            "host-b34c7ffd",
        ]

        if single_gpu:
            ranks_per_node = 1
            nodelist = all_nodes[:1]
        else:
            ranks_per_node = 8
            nodelist = all_nodes
        return (
            JobInfoBuilder(
                id="1887602",
                cluster="cluster_b722552b",
                state="CANCELLED",
                type="training",
                framework="megatron",
                node_to_ranks=make_node_to_ranks(nodelist, ranks_per_node),
                job_start_time=start,
                source_name="artifical slurm",
                timezone=timezone,
            )
            .srun_cmd(
                "srun -l --container-image=/redacted_path/aef53de8+megatron-lm+pytorch+24.08-py3-test-model-v1.sqsh --container-mounts=/lustre:/lustre,/redacted_path/8a6f6ab0:/root/.netrc --output=/redacted_path/eba7bc22%x_%j_date_25-03-13_time_03-00-16.log sh -c python -u /redacted_path/c7da3173  \t--disable-gloo-process-groups \t--ckpt-format torch_dist \t--ckpt-fully-parallel-save \t--ckpt-fully-parallel-load \t--ckpt-assume-constant-structure \t--async-save         \t--squared-relu         --no-mmap-bin-files \t--tp-comm-overlap \t--distributed-timeout-minutes 10 \t--exit-duration-in-mins 230 \t--no-create-attention-mask-in-dataloader                 --overlap-grad-reduce         --overlap-param-gather         --sequence-parallel         --tensor-model-parallel-size 4         --use-distributed-optimizer         \t--window-size 512 0 \t--untie-embeddings-and-output-weights         --init-method-std 0.014         --position-embedding-type none         --num-layers 32         --hidden-size 2048         --num-attention-heads 16         --group-query-attention         --num-query-groups 8         --hybrid-override-pattern M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M- \t--ffn-hidden-size 8192 \t--kv-channels 128         --seq-length 4096         --max-position-embeddings 4096         --train-samples 1000000         --lr-decay-style cosine         --lr-warmup-samples 50000         --lr-decay-samples 950000         --save /redacted_path/39d73042         --load /redacted_path/39d73042 \t--per-split-data-args-path /redacted_path/03b2db45 \t--data-cache-path /redacted_path/5a0e7c28         --tiktoken-pattern v2 \t--tokenizer-type TikTokenizer         --tokenizer-model /redacted_path/0ab2e758         --distributed-backend nccl         --micro-batch-size 4         --global-batch-size 256         --lr 1.2e-3         --min-lr 3.0e-6         --weight-decay 0.1         --clip-grad 1.0         --attention-dropout 0.0         --hidden-dropout 0.0         --disable-bias-linear         --normalization RMSNorm         --adam-beta1 0.9         --adam-beta2 0.95         --log-interval 10 \t--log-params-norm \t--log-num-zeros-in-grad \t--log-throughput         --save-interval 2000 \t--save-retain-interval 2000         --eval-interval 1000         --eval-iters 14         --bf16         --use-mcore-models         --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec         --tensorboard-dir /redacted_path/2bce3ee3"
            )
            .exit_code_signal(15)
            .job_end_time(end)
            .last_tflops_time(tflops_time)
            .time_limit(6000)
            .build()
        )

    return _build_jobinfo


def gen_mfu(jobinfo, tflops_start, obs_data):
    if tflops_start:
        time = jobinfo.job_start_time
    else:
        time = jobinfo.job_end_time
    obase = obs_data(jobinfo, [], ObservationSource.APPLOG)
    src = Source(
        source_context=obase.context,
        line_num=None,
        datum="<pseudo mfu>",
        time=time,
        resources=AllJobResources,
        rule="Synthetic MFU",
    )

    return ("Application.mfu", {"datetime": time}, src)


@pytest.fixture(scope="class")
def syslog_jobinfo(timezone, syslog_jobinfo_builder):
    return syslog_jobinfo_builder(True)


class TestNodes:
    @pytest.fixture
    def raw_result(self, request, sys_tst_bldr, syslog_jobinfo):
        filename = TEST_LOG_ROOT / "syslog_otlp_entries.json"
        return sys_tst_bldr(syslog_jobinfo, filename, stream_id=request.node.name)

    def test_raw(self, raw_result):
        """Test node attribution from syslog data."""
        raw_nodes = raw_result.get_node_attributions()
        assert "XID-95-UNCONTAINED_ERROR" in raw_nodes.attributions["host-ac28afcc"].attributions

    @pytest.fixture
    def plain_result(self, request, sys_tst_bldr, syslog_jobinfo):
        filename = TEST_LOG_ROOT / "syslog_plain_entries.json"
        return sys_tst_bldr(syslog_jobinfo, filename, stream_id=request.node.name)

    def test_plain(self, plain_result):
        """Test node attribution from syslog data."""
        plain_nodes = plain_result.get_node_attributions()
        assert "XID-95-UNCONTAINED_ERROR" in plain_nodes.attributions["host-ac28afcc"].attributions
        assert "XID-95-UNCONTAINED_ERROR" in plain_nodes.attributions["host-98f6741a"].attributions


@pytest.mark.parametrize(
    ("contained_detail", "expected_symptom"),
    [
        ("CE User Channel (0x36)", "XID-94-CONTAINED_ERROR-CE"),
        ("SM (0x1)", "XID-94-CONTAINED_ERROR-SM"),
        ("GSP", "XID-94-CONTAINED_ERROR-GSP"),
        ("Unknown Engine", "XID-94-CONTAINED_ERROR"),
    ],
)
def test_xid94_poison_victim_selects_concrete_symptom(
    contained_detail,
    expected_symptom,
    get_one_line_attr_builder,
    syslog_jobinfo,
):
    """Emit concrete XID-94 identity when the poison victim is recognized."""

    node = syslog_jobinfo.nodes[0]
    line = f"NVRM: Xid (PCI:0000:06:00): 94, Contained: {contained_detail}. RST: No, D-RST: No"
    node_attributions = get_one_line_attr_builder(
        line,
        node,
        syslog_jobinfo.job_start_time.timestamp(),
        syslog_jobinfo,
    )

    symptoms = node_attributions.attributions[node].attributions
    assert expected_symptom in symptoms
    assert any(source.datum == line and source.rule == "known-xid-event" for source in symptoms[expected_symptom])
    if expected_symptom != "XID-94-CONTAINED_ERROR":
        assert "XID-94-CONTAINED_ERROR" not in symptoms


class TestKernelFailureCascade:
    """End-to-end tests for kernel panic, invalid-memory-access, and oops rules.

    Per-line extractor matching is covered declaratively in
    ``syslog_rules.test.toml`` (including the slurmd-debug printk-anchor
    false-positive guard). These tests cover the parts the declarative
    format can't express: full pipeline resolution to ``Fail InfraSW``
    via outcome calculation, and coexistence of both kernel symptoms on
    a real crash burst.
    """

    @pytest.fixture(scope="class")
    def job_info(self, simple_job_factory):
        return simple_job_factory(
            state="NODE_FAIL",
            framework="pytorch",
            ranks_per_node=4,
            start_timestamp=1746000000,
            end_timestamp=1746003600,  # 1h job
            nodes=["test-node-001"],
        )

    def _run_attribution(self, request, sys_tst_bldr, job_info, lines, *, include_node_failure=False):
        # Time-stamp lines mid-job so the SYSLOG source-config window keeps them.
        base_ns = int(job_info.job_start_time.timestamp() * 1e9) + 1800 * 10**9
        entries = [
            make_syslog_entry(body=line, node="test-node-001", timestamp_ns=base_ns + i * 1_000_000)
            for i, line in enumerate(lines)
        ]
        result = sys_tst_bldr(job_info, entries, stream_id=request.node.name)
        bad_nodes = core.CalcBadNodesAttributionResult()
        bad_nodes.my_bad_nodes = result.get_node_attributions()
        if include_node_failure:
            source = job_info.source(
                f"{job_info.state}test-node-001",
                "Node failure reported by slurm",
            )
            bad_nodes.my_bad_nodes.add(
                ["test-node-001"],
                SourceSet.new_one("Node failure reported by slurm", source),
            )
        return core.attribute(
            event_stream=result.event_stream,
            job_info=result.job_info,
            bad_node_result=bad_nodes,
            extract_job_metadata_from_logfile=False,
        )

    def test_soft_lockup_panic_resolves_to_fail_infra_sw(self, request, sys_tst_bldr, job_info):
        """The observed cloud-platform GB200 panic wording produces the specific soft-lockup symptom."""
        attribution = self._run_attribution(
            request,
            sys_tst_bldr,
            job_info,
            ["[36621.012637] Kernel panic - not syncing: softlockup: hung tasks\r\n"],
            include_node_failure=True,
        )
        assert "kernel soft lockup" in attribution.symptoms
        assert "kernel panic" not in attribution.symptoms
        assert "Node failure reported by slurm" not in attribution.symptoms
        assert attribution.outcome_attr.value().outcome == "Fail InfraSW"

    def test_other_kernel_panic_resolves_to_fail_infra_sw(self, request, sys_tst_bldr, job_info):
        """An unrecognized panic reason is retained by the generic fallback."""
        attribution = self._run_attribution(
            request,
            sys_tst_bldr,
            job_info,
            ["[42.000000] Kernel panic - not syncing: fatal exception"],
            include_node_failure=True,
        )
        assert "kernel panic" in attribution.symptoms
        assert "kernel soft lockup" not in attribution.symptoms
        assert "Node failure reported by slurm" not in attribution.symptoms
        assert attribution.outcome_attr.value().outcome == "Fail InfraSW"

    def test_invalid_mem_access_resolves_to_fail_infra_sw(self, request, sys_tst_bldr, job_info):
        """``Unable to handle kernel`` alone produces Fail InfraSW."""
        attribution = self._run_attribution(
            request,
            sys_tst_bldr,
            job_info,
            ["[3588119.522055] Unable to handle kernel NULL pointer dereference at virtual address 0000000000000008"],
        )
        assert "kernel invalid memory access" in attribution.symptoms
        assert attribution.outcome_attr.value().outcome == "Fail InfraSW"

    def test_oops_alone_resolves_to_fail_infra_sw(self, request, sys_tst_bldr, job_info):
        """Bare ``Internal error: Oops`` (no printk prefix, no ``Unable to handle kernel``)
        still fires ``kernel oops`` -> Fail InfraSW - the pure-oops fallback; nothing to dedup."""
        attribution = self._run_attribution(
            request,
            sys_tst_bldr,
            job_info,
            ["Internal error: Oops: 00000000bf000002 [#1] SMP"],
        )
        assert "kernel oops" in attribution.symptoms
        assert attribution.outcome_attr.value().outcome == "Fail InfraSW"

    def test_oops_fpac_resolves_to_fail_infra_sw(self, request, sys_tst_bldr, job_info):
        """ARM64 tagged form ``Internal error: Oops - FPAC:`` (cloud-region serial console)
        must also fire ``kernel oops`` -> Fail InfraSW."""
        attribution = self._run_attribution(
            request,
            sys_tst_bldr,
            job_info,
            ["[2288200.381318] Internal error: Oops - FPAC: 0000000072000000 [#1]  SMP"],
        )
        assert "kernel oops" in attribution.symptoms
        assert attribution.outcome_attr.value().outcome == "Fail InfraSW"

    def test_full_crash_burst_dedups_oops_to_invalid_mem_access(self, request, sys_tst_bldr, job_info):
        """Both ``Unable to handle kernel`` and ``Internal error: Oops`` print for one crash;
        the generic ``kernel oops`` is deduped in favour of ``kernel invalid memory access``."""
        attribution = self._run_attribution(
            request,
            sys_tst_bldr,
            job_info,
            [
                "[3588119.522055] Unable to handle kernel NULL pointer dereference at virtual address 0000000000000008",
                "[3588119.546745] Internal error: Oops: 0000000096000007 [#1] SMP",
                "[3588119.629144] Call trace:",
                "[3588119.602564] pc : mas_next_slot+0xb4/0x458",
            ],
        )
        assert "kernel invalid memory access" in attribution.symptoms
        assert "kernel oops" not in attribution.symptoms
        assert attribution.outcome_attr.value().outcome == "Fail InfraSW"

    def test_no_prefix_lines_resolve_to_fail_infra_sw(self, request, sys_tst_bldr, job_info):
        """Crash lines without a printk prefix produce one event, so
        ``kernel oops`` is deduped and the job flips to Fail InfraSW via invalid memory access."""
        attribution = self._run_attribution(
            request,
            sys_tst_bldr,
            job_info,
            [
                "Unable to handle kernel paging request at virtual address 0000001ff8a2e988",
                "Internal error: Oops: 0000000096000006 [#1]  SMP",
            ],
        )
        assert "kernel invalid memory access" in attribution.symptoms
        assert "kernel oops" not in attribution.symptoms
        assert attribution.outcome_attr.value().outcome == "Fail InfraSW"

    def test_slurmd_oops_handler_debug_string_does_not_fire(self, request, sys_tst_bldr, job_info):
        """The slurmd debug string "Internal error: Oops handler" (no colon)
        must NOT produce a kernel-oops symptom, even with a printk-style prefix."""
        attribution = self._run_attribution(
            request,
            sys_tst_bldr,
            job_info,
            ["[1234.567890] slurmd: Internal error: Oops handler invoked but no callback registered"],
        )
        assert "kernel oops" not in attribution.symptoms

    def test_ext4_data_loss_resolves_to_fail_infra_sw(self, request, sys_tst_bldr, job_info):
        """The delayed-allocation failure pair from a corrupted node-local md0 produces Fail InfraSW."""
        attribution = self._run_attribution(
            request,
            sys_tst_bldr,
            job_info,
            [
                "<2>1 2026-10-15T21:53:07.933349-07:00 test-node-001 kernel - - -  EXT4-fs (md0): Delayed block "
                "allocation failed for inode 285219254 at logical offset 0 with max blocks 1 with error 117",
                "<2>1 2026-10-15T21:53:07.933746-07:00 test-node-001 kernel - - -  EXT4-fs (md0): This should not "
                "happen!! Data will be lost",
            ],
        )
        assert "ext4 filesystem corruption" in attribution.symptoms
        outcome = attribution.outcome_attr.value()
        assert outcome.outcome == "Fail InfraSW"
        assert outcome.root_cause_symptom == "ext4 filesystem corruption"

    def test_ext4_enospc_data_loss_does_not_fire(self, request, sys_tst_bldr, job_info):
        """A delayed-allocation failure on a full disk (ENOSPC) is not corruption."""
        attribution = self._run_attribution(
            request,
            sys_tst_bldr,
            job_info,
            [
                "EXT4-fs (md0): Delayed block allocation failed for inode 285219254 at logical offset 0 with max "
                "blocks 1 with error 28",
                "EXT4-fs (md0): This should not happen!! Data will be lost",
            ],
        )
        assert "ext4 filesystem corruption" not in attribution.symptoms


class TestXid140Outcome:
    """Exercise XID-140 classification and node-scoped cleanup from raw syslog."""

    _XID140 = "XID-140-UNRECOVERABLE_ECC_ERROR_ESCAPE"
    _XID94_GSP = "XID-94-CONTAINED_ERROR-GSP"
    _FATAL = "NV_PGSP_FALCON_IRQSTAT_FATAL_ERROR"
    _LINES = (
        "NVRM: Xid (PCI:0000:06:00): 140, pid=1234, name=test, Poison error in GSP.",
        "NVRM: kgspServiceFatalHwError_GB100: NV_PGSP_FALCON_IRQSTAT_FATAL_ERROR FB Poison pending (mask: 0x1)",
        "NVRM: Xid (PCI:0000:06:00): 94, Contained: GSP. RST: Yes, D-RST: No",
        "NVRM: nvCheckOkFailedNoLog: Check failed: [NV_ERR_RESET_REQUIRED] (0x00000062) @ kernel_gsp.c:100",
    )

    @pytest.fixture
    def attribute(self, request, simple_job_factory, sys_tst_bldr):
        """Build an attribution from timestamped syslog on two allocated nodes.

        Args:
            request: Pytest request providing the stream identifier.
            simple_job_factory: Fixture constructing job metadata.
            sys_tst_bldr: Fixture running real syslog extraction.

        Returns:
            Callable accepting node/message pairs and returning an attribution.
        """
        job_info = simple_job_factory(
            state="FAILED",
            framework="pytorch",
            ranks_per_node=4,
            start_timestamp=1746000000,
            end_timestamp=1746003600,
            nodes=["test-node-001", "test-node-002"],
        )

        def run(node_lines):
            """Attribute supplied syslog through the full pipeline.

            Args:
                node_lines: Node/message pairs in observation order.

            Returns:
                Attribution result with outcomes and per-node evidence.
            """
            base_ns = int(job_info.job_start_time.timestamp() * 1e9) + 1800 * 10**9
            entries = [
                make_syslog_entry(body=line, node=node, timestamp_ns=base_ns + i * 1_000_000)
                for i, (node, line) in enumerate(node_lines)
            ]
            result = sys_tst_bldr(job_info, entries, stream_id=request.node.name)
            bad_nodes = core.CalcBadNodesAttributionResult()
            bad_nodes.my_bad_nodes = result.get_node_attributions()
            return core.attribute(
                event_stream=result.event_stream,
                job_info=result.job_info,
                bad_node_result=bad_nodes,
                extract_job_metadata_from_logfile=False,
            )

        return run

    @pytest.mark.parametrize("indices", [(0,), (0, 1), (0, 2), (0, 1, 2), (0, 1, 2, 3), (3, 2, 1, 0), (0, 1, 2, 3) * 2])
    def test_xid140_is_infra_hw_and_retains_cascade_evidence(self, attribute, indices):
        """Promote XID-140 and demote its two supporting signals on the same node.

        Args:
            attribute: Fixture running extraction and attribution.
            indices: Message sequence, including isolated, reordered and repeated signals.

        Returns:
            None.
        """
        attribution = attribute([("test-node-001", self._LINES[i]) for i in indices])

        outcome = attribution.outcome_attr.value()
        assert outcome.outcome == "Fail InfraHW"
        assert outcome.root_cause_symptom == self._XID140
        match = attribution_lib.get_symptom_registry().outcome_for([self._XID140])
        assert match.doc == "xid-140"
        assert match.hw_cause == "Known HW Cause - Node"
        node_symptoms = attribution.bad_nodes_attr.my_bad_nodes["test-node-001"].attributions
        for symptoms in [attribution.symptoms, node_symptoms]:
            assert self._XID140 in symptoms
            assert "poisoned gsp" not in symptoms
            assert self._FATAL not in symptoms
            assert "XID-94-CONTAINED_ERROR" not in symptoms
            assert self._XID94_GSP not in symptoms
        for index, symptom in [(1, self._FATAL), (2, self._XID94_GSP)]:
            if index in indices:
                assert node_symptoms[symptom].has_interesting()
                assert len(list(node_symptoms[symptom].all_sources())) == indices.count(index)
        assert len(list(node_symptoms[self._XID140])) == indices.count(0)
        if 3 in indices:
            assert "NV_ERR_RESET_REQUIRED kernel_gsp.c" in node_symptoms

    @pytest.mark.parametrize("with_reset", [False, True])
    def test_xid140_preserves_other_nodes_evidence(self, attribute, with_reset):
        """Keep another node's fatal/94 evidence or poisoned-gsp diagnosis intact.

        Args:
            attribute: Fixture running extraction and attribution.
            with_reset: Whether the other node has a complete poisoned-gsp trio.

        Returns:
            None.
        """
        other_lines = self._LINES[1:] if with_reset else self._LINES[1:3]
        attribution = attribute([("test-node-001", self._LINES[0]), *[("test-node-002", line) for line in other_lines]])

        assert attribution.outcome_attr.value().outcome == "Fail InfraHW"
        other = attribution.bad_nodes_attr.my_bad_nodes["test-node-002"].attributions
        if with_reset:
            assert "poisoned gsp" in other
            assert {source.datum for source in other["poisoned gsp"]} == set(other_lines)
        else:
            assert self._FATAL in other
            assert self._XID94_GSP in other
            assert self._FATAL in attribution.symptoms
            assert self._XID94_GSP in attribution.symptoms


class TestPoisonedGspOutcome:
    """Exercise the three-signal ECC diagnosis from syslog through attribution."""

    _LINES = (
        "NVRM: kgspServiceFatalHwError_GB100: NV_PGSP_FALCON_IRQSTAT_FATAL_ERROR FB Poison pending (mask: 0x1)",
        "NVRM: Xid (PCI:0000:06:00): 94, Contained: GSP. RST: Yes, D-RST: No",
        "NVRM: nvCheckOkFailedNoLog: Check failed: [NV_ERR_RESET_REQUIRED] (0x00000062) @ kernel_gsp.c:100",
    )

    @pytest.fixture(scope="class")
    def job_info(self, simple_job_factory):
        return simple_job_factory(
            state="FAILED",
            framework="pytorch",
            ranks_per_node=4,
            start_timestamp=1746000000,
            end_timestamp=1746003600,
            nodes=["test-node-001", "test-node-002"],
        )

    def _attribute(self, request, sys_tst_bldr, job_info, node_lines, offsets_ns=None):
        base_ns = int(job_info.job_start_time.timestamp() * 1e9)
        if offsets_ns is None:
            offsets_ns = [1800 * 10**9 + i * 1_000_000 for i in range(len(node_lines))]
        entries = [
            make_syslog_entry(body=line, node=node, timestamp_ns=base_ns + offset_ns)
            for (node, line), offset_ns in zip(node_lines, offsets_ns, strict=True)
        ]
        result = sys_tst_bldr(job_info, entries, stream_id=request.node.name)
        bad_nodes = core.CalcBadNodesAttributionResult()
        bad_nodes.my_bad_nodes = result.get_node_attributions()
        return core.attribute(
            event_stream=result.event_stream,
            job_info=result.job_info,
            bad_node_result=bad_nodes,
            extract_job_metadata_from_logfile=False,
        )

    @pytest.mark.parametrize("reverse", [False, True])
    def test_all_three_signals_resolve_to_ecc_infra_hw(self, request, sys_tst_bldr, job_info, reverse):
        lines = self._LINES[::-1] if reverse else self._LINES
        attribution = self._attribute(request, sys_tst_bldr, job_info, [("test-node-001", line) for line in lines])

        assert set(attribution.symptoms.symptoms()) == {"poisoned gsp"}
        outcome = attribution.outcome_attr.value()
        assert outcome.outcome == "Fail InfraHW"
        assert outcome.root_cause_symptom == "poisoned gsp"
        sources = list(attribution.symptoms["poisoned gsp"])
        assert {source.datum for source in sources} == set(self._LINES)
        assert {source.rule for source in sources} == {"SyslogGspFatalError", "known-xid-event", "NV_ERR_"}
        registry = attribution_lib.get_symptom_registry()
        match = registry.outcome_for(["poisoned gsp"])
        assert match.hw_cause == "Known HW Cause - Node"
        assert match.doc == "ecc-error"

    @pytest.mark.parametrize("suffix", ["", " (0x00000062) @ kernel_gsp.c:100"])
    def test_unrelated_reset_code_stays_active_alongside_complete_combination(
        self, request, sys_tst_bldr, job_info, suffix
    ):
        unrelated_line = "NVRM: [NV_ERR_RESET_REQUIRED_OTHER]" + suffix
        attribution = self._attribute(
            request,
            sys_tst_bldr,
            job_info,
            [("test-node-001", line) for line in [*self._LINES, unrelated_line]],
        )

        node_symptoms = attribution.bad_nodes_attr.my_bad_nodes["test-node-001"].attributions
        for symptoms in [attribution.symptoms, node_symptoms]:
            unrelated = [symptom for symptom in symptoms.symptoms() if "NV_ERR_RESET_REQUIRED_OTHER" in symptom]
            assert len(unrelated) == 1
            assert {source.datum for source in symptoms[unrelated[0]]} == {unrelated_line}
            assert {source.datum for source in symptoms["poisoned gsp"]} == set(self._LINES)
        assert attribution.outcome_attr.value().root_cause_symptom == "poisoned gsp"

    def test_repeated_signals_produce_one_composite_with_all_sources(self, request, sys_tst_bldr, job_info):
        attribution = self._attribute(
            request,
            sys_tst_bldr,
            job_info,
            [("test-node-001", line) for line in self._LINES * 2],
        )

        assert list(attribution.symptoms.symptoms()) == ["poisoned gsp"]
        sources = list(attribution.symptoms["poisoned gsp"])
        assert len(sources) == 6
        for line in self._LINES:
            assert sum(source.datum == line for source in sources) == 2

    def test_signals_correlate_across_the_node_job_evidence_window(self, request, sys_tst_bldr, job_info):
        attribution = self._attribute(
            request,
            sys_tst_bldr,
            job_info,
            [("test-node-001", line) for line in self._LINES],
            offsets_ns=[10**9, 1800 * 10**9, 3599 * 10**9],
        )

        assert set(attribution.symptoms.symptoms()) == {"poisoned gsp"}
        assert {source.datum for source in attribution.symptoms["poisoned gsp"]} == set(self._LINES)
        assert attribution.outcome_attr.value().outcome == "Fail InfraHW"

    @pytest.mark.parametrize("xid", [171, 172])
    def test_existing_memory_error_keeps_root_cause_precedence(self, request, sys_tst_bldr, job_info, xid):
        memory_error = f"NVRM: Xid (PCI:0000:06:00): {xid}, Uncorrectable memory error"
        attribution = self._attribute(
            request,
            sys_tst_bldr,
            job_info,
            [("test-node-001", line) for line in [*self._LINES, memory_error]],
        )

        outcome = attribution.outcome_attr.value()
        assert outcome.outcome == "Fail InfraHW"
        assert outcome.root_cause_symptom.startswith(f"XID-{xid}-")

    @pytest.mark.parametrize("indices", [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)])
    def test_incomplete_combination_does_not_produce_poisoned_gsp(self, request, sys_tst_bldr, job_info, indices):
        attribution = self._attribute(
            request,
            sys_tst_bldr,
            job_info,
            [("test-node-001", self._LINES[i]) for i in indices],
        )

        assert "poisoned gsp" not in attribution.symptoms

    @pytest.mark.parametrize("other_node_index", [0, 1, 2])
    def test_signals_on_different_nodes_do_not_combine(self, request, sys_tst_bldr, job_info, other_node_index):
        attribution = self._attribute(
            request,
            sys_tst_bldr,
            job_info,
            [
                ("test-node-002" if i == other_node_index else "test-node-001", line)
                for i, line in enumerate(self._LINES)
            ],
        )

        assert "poisoned gsp" not in attribution.symptoms
        assert attribution.outcome_attr.value().outcome != "Fail InfraHW"

    @pytest.mark.parametrize(
        "replacement",
        [
            "NVRM: Xid (PCI:0000:06:00): 154, GPU recovery action changed from 0x0 (None) to 0x1 (GPU Reset Required)",
            "NVRM: [NV_ERR_RESET_REQUIRED_OTHER]",
        ],
    )
    def test_other_reset_messages_do_not_complete_combination(self, request, sys_tst_bldr, job_info, replacement):
        attribution = self._attribute(
            request,
            sys_tst_bldr,
            job_info,
            [("test-node-001", line) for line in [*self._LINES[:2], replacement]],
        )

        assert "poisoned gsp" not in attribution.symptoms
        assert attribution.outcome_attr.value().outcome != "Fail InfraHW"


class TestBadNicStateOutcome:
    """Correlate public mlx5 messages without mixing nodes, NICs, or stale evidence."""

    _LINES = (
        "mlx5_core 0000:01:00.0: wait_fw_init:201:(pid 10): "
        "Firmware over 10000 MS in initializing state, aborting",
        "mlx5_core 0000:01:00.0: probe_one:2002:(pid 10): mlx5_init_one failed with error code -110",
    )
    _COMPONENTS = ("mlx5 firmware in initialization state", "mlx5 device initialization failed")
    _QP = "Call to ibv_modify_qp failed"

    @pytest.fixture(scope="class")
    def job_info(self, simple_job_factory):
        return simple_job_factory(
            state="FAILED",
            framework="pytorch",
            ranks_per_node=4,
            start_timestamp=1746000000,
            end_timestamp=1746003600,
            nodes=["test-node-001", "test-node-002"],
        )

    def _attribute(
        self, request, sys_tst_bldr, job_info, node_lines, offsets, extra_symptoms=()
    ):
        end_ns = int(job_info.job_end_time.timestamp() * 1e9)
        entries = [
            make_syslog_entry(body=line, node=node, timestamp_ns=end_ns + int(offset * 1e9))
            for (node, line), offset in zip(node_lines, offsets, strict=True)
        ]
        # Match the producer's NIC post-job observation contract; the default
        # observation cutoff is covered separately below.
        result = sys_tst_bldr(
            job_info,
            entries,
            stream_id=request.node.name,
            time_interval=TimeInterval(job_info.job_start_time, job_info.job_end_time + timedelta(minutes=2)),
        )
        bad_nodes = core.CalcBadNodesAttributionResult()
        bad_nodes.my_bad_nodes = result.get_node_attributions()
        for symptom in extra_symptoms:
            bad_nodes.my_bad_nodes.add(
                ["test-node-001"],
                SourceSet.new_one(
                    symptom,
                    job_info.source(symptom, "test")._replace(time=job_info.job_end_time - timedelta(seconds=1)),
                ),
            )
        return core.attribute(
            event_stream=result.event_stream,
            job_info=result.job_info,
            bad_node_result=bad_nodes,
            extract_job_metadata_from_logfile=False,
        )

    @pytest.mark.parametrize("reverse,extra_symptoms", [(False, ()), (True, ()), (False, (_QP,))])
    def test_pair_near_job_end_retains_both_sources(
        self, request, sys_tst_bldr, job_info, reverse, extra_symptoms
    ):
        lines = self._LINES[::-1] if reverse else self._LINES
        result = self._attribute(
            request,
            sys_tst_bldr,
            job_info,
            [("test-node-001", line) for line in lines],
            [4, 4.002],
            extra_symptoms=extra_symptoms,
        )

        assert set(result.symptoms.symptoms()) == {"detected bad NIC state", *extra_symptoms}
        assert result.outcome_attr.value().outcome == "Fail InfraSW"
        assert result.outcome_attr.value().root_cause_symptom == "detected bad NIC state"
        sources = list(result.symptoms["detected bad NIC state"])
        assert {source.datum for source in sources} == set(self._LINES)
        assert {source.rule for source in sources} == {"SyslogMlx5FirmwareInitTimeout", "SyslogMlx5InitFailure"}
        node_symptoms = result.bad_nodes_attr.my_bad_nodes["test-node-001"].attributions
        for symptom in extra_symptoms:
            assert {source.datum for source in result.symptoms[symptom]} == {symptom}
            assert {source.datum for source in node_symptoms[symptom]} == {symptom}
        for component in self._COMPONENTS:
            assert component not in node_symptoms
            assert node_symptoms[component].has_interesting()

    @pytest.mark.parametrize("offsets", [(-1, 0), (0, 121)])
    def test_outside_window_does_not_combine(self, request, sys_tst_bldr, job_info, offsets):
        result = self._attribute(
            request,
            sys_tst_bldr,
            job_info,
            [("test-node-001", line) for line in self._LINES],
            offsets,
        )
        assert "detected bad NIC state" not in result.symptoms

    @pytest.mark.parametrize("index", [0, 1])
    def test_single_signal_does_not_produce_compound_outcome(self, request, sys_tst_bldr, job_info, index):
        result = self._attribute(
            request, sys_tst_bldr, job_info, [("test-node-001", self._LINES[index])], [4]
        )
        assert "detected bad NIC state" not in result.symptoms
        assert result.outcome_attr.value().outcome != "Fail InfraSW"

    @pytest.mark.parametrize(
        "other_node,other_pci",
        [("test-node-002", "0000:01:00.0"), ("test-node-001", "0000:02:00.0")],
    )
    def test_different_nodes_or_devices_do_not_combine(
        self, request, sys_tst_bldr, job_info, other_node, other_pci
    ):
        result = self._attribute(
            request,
            sys_tst_bldr,
            job_info,
            [
                ("test-node-001", self._LINES[0]),
                (other_node, self._LINES[1].replace("0000:01:00.0", other_pci)),
            ],
            [4, 4.002],
        )
        assert "detected bad NIC state" not in result.symptoms

    @pytest.mark.parametrize("extra_symptoms", [(), (_QP,)])
    def test_slurm_node_failure_and_bad_nic_state_both_remain_active(
        self, request, sys_tst_bldr, job_info, extra_symptoms
    ):
        job_info = job_info._replace(state="NODE_FAIL", failed_node="test-node-001")
        result = self._attribute(
            request,
            sys_tst_bldr,
            job_info,
            [("test-node-001", line) for line in self._LINES],
            [4, 4.002],
            extra_symptoms=extra_symptoms,
        )

        expected = {"Node failure reported by slurm", "detected bad NIC state"}
        assert set(result.symptoms.symptoms()) == expected
        node_symptoms = result.bad_nodes_attr.my_bad_nodes["test-node-001"].attributions
        assert set(node_symptoms.symptoms()) == expected
        for symptom in expected:
            assert node_symptoms[symptom].has_bad()
            assert result.symptoms[symptom].has_bad()
        for symptoms in (result.symptoms, node_symptoms):
            for symptom in extra_symptoms:
                assert not symptoms[symptom].has_bad()
                assert symptoms[symptom].has_interesting()
                assert {source.datum for source in symptoms[symptom].all_sources()} == {symptom}
        assert {source.datum for source in node_symptoms["detected bad NIC state"]} == set(self._LINES)
        for component in self._COMPONENTS:
            assert component not in node_symptoms
            assert node_symptoms[component].has_interesting()
        outcome = result.outcome_attr.value()
        assert outcome.outcome == "Fail InfraSW"
        assert outcome.root_cause_symptom == "Node failure reported by slurm"

    def test_variable_timeout_and_error_code(self, request, sys_tst_bldr, job_info):
        lines = [self._LINES[0].replace("10000", "20000"), self._LINES[1].replace("-110", "-5")]
        result = self._attribute(
            request,
            sys_tst_bldr,
            job_info,
            [("test-node-001", line) for line in lines],
            [4, 5],
        )
        assert "detected bad NIC state" in result.symptoms

    def test_repeated_pair_preserves_unmatched_device_evidence(self, request, sys_tst_bldr, job_info):
        other = self._LINES[0].replace("0000:01:00.0", "0000:02:00.0")
        lines = [*self._LINES, *self._LINES, other]
        result = self._attribute(
            request,
            sys_tst_bldr,
            job_info,
            [("test-node-001", line) for line in lines],
            [1, 2, 3, 4, 5],
        )
        assert len(result.symptoms["detected bad NIC state"]) == 4
        assert {source.datum for source in result.symptoms[self._COMPONENTS[0]]} == {other}

    @pytest.mark.parametrize(
        "secondary,independent",
        [
            ("ibv_open_device failed", None),
            ("Call to ibv_modify_qp failed", "Node unexpectedly rebooted"),
            ("Call to ibv_modify_qp failed", "Page Retirement"),
        ],
    )
    def test_outcome_precedence_preserves_evidence(
        self, request, sys_tst_bldr, job_info, secondary, independent
    ):
        extra_symptoms = [secondary]
        if independent is not None:
            extra_symptoms.append(independent)
        result = self._attribute(
            request,
            sys_tst_bldr,
            job_info,
            [("test-node-001", line) for line in self._LINES],
            [4, 4.002],
            extra_symptoms=extra_symptoms,
        )

        expected = {"detected bad NIC state", independent or secondary}
        node_symptoms = result.bad_nodes_attr.my_bad_nodes["test-node-001"].attributions
        assert set(result.symptoms.symptoms()) == expected
        assert set(node_symptoms.symptoms()) == expected
        for symptom in expected:
            assert result.symptoms[symptom].has_bad()
            assert node_symptoms[symptom].has_bad()
        for symptoms in (result.symptoms, node_symptoms):
            for symptom in extra_symptoms:
                assert {source.datum for source in symptoms[symptom].all_sources()} == {symptom}
            if independent is not None:
                # Existing IB cleanup demotes generic QP after finding a non-IB root cause.
                assert not symptoms[secondary].has_bad()
                assert symptoms[secondary].has_interesting()
        outcome = result.outcome_attr.value()
        if independent is None:
            expected_outcome = "Fail InfraSW"
        else:
            expected_outcome = "Fail InfraHW"
        assert outcome.outcome == expected_outcome
        assert outcome.root_cause_symptom == (independent or "detected bad NIC state")

    @pytest.mark.parametrize("offset,expected", [(0, True), (4, False)])
    def test_default_observation_window(self, request, sys_tst_bldr, job_info, offset, expected):
        end_ns = int(job_info.job_end_time.timestamp() * 1e9)
        result = sys_tst_bldr(
            job_info,
            [
                make_syslog_entry(body=line, node="test-node-001", timestamp_ns=end_ns + offset * 10**9)
                for line in self._LINES
            ],
            stream_id=request.node.name,
        )
        bad_nodes = core.CalcBadNodesAttributionResult()
        bad_nodes.my_bad_nodes = result.get_node_attributions()
        attribution = core.attribute(
            event_stream=result.event_stream,
            job_info=result.job_info,
            bad_node_result=bad_nodes,
            extract_job_metadata_from_logfile=False,
        )
        assert ("detected bad NIC state" in attribution.symptoms) is expected


class TestFmLostAllGpusOutcome:
    """End-to-end: when all GPUs on a node lose fabric-manager connection, the
    aggregated 'Fabric manager lost connection to all GPUs' symptom resolves to
    Fail InfraHW and outranks the co-morbid user-space symptom it usually
    co-occurs with (HIGH_HW beats LOW_USER)."""

    @pytest.fixture(scope="class")
    def job_info(self, simple_job_factory):
        # slurm COMPLETE is a low-fidelity signal; the node-level FM fault still wins.
        return simple_job_factory(
            state="COMPLETED",
            framework="pytorch",
            ranks_per_node=4,
            start_timestamp=1746000000,
            end_timestamp=1746003600,
            nodes=["test-node-001"],
        )

    def test_fm_lost_all_gpus_resolves_to_fail_infra_hw(self, request, sys_tst_bldr, job_info):
        base_ns = int(job_info.job_start_time.timestamp() * 1e9) + 1800 * 10**9
        lines = [
            f"nmxc-fm: Lost connection to GPU on nodeId {i}, gpuGUID 0x{i:016x}, GPU state Error." for i in range(4)
        ]
        entries = [
            make_syslog_entry(body=line, node="test-node-001", timestamp_ns=base_ns + i * 1_000_000)
            for i, line in enumerate(lines)
        ]
        result = sys_tst_bldr(job_info, entries, stream_id=request.node.name)
        bad_nodes = core.CalcBadNodesAttributionResult()
        bad_nodes.my_bad_nodes = result.get_node_attributions()
        attribution = core.attribute(
            event_stream=result.event_stream,
            job_info=result.job_info,
            bad_node_result=bad_nodes,
            extract_job_metadata_from_logfile=False,
        )
        assert "Fabric manager lost connection to all GPUs" in attribution.symptoms
        assert attribution.outcome_attr.value().outcome == "Fail InfraHW"

    def test_fm_lost_all_gpus_outranks_user_symptom(self):
        # The exact production scenario: the FM fault co-morbid with a user-space
        # symptom must attribute to infra, not user.
        registry = attribution_lib.get_symptom_registry()
        match = registry.outcome_for(["Fabric manager lost connection to all GPUs", "Python Exception"])
        assert match is not None
        assert match.symptom == "Fabric manager lost connection to all GPUs"
        assert match.outcome == "Fail InfraHW"


class TestPodOomCascade:
    """End-to-end tests for pod-slice OOM plus kubelet status attribution."""

    @pytest.fixture(scope="class")
    def job_info(self, simple_job_factory):
        return simple_job_factory(
            state="NODE_FAIL",
            framework="pytorch",
            ranks_per_node=4,
            start_timestamp=1746000000,
            end_timestamp=1746003600,
            nodes=["test-node-001"],
        )

    def test_slurmd_kubelet_status_overrides_pod_slice_user_oom(self, request, sys_tst_bldr, job_info):
        base_ns = int(job_info.job_start_time.timestamp() * 1e9) + 1800 * 10**9
        lines = [
            "kubepods-besteffort-pod239fec60_2cf7_4396_827c_dc8167f7a1b1.slice: "
            "A process of this unit has been killed by the OOM killer.",
            'Patch status for pod containerStatuses:[{"name":"slurmd","ready":false,'
            '"lastState":{"terminated":{"exitCode":137,"reason":"OOMKilled"}}}]',
        ]
        entries = [
            make_syslog_entry(body=line, node="test-node-001", timestamp_ns=base_ns + i * 1_000_000)
            for i, line in enumerate(lines)
        ]
        result = sys_tst_bldr(job_info, entries, stream_id=request.node.name)
        bad_nodes = core.CalcBadNodesAttributionResult()
        bad_nodes.my_bad_nodes = result.get_node_attributions()

        attribution = core.attribute(
            event_stream=result.event_stream,
            job_info=result.job_info,
            bad_node_result=bad_nodes,
            extract_job_metadata_from_logfile=False,
        )

        assert attribution.outcome_attr is not None
        outcome = attribution.outcome_attr.value()
        assert outcome.outcome == "Fail InfraSW"
        assert outcome.root_cause_symptom == "OOM kill of a system pod"
        assert RegisteredSymptoms.OOM_KILL_OF_SYSTEM_POD.value in attribution.symptoms
        assert RegisteredSymptoms.USER_TASK_OOM_KILL.value not in attribution.symptoms

    def _run_kubelet_symptoms(self, request, sys_tst_bldr, job_info, line):
        # Stamp the kubelet line mid-job so the SYSLOG source-config window keeps
        # it: the kubelet re-publishes the same pod status continuously, so the
        # *line* time is always within the window even when the OOM is stale.
        base_ns = int(job_info.job_start_time.timestamp() * 1e9) + 1800 * 10**9
        entries = [make_syslog_entry(body=line, node="test-node-001", timestamp_ns=base_ns)]
        result = sys_tst_bldr(job_info, entries, stream_id=request.node.name)
        nodes = result.get_node_attributions()
        # A suppressed OOM leaves the node with no attributions at all, so the
        # node may be absent from the result entirely.
        group = nodes.attributions.get("test-node-001")
        return list(group.attributions.symptoms()) if group is not None else []

    @staticmethod
    def _kubelet_slurmd_oom_line(finished_at):
        # Mirrors the real "Ignoring same status for pod" re-publication: slurmd has
        # recovered (state.running) and the OOM lives only in lastState.terminated.
        return (
            'Ignoring same status for pod containerStatuses:[{"name":"slurmd","ready":true,'
            '"state":{"running":{"startedAt":"2099-01-01T00:00:00Z"}},'
            '"lastState":{"terminated":{"exitCode":137,"reason":"OOMKilled",'
            f'"startedAt":"2020-01-01T00:00:00Z","finishedAt":"{finished_at}"}}}}]'
        )

    def test_stale_kubelet_oom_outside_window_does_not_fire(self, request, sys_tst_bldr, job_info):
        """A kubelet status re-publishing a days-old slurmd OOM (finishedAt far
        before job start) must NOT be attributed to this job. This regression
        verifies that one stale June-22 OOM is not re-tagged onto every later job."""
        line = self._kubelet_slurmd_oom_line("2020-01-01T00:00:00Z")
        symptoms = self._run_kubelet_symptoms(request, sys_tst_bldr, job_info, line)
        assert RegisteredSymptoms.OOM_KILL_OF_SYSTEM_POD.value not in symptoms

    def test_kubelet_oom_within_window_fires(self, request, sys_tst_bldr, job_info):
        """An OOM whose finishedAt falls inside the job's execution window is a
        legitimate attribution and must still fire."""
        finished_at = (job_info.job_start_time + timedelta(seconds=1800)).isoformat()
        line = self._kubelet_slurmd_oom_line(finished_at)
        symptoms = self._run_kubelet_symptoms(request, sys_tst_bldr, job_info, line)
        assert RegisteredSymptoms.OOM_KILL_OF_SYSTEM_POD.value in symptoms

    def test_kubelet_oom_timezone_naive_finished_at_does_not_crash(self, request, sys_tst_bldr, job_info):
        """A finishedAt without a timezone offset parses to a naive datetime, which
        cannot be compared against the always-tz-aware job_start_time (TypeError).
        We can't gate on an unusable timestamp, so the staleness check must fall
        back to attributing the OOM rather than raising out of the rule."""
        line = self._kubelet_slurmd_oom_line("2020-01-01T00:00:00")  # no Z / offset
        symptoms = self._run_kubelet_symptoms(request, sys_tst_bldr, job_info, line)
        assert RegisteredSymptoms.OOM_KILL_OF_SYSTEM_POD.value in symptoms


def test_node_filtering_logging(simple_job_factory, sys_tst_bldr):
    """Test that syslog lines from nodes not in job are logged and ignored."""
    # Create job info with limited node list
    job_info = simple_job_factory(
        state="RUNNING",
        nodes=["job-node-001", "job-node-002"],  # Only these nodes are in the job
    )

    # Create test syslog data with helper function
    test_syslog_data = [
        make_syslog_entry(
            body="Nov 15 10:30:00 job-node-001 kernel: NVRM: Xid (PCI:0000:bf:00): 79, pid=1234, name=test",
            node="job-node-001",
            timestamp_ns=1700005000000000000,
        ),
        make_syslog_entry(
            body="Nov 15 10:31:00 other-node-999 kernel: NVRM: Xid (PCI:0000:bf:00): 95, pid=5678, name=test",
            node="other-node-999",
            timestamp_ns=1700005100000000000,
        ),
        make_syslog_entry(
            body="Nov 15 10:32:00 job-node-002 kernel: System is powering down",
            node="job-node-002",
            timestamp_ns=1700005200000000000,
        ),
    ]

    # Mock the node filtering logger to capture debug calls
    with patch("attribution_lib.logline_filtering.logger.logger") as mock_logger:
        result = sys_tst_bldr(job_info, test_syslog_data)

        # Verify that debug logging was called for the ignored node
        mock_logger.debug.assert_called()
        debug_calls = mock_logger.debug.call_args_list

        # Check that one of the debug calls was for the ignored node
        ignored_node_logged = False
        for call in debug_calls:
            args, kwargs = call
            if len(args) > 0 and "other-node-999" in str(args[0]):
                ignored_node_logged = True
                break

        assert ignored_node_logged, "Expected debug log for ignored node 'other-node-999'"

    # Extract nodes from event stream
    nodes = result.get_node_attributions()

    # Verify that valid nodes were processed correctly
    assert "job-node-001" in nodes.attributions
    assert "job-node-002" in nodes.attributions

    # Verify that ignored node is NOT in attributions
    assert "other-node-999" not in nodes.attributions

    # Verify expected symptoms from valid nodes
    assert "XID-79-GPU_HAS_FALLEN_OFF_THE_BUS" in nodes.attributions["job-node-001"].attributions
    assert "System is powering down" in nodes.attributions["job-node-002"].attributions


def test_node_filtering_logging_extraction_failure(simple_job_factory, sys_tst_bldr):
    """Test logging when node extraction fails."""
    # Create job info
    job_info = simple_job_factory(
        nodes=["valid-node"],
    )

    # No declared node and none in the body, so the entry is dropped as unidentifiable.
    test_syslog_data = [{"ts": 1700005000000000000, "body": "Malformed log line without proper hostname"}]

    # Mock only the node filtering logger to capture debug calls, but let natural flow cause UnidentifiableNodeError
    with patch("attribution_lib.logline_filtering.logger.logger") as mock_logger:
        _ = sys_tst_bldr(job_info, test_syslog_data)

        # Verify that debug logging was called for the node that couldn't be extracted
        mock_logger.debug.assert_called()
        debug_calls = mock_logger.debug.call_args_list

        # Should have at least one debug call for the unidentifiable node
        debug_logged = any(len(call.args) > 0 and "Source.SYSLOG ignored:" in str(call.args[0]) for call in debug_calls)

        assert debug_logged, "Expected debug log for unidentifiable node"


def test_xid_pci_mapping_extraction(simple_job_factory, sys_tst_bldr):
    """Test XID PCI mapping extraction."""
    job_info = simple_job_factory()

    test_syslog_data = [
        make_syslog_entry(
            body="Nov 15 10:30:00 test-node-001 kernel: NVRM: Xid (PCI:0000:bf:00): 79, pid=1234, name=test, GPU has fallen off the bus.",
            node="test-node-001",
            timestamp_ns=1700005000000000000,
        ),
        make_syslog_entry(
            body="Nov 15 10:31:00 test-node-001 kernel: NVRM: Xid (PCI:0000:e4:00): 119, pid=5678, name=python, Timeout after 45s of waiting for RPC response.",
            node="test-node-001",
            timestamp_ns=1700005100000000000,
        ),
    ]

    result = sys_tst_bldr(job_info, test_syslog_data)

    nodes = result.get_node_attributions()
    assert "XID-79-GPU_HAS_FALLEN_OFF_THE_BUS" in nodes.attributions["test-node-001"].attributions

    xid_pci_mappings = SyslogAttributionParams.collect_xid_pci_mapping(result.event_stream)
    assert xid_pci_mappings["XID-79"] == "test-node-001:0000:bf:00"
    assert xid_pci_mappings["XID-119"] == "test-node-001:0000:e4:00"


def test_xid_150_omits_error_status_from_symptom_and_pci_mapping(simple_job_factory, sys_tst_bldr):
    """Verify XID 150 omits error status from both emitted identifiers."""
    job_info = simple_job_factory()
    test_syslog_data = [
        make_syslog_entry(
            body=(
                "NVRM: Xid (PCI:0000:01:00): 150, SAW_MSEFW Fatal XC0 i0 Link 0 "
                "(0x00000002 0xffffffff 0x00000000 0x00000000 0x00000000 0x00000000)"
            ),
            node="test-node-001",
            timestamp_ns=1700005000000000000,
        ),
    ]

    result = sys_tst_bldr(job_info, test_syslog_data)
    nodes = result.get_node_attributions()
    symptoms = set(nodes.attributions["test-node-001"].attributions.symptoms())

    assert "XID-150-SAW_MSEFW" in symptoms
    assert "XID-150-SAW_MSEFW-4294967295" not in symptoms

    xid_pci_mappings = SyslogAttributionParams.collect_xid_pci_mapping(result.event_stream)
    assert xid_pci_mappings["XID-150-SAW_MSEFW"] == "test-node-001:0000:01:00"
    assert "XID-150-SAW_MSEFW-4294967295" not in xid_pci_mappings


def test_xid_154_action_matches_public_catalog(simple_job_factory, sys_tst_bldr):
    """Verify XID 154 reports the action encoded in the driver message."""

    job_info = simple_job_factory(start_timestamp=1700000000, end_timestamp=1700010000, nodes=["test-node-001"])
    test_syslog_data = [
        make_syslog_entry(
            body=(
                "NVRM: Xid (PCI:0009:01:00): 154, GPU recovery action changed from 0x0 (None) to 0x4 (Drain and Reset)"
            ),
            node="test-node-001",
            timestamp_ns=1700005000000000000,
        ),
    ]

    result = sys_tst_bldr(job_info, test_syslog_data)
    recovery_actions, investigatory_actions = SyslogAttributionParams.collect_xidparser_actions(result.event_stream)

    assert set(recovery_actions.symptoms()) == {"Drain and Reset"}
    assert set(investigatory_actions.symptoms()) == set()


def test_xid_contact_support_action_is_tbd(simple_job_factory, sys_tst_bldr):
    """Verify ``CONTACT_SUPPORT`` guidance is exposed as ``TBD``."""

    job_info = simple_job_factory(start_timestamp=1700000000, end_timestamp=1700010000, nodes=["test-node-001"])
    test_syslog_data = [
        make_syslog_entry(
            body="NVRM: Xid (PCI:0009:01:00): 79, GPU has fallen off the bus",
            node="test-node-001",
            timestamp_ns=1700005000000000000,
        ),
    ]

    result = sys_tst_bldr(job_info, test_syslog_data)
    recovery_actions, investigatory_actions = SyslogAttributionParams.collect_xidparser_actions(result.event_stream)

    assert set(recovery_actions.symptoms()) == {"RESTART_BM"}
    assert set(investigatory_actions.symptoms()) == {"TBD"}


def test_xid_processor_ignores_sxid_catalog_entries(simple_job_factory, sys_tst_bldr):
    """Verify an SXID code on the XID log path does not enter XID action decoding."""

    job_info = simple_job_factory(start_timestamp=1700000000, end_timestamp=1700010000, nodes=["test-node-001"])
    test_syslog_data = [
        make_syslog_entry(
            body="NVRM: Xid (PCI:0009:01:00): 10001, unexpected switch code",
            node="test-node-001",
            timestamp_ns=1700005000000000000,
        ),
    ]

    result = sys_tst_bldr(job_info, test_syslog_data)
    recovery_actions, investigatory_actions = SyslogAttributionParams.collect_xidparser_actions(result.event_stream)
    xid_pci_mappings = SyslogAttributionParams.collect_xid_pci_mapping(result.event_stream)

    assert set(recovery_actions.symptoms()) == set()
    assert set(investigatory_actions.symptoms()) == set()
    assert xid_pci_mappings["XID-10001"] == "test-node-001:0009:01:00"


def test_xid_with_no_documented_actions_emits_tbd(simple_job_factory, sys_tst_bldr):
    """Verify an XID with no documented recovery action is exposed as ``TBD``."""

    job_info = simple_job_factory(start_timestamp=1700000000, end_timestamp=1700010000, nodes=["test-node-001"])
    test_syslog_data = [
        make_syslog_entry(
            body="NVRM: Xid (PCI:0009:01:00): 162, power smoothing reengaged",
            node="test-node-001",
            timestamp_ns=1700005000000000000,
        ),
    ]

    result = sys_tst_bldr(job_info, test_syslog_data)
    recovery_actions, investigatory_actions = SyslogAttributionParams.collect_xidparser_actions(result.event_stream)

    assert set(recovery_actions.symptoms()) == {"TBD"}
    assert set(investigatory_actions.symptoms()) == {"TBD"}


def test_tbd_public_action_takes_precedence_over_ignore(simple_job_factory, sys_tst_bldr):
    """Verify ``TBD`` remains visible when combined with ``IGNORE``."""

    job_info = simple_job_factory(start_timestamp=1700000000, end_timestamp=1700010000, nodes=["test-node-001"])
    test_syslog_data = [
        make_syslog_entry(
            body="NVRM: Xid (PCI:0009:01:00): 162, power smoothing reengaged",
            node="test-node-001",
            timestamp_ns=1700005000000000000,
        ),
        make_syslog_entry(
            body="NVRM: Xid (PCI:0009:02:00): 160, power brake engaged",
            node="test-node-001",
            timestamp_ns=1700005001000000000,
        ),
    ]

    result = sys_tst_bldr(job_info, test_syslog_data)
    recovery_actions, investigatory_actions = SyslogAttributionParams.collect_xidparser_actions(result.event_stream)

    assert set(recovery_actions.symptoms()) == {"TBD"}
    assert set(investigatory_actions.symptoms()) == {"INVESTIGATE_SW", "TBD"}


def test_xid_171_detection(simple_job_factory, sys_tst_bldr):
    """Test XID-171 (UNCORRECTABLE_DRAM_ERROR) detection."""
    job_info = simple_job_factory(start_timestamp=1700000000, end_timestamp=1700010000, nodes=["test-node-001"])

    test_syslog_data = [
        make_syslog_entry(
            body="Nov 15 10:30:00 test-node-001 kernel: NVRM: Xid (PCI:0000:bf:00): 171, pid=1234, name=test, UNCORRECTABLE_DRAM_ERROR",
            node="test-node-001",
            timestamp_ns=1700005000000000000,
        ),
    ]

    result = sys_tst_bldr(job_info, test_syslog_data)

    nodes = result.get_node_attributions()
    # Verify XID-171 was detected
    assert "test-node-001" in nodes.attributions
    attributions = nodes.attributions["test-node-001"].attributions
    xid_171_found = any("XID-171" in symptom for symptom in attributions.symptoms())
    assert xid_171_found, f"XID-171 not found in attributions: {list(attributions.symptoms())}"

    # Verify PCI mapping
    xid_pci_mappings = SyslogAttributionParams.collect_xid_pci_mapping(result.event_stream)
    assert "XID-171" in xid_pci_mappings
    assert xid_pci_mappings["XID-171"] == "test-node-001:0000:bf:00"


def test_xid_172_detection(simple_job_factory, sys_tst_bldr):
    """Test XID-172 (UNCORRECTABLE_SRAM_ERROR) detection."""
    job_info = simple_job_factory(start_timestamp=1700000000, end_timestamp=1700010000, nodes=["test-node-001"])

    test_syslog_data = [
        make_syslog_entry(
            body="Nov 15 10:30:00 test-node-001 kernel: NVRM: Xid (PCI:0000:e4:00): 172, pid=5678, name=python, UNCORRECTABLE_SRAM_ERROR",
            node="test-node-001",
            timestamp_ns=1700005000000000000,
        ),
    ]

    result = sys_tst_bldr(job_info, test_syslog_data)

    nodes = result.get_node_attributions()
    # Verify XID-172 was detected
    assert "test-node-001" in nodes.attributions
    attributions = nodes.attributions["test-node-001"].attributions
    xid_172_found = any("XID-172" in symptom for symptom in attributions.symptoms())
    assert xid_172_found, f"XID-172 not found in attributions: {list(attributions.symptoms())}"

    # Verify PCI mapping
    xid_pci_mappings = SyslogAttributionParams.collect_xid_pci_mapping(result.event_stream)
    assert "XID-172" in xid_pci_mappings
    assert xid_pci_mappings["XID-172"] == "test-node-001:0000:e4:00"


def rule_lines(lines, attr, symptom, rule, tflops_start, multi_gpu, fail_reason=None):
    if fail_reason is not None:
        return [
            pparam(li, attr, symptom, rule, tflops_start, multi_gpu, marks=xfail(reason=fail_reason)) for li in lines
        ]
    else:
        return [(li, attr, symptom, rule, tflops_start, multi_gpu) for li in lines]


@pytest.fixture
def get_one_line_attr_builder(sys_tst_bldr):
    """Build node attributions from a single syslog line."""

    def _gen_stream(line, node, time, jobinfo, events_to_add=[]):
        test_data = [make_syslog_entry(body=line, node=node, timestamp_ns=int(time * 1e9))]
        result = sys_tst_bldr(jobinfo, test_data, events_to_add=events_to_add, stream_id=line)
        return result.get_node_attributions()

    return _gen_stream


class TestSyslogRules:
    syslog_rules = [
        # moved to syslog_rules.test.toml
        pparam(
            "Abrupt nvidia-imex daemon shutdown detected, robust channel recovery invoked!",
            "bad",
            "",
            "?",
            True,
            True,
            marks=xfail(reason="test not implemented"),
        ),
    ]

    @pytest.mark.parametrize(
        "line,attr,symptom,rule,tflops_start,multi_gpu",
        syslog_rules,
    )
    def test_rule(
        self,
        line,
        attr,
        symptom,
        rule,
        get_one_line_attr_builder,
        tflops_start,
        multi_gpu,
        syslog_jobinfo_builder,
        request,
        obs_data,
    ):
        syslog_jobinfo = syslog_jobinfo_builder(tflops_start, not multi_gpu)
        node = syslog_jobinfo.nodes[0]
        mfu_event = gen_mfu(syslog_jobinfo, tflops_start, obs_data)
        node_attr = get_one_line_attr_builder(
            line, node, syslog_jobinfo.job_start_time.timestamp(), syslog_jobinfo, events_to_add=[mfu_event]
        )
        match attr:
            case "bad":
                match_bad(node_attr, node, symptom, rule)
            case "interesting":
                match_interesting(node_attr, node, symptom, rule)
            case _:
                raise UnknownAttr(attr)

    @pytest.mark.parametrize(
        "line,attr,symptom,rule",
        [
            #       ("", "bad", sub_category, "xid-nvlink-event"),
            #       ("", "bad", row["dynamic_symptom"], "xid-nvlink-event"),
            pparam("", "bad", "FalsePositive-XID-64", "xid-64-pointer-error", marks=xfail(reason="test incomplete")),
            pparam("", "bad", "DriverBug-XID-64", "xid-64-event", marks=xfail(reason="test incomplete")),
            pparam("", "bad", "FalsePositive-XID-63", "xid-63-event", marks=xfail(reason="test incomplete")),
            #       ("", "bad", f"FalsePositive-XID-{xid_pattern}", "unknown-false-positive-xid-event"),
            #       ("", "interesting", sub_category, "xid-event"),
        ],
    )
    def test_xid_parser_rules(self, line, attr, symptom, rule, get_one_line_attr_builder, syslog_jobinfo):
        node = syslog_jobinfo.nodes[0]
        node_attr = get_one_line_attr_builder(line, node, syslog_jobinfo.job_start_time.timestamp(), syslog_jobinfo)
        match attr:
            case "bad":
                match_bad(node_attr, node, symptom, rule)
            case "interesting":
                match_interesting(node_attr, node, symptom, rule)
            case _:
                raise UnknownAttr(attr)
