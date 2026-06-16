# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Applog pre-filter data: include and exclude term lists.

Ported from attribution_lib/rules/applog_include_list.py and
attribution_lib/rules/applog_exclude_list.py.  Importing this module
registers the pre-filter with the rules-engine framework.
"""

from attribution_lib.observation.inbound import Source as ObservationSource
from attribution_lib.rules_engine import PreFilterSpec, register_pre_filter

# ---------------------------------------------------------------------------
# Exclude terms  (from applog_exclude_list.py)
# ---------------------------------------------------------------------------
exclude_terms = [
    "Setting `pad_token_id`",
    "NCCL INFO",
    "INFO:megatron",
    "WARNING:megatron",
    "NeMo I",
    "using param module",
    "name decoder",
    "json.exception",
    "Launch NCCL",
    "handlers.py:34: UserWarning",
    "OFI Request completed with error",
    "truncated because CLIP",
    "Token indices sequence length is longer",
    "Operation: list objects - Attempt",
    "\\n",
    "\\'",
    "<|eot_id|>",
    "<|start_header_id|>",
    "Preparing custom initialization in the MoE-MLP layer",
    "Checkpoint file not found",
    "[DEBUG] Error processing",
    "libmpg123",
    ".wav:",
    "not a git repository",
    "Whiten Error",  # Whiten Error (florence (microsoft/Florence-2-large)): 0.1743464723205801
    "tensorflow/stream_executor/platform/default/dso_loader.cc",
    "Installing collected packages",  # "28: Installing collected packages: antlr4-python3-runtime, smmap, setproctitle, sentry-sdk, omegaconf, docker-pycreds, hydra-core, gitdb, gitpython, wandb, one-logger\n",
    "Error parsing requirements for",  # 30: WARNING: Error parsing requirements for pyarrow: [Errno 2] No such file or directory: '/usr/local/lib/python3.10/dist-packages/pyarrow-14.0.1.dist-info/METADATA'
    "retry: 0 /",
    "NCCL WARN NCCL Inspector",
    "CUDA kernel errors might be asynchronously reported",
    "[DEBUG",  # Example: [DEBUG | CommunicationMesh]: self.peer_server_addresses={...}
    "wandb: WARNING",  # wandb: WARNING (User provided step: 71150 is less than current step: 81337. Dropping entry: {'iter': 71150, 'step': 71150, 'token_count': 1748705280, '_timestamp': 1729378973.9152606}).
    "WARNING: shuffle index length",  # 116: WARNING: shuffle index length (127428895) is not equal to sample index length (127428896)\n
    "Error comparing answers",  # 0: Error comparing answers: name 'ProcessPool' is not defined\n"
    "from botocore.exceptions",  # used by inference jobs
    "Could not install packages",  # 201: ERROR: Could not install packages due to an OSError: [Errno 2] No such file or directory: '/usr/local/lib/python3.10/dist-packages/nvidia/nvcomp/nvcomp_impl.cpython-311-x86_64-linux-gnu.so'
    "No such file or directory: '/tmp",  # this is error from dirty exit
    "you are running uWSGI as root",
    "This is a development server",
    "dist-packages/transformers",  # [rank812]:   File "/usr/local/lib/python3.10/dist-packages/torch/distributed/checkpoint/default_planner.py", line 354, in create_default_local_load_plan
    "dist-packages/torch",  # [rank812]:   File "/usr/local/lib/python3.10/dist-packages/torch/distributed/checkpoint/default_planner.py", line 354, in create_default_local_load_plan
    "megatron/core/dist_checkpointing",
    "failed due to rate limit",
    "Stage 0 - VideoDownloader",
    "Stage 1 - FrameExtractionStage",
    "Stage 2 - TransNetV2ClipExtractionStage",
    "Stage 3 - SplitWriterStage",
    "WARNING:dataverse",
    "does not support per-request generators. Falling back to PyTorch-native implementation",  # FlashInfer 0.2.3+ does not support per-request generators. Falling back to PyTorch-native implementation.\n",
    "readResponse: Failed to read sqs proxy response body",
    "py_run_labeler_image.binary.runfiles",
    "dlav.plannet.data_labeling.data_labeler",
    "APIServer",
    "results:          [",
    "One of the special magic",
    "Failed to import DrafterBench",
    "DrafterBench evaluation will not be available",
    "nemo_skills.inference.generate",
]

# ---------------------------------------------------------------------------
# Include terms  (from applog_include_list.py)
# ---------------------------------------------------------------------------
include_terms = [
    "CANCELLED AT",
    "error",
    "Error",
    "ERROR",
    "Errno",
    "errno",
    "WARNING",
    "failed",
    "exception",
    "Exception",
    "peer",
    "Segmentation fault",
    "NODE FAILURE",
    "pushrecv",
    "NCCL version",
    # "NCCL WARN",
    "NCCL WARN Cuda failure",
    "NCCL WARN Message truncated",
    "device is incompatible",
    "insufficient shared memory",
    "Address already in use",
    "exiting program at",
    "end the job",
    "exiting program after",
    "after training is done",
    "validation loss at iteration",  # 255:  validation loss at iteration 1049041 on validation set | lm loss value: 2.028079E+00 | lm loss PPL: 7.599470E+00 |
    "Inference job is done",
    "Disk quota exceeded",
    "Argument list too long",
    "Start calculating task metric",
    "clean up dameons",
    "DUE TO TIME LIMIT",
    "Barrier timed out",
    "No CUDA runtime is found",
    'File "',
    # "Serving Flask app",
    "fatal",
    "Done waiting for other agents",  # I0909 10:56:19.535000 23456244205376 torch/distributed/elastic/agent/server/api.py:902] Done waiting for other agents. Elapsed: 13.589454412460327 seconds
    "Writing temporary files back into original files",  # https://github.com/Kipok/NeMo-Skills/blob/590881a640910fa77a233c510ecf3054afe9b7ef/nemo_skills/code_execution/sandbox.py#L82
    "watchdog",  # ProcessGroupNCCL's watchdog got stuck for 600 seconds without making progress in monitoring enqueued collectives.
    "Main process ended cleanly",  # yotta.launcher.ray_cluster:_launch_primary_node:254 - Main process ended cleanly.
    "wandb",
    "Validating solutions: 100%",
    "test results at the end of training",
    "Stopping cuda profiler",
    "Gracefully skipping multicast initialization",
    "sigkill_handler] Killing subprocess",
    "Avg bus bandwidth",  # completion mark of nccl-tests https://github.com/NVIDIA/nccl-tests/blob/8dfeab9eb9bdfdf13503e71e1f33e7f8a208b540/src/common.cu#L1130
    "raise ",
    "http.client",
    "compiling and loading fused kernels",
]

# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
register_pre_filter(
    ObservationSource.APPLOG,
    PreFilterSpec(include_terms=include_terms, exclude_terms=exclude_terms),
)
