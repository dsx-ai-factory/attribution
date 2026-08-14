# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Comprehensive enum of NVIDIA Attribution symptoms."""

from enum import Enum


class RegisteredSymptoms(Enum):
    # XID Symptoms
    XID_13 = "XID-13-"
    XID_13_GR_EXCEPTION = "XID-13-GR_EXCEPTION"
    XID_13_GR_EXCEPTION_SW_NOTIFY = "XID-13-GR_EXCEPTION /"
    XID_13_GR_EXCEPTION_SW_NOTIFY_GR_ERROR_SW_NOTIFY = "XID-13-GR_EXCEPTION /\nGR_ERROR_SW_NOTIFY"
    XID_31 = "XID-31-"
    XID_31_MMU_ERR_FLT = "XID-31-MMU_ERR_FLT"
    XID_32 = "XID-32-"
    XID_38 = "XID-38-"
    XID_43 = "XID-43-"
    XID_43_RESETCHANNEL_VERIF_ERROR = "XID-43-RESETCHANNEL_VERIF_ERROR"
    XID_44 = "XID-44-"
    XID_44_GR_FAULT_DURING_CTXSW = "XID-44-GR_FAULT_DURING_CTXSW"
    XID_45 = "XID-45-"
    XID_45_PREEMPTIVE_REMOVAL = "XID-45-PREEMPTIVE_REMOVAL"
    XID_48 = "XID-48-"
    XID_48_GPU_ECC_DBE = "XID-48-GPU_ECC_DBE"
    XID_62 = "XID-62-"
    XID_62_PMU_HALT_ERROR = "XID-62-PMU_HALT_ERROR"
    XID_63 = "XID-63-"
    XID_63_INFOROM_DRAM_RETIREMENT_EVENT = "XID-63-INFOROM_DRAM_RETIREMENT_EVENT"
    XID_74 = "XID-74-"
    XID_74_NVLINK_ERROR = "XID-74-NVLINK_ERROR"
    XID_79 = "XID-79-"
    XID_79_GPU_HAS_FALLEN_OFF_THE_BUS = "XID-79-GPU_HAS_FALLEN_OFF_THE_BUS"
    XID_92 = "XID-92-"
    XID_92_EXCESSIVE_SBE_INTERRUPTS = "XID-92-EXCESSIVE_SBE_INTERRUPTS"
    XID_94 = "XID-94-"
    XID_94_CONTAINED_ERROR = "XID-94-CONTAINED_ERROR"
    XID_95 = "XID-95-"
    XID_95_UNCONTAINED_ERROR = "XID-95-UNCONTAINED_ERROR"
    XID_108_NVLINK_REMOTE_TRANSLATION_ERROR = "XID-108-NVLINK_REMOTE_TRANSLATION_ERROR"
    XID_109 = "XID-109-"
    XID_109_CTXSW_TIMEOUT_ERROR = "XID-109-CTXSW_TIMEOUT_ERROR"
    XID_120 = "XID-120-"
    XID_120_GSP_ERROR = "XID-120-GSP_ERROR"
    XID_119_GSP_RPC_TIMEOUT = "XID-119-GSP_RPC_TIMEOUT"
    XID_137 = "XID-137-"
    XID_145 = "XID-145"
    XID_145_INFRAHW = "XID-145-InfraHW"
    XID_145_INFRASW = "XID-145-InfraSW"
    XID_145_MULTICATEGORY = "XID-145-MultiCategory"
    XID_145_UNDETERMINED = "XID-145-Undetermined"
    XID_145_RLW_SRC_TRACK_4 = "XID-145-RLW_SRC_TRACK-4"
    XID_145_RLW_SRC_TRACK_8 = "XID-145-RLW_SRC_TRACK-8"
    XID_145_RLW_REMAP_4 = "XID-145-RLW_REMAP-4"
    XID_145_RLW_REMAP_8 = "XID-145-RLW_REMAP-8"
    XID_149 = "XID-149"
    XID_149_0_NETIR_INT_NO_LINK_DOWN_0 = "XID-149.0-NETIR_INT-NO_LINK_DOWN-0"
    XID_149_37_NETIR_LINK_DOWN_PEER_RESET_EVENT_0 = "XID-149.37-NETIR_LINK_DOWN-PEER_RESET_EVENT-0"
    XID_149_4_NETIR_LINK_DOWN_ALIGNMENT_LOSS_0 = "XID-149.4-NETIR_LINK_DOWN-ALIGNMENT_LOSS-0"
    XID_149_10_NETIR_LINK_DOWN_PHY_LAYER_RETRANSMIT_TIMEOUT_0 = (
        "XID-149.10-NETIR_LINK_DOWN-PHY_LAYER_RETRANSMIT_TIMEOUT-0"
    )
    XID_162_PSHC_REENGAGED = "XID-162-PSHC_REENGAGED"
    XID_163_PSHC_DISENGAGED = "XID-163-PSHC_DISENGAGED"
    XID_171 = "XID-171-"
    XID_171_UNCORRECTABLE_DRAM_ERROR = "XID-171-UNCORRECTABLE_DRAM_ERROR"
    XID_172 = "XID-172-"
    XID_172_UNCORRECTABLE_SRAM_ERROR = "XID-172-UNCORRECTABLE_SRAM_ERROR"
    XID_GENERIC = "XID-"
    FALSE_POSITIVE_XID_64 = "FalsePositive-XID-64"
    FALSE_POSITIVE_XID_63 = "FalsePositive-XID-63"
    XID_64_INFOROM_DRAM_RETIREMENT_FAILURE = "XID-64-INFOROM_DRAM_RETIREMENT_FAILURE"
    DRIVER_BUG_XID_64 = "DriverBug-XID-64"

    # System Symptoms
    RUNNING_WITHOUT_ERROR = "running without error"
    THE_PROCESS_COMPLETES = "the process completes"
    NOT_RESPONDING = "Not responding"
    STOPPING_NVIDIA_IMEX_SERVICE = "Stopping nvidia-imex.service"
    SYSTEM_POWERING_DOWN = "System is powering down"
    SYSTEM_SERVICE_OOM_KILL = "OOM kill of system service"
    OOM_KILL_OF_SYSTEM_POD = "OOM kill of a system pod"
    USER_TASK_OOM_KILL = "OOM kill of user task"
    OOM_KILL_OTHER = "OOM kill (other)"
    STOPPING_CUDA_PROFILER = "Stopping cuda profiler"
    NMI_WATCHDOG_HARD_LOCKUP = "NMI watchdog hard LOCKUP"
    NODE_REBOOT_INITIATED = "Node reboot initiated"
    NODE_UNEXPECTEDLY_REBOOTED = "Node unexpectedly rebooted"
    NODE_FAILURE_REPORTED_BY_SLURM = "Node failure reported by slurm"
    GPU_IS_LOST_REPORTED_BY_SLURM = "GPU is lost reported by slurm"
    NODE_OFFLINE = "node offline"

    # Job Symptoms
    JOB_DIDNT_START = "Job didn't start"
    SRUN_DIDNT_HAPPEN = "srun didn't happen"
    CANCELLED_BY_TIMEOUT = "Cancelled by Timeout"
    CANCELLED_BY_USER = "Cancelled by user"
    CANCELLED_DUE_TO_TASK_FAILURE = "Cancelled due to task failure"
    CANCELLED_BY_PYTORCH_NCCL_WATCHDOG = "Cancelled by PyTorch NCCL watchdog"
    CANCELLED_BY_SITE_AUTOMATION = "Cancelled by site automation"
    CANCELLED_BY_CLUSTER_OPERATORS = "Cancelled by cluster operators"
    JOB_FAILED = "job failed"
    JOB_REQUEUED = "Job requeued"

    # Application Symptoms
    MISSING_KEY_IN_CHECKPOINT = "Missing key in checkpoint"
    AFTER_TRAINING_IS_DONE = "after training is done"
    VALIDATION_LOSS_AT_ITERATION = "validation loss at iteration"
    ALIAS_ELSE = "alias else"
    DISK_QUOTA_EXCEEDED = "Disk quota exceeded"
    ASSERT_NUMBER_IMAGE_TAGS = "assert number_image_tags <= number_of_images"
    WANDB_COMM_ERROR = "wandb.errors.CommError: Run initialization has timed out"
    UNEXPECTED_ERROR_FROM_CUDAGETDEVICECOUNT = "Unexpected error from cudaGetDeviceCount()"
    CLIENT_SOCKET_HAS_TIMED_OUT = "client socket has timed out"
    SYSTEM_OUT_OF_MEMORY = "System-out of memory"
    NAN_REPORTED_BY_MEGATRON = "NaN reported by Megatron"
    BAD_TOKEN = "Bad Token"
    NCCL_IB_DEVICE_INCOMPATIBILITY = "NCCL IB device incompatibility"
    MEGATRON_CHECKPOINTING_EXCEPTION = "megatron.core.dist_checkpointing.core.CheckpointingException"
    DATA_LOADER_EXCEPTION = "DataLoader Exception"
    RUNTIME_ERROR_GENERIC = "RuntimeError"
    RUNTIME_ERROR_NUMPY_NOT_AVAILABLE = "RuntimeError: Numpy is not available"
    PYXIS_ENROOT_MOUNT_FAILURE_FROM_USER_DIRECTORY = "pyxis: enroot mount failure from user directory"
    PYXIS_ENROOT_MOUNT_FAILURE_FROM_SYSTEM_DIRECTORY = "pyxis: enroot mount failure from system directory"
    PYXIS_ENROOT_MOUNT_PATH_NOT_FOUND = "pyxis: enroot mount path not found"
    PYXIS_CANNOT_FIND_CONTAINER_FILE = "pyxis: cannot find container file"
    PYXIS_COULDNT_START_CONTAINER = "pyxis: couldn't start container"
    PYXIS_NVIDIA_HOOK_FAILED = "pyxis: nvidia hook failed"
    PYXIS_CGROUPS_HOOK_FAILED = "pyxis: cgroup hook failed"
    PYXIS_NVIDIA_CLI_MOUNT_ERROR = "pyxis: nvidia-container-cli mount error"
    PYXIS_NVIDIA_CLI_INIT_ERROR = "pyxis: nvidia-container-cli init error"
    PYXIS_NVIDIA_CLI_DEVICE_ERROR = "pyxis: nvidia-container-cli device error"
    PYXIS_NVIDIA_UNKNOWN_DRIVER_CAPABILITY = "pyxis: nvidia unknown driver capability"
    PYXIS_NVIDIA_CLI_REQUIREMENT_ERROR = "pyxis: nvidia-container-cli requirement error"
    PYXIS_NVIDIA_CLI_LDCACHE_ERROR = "pyxis: nvidia-container-cli ldcache error"
    PYXIS_NVIDIA_CLI_DETECTION_ERROR = "pyxis: nvidia-container-cli detection error"
    PYXIS_NVIDIA_CLI_NOT_INSTALLED = "pyxis: nvidia-container-cli not installed"
    PYXIS_MELLANOX_HOOK_FAILED = "pyxis: mellanox hook failed"
    # Registry / image pull errors.
    ENROOT_REGISTRY_AUTH_DENIED = "enroot: registry auth denied"
    ENROOT_REGISTRY_NOT_FOUND = "enroot: registry image not found"
    ENROOT_INVALID_IMAGE_REFERENCE = "enroot: invalid image reference"
    ENROOT_DIGEST_ERROR = "enroot: digest error"
    ENROOT_CURL_CONNECT_TUNNEL_FAILED = "enroot: curl tunnel failure"
    ENROOT_MISSING_CREDS_DOCKER_HUB_FALLBACK = "enroot: missing credentials (docker hub fallback)"
    ENROOT_CORRUPTED_CACHE_LAYER = "enroot: corrupted cache layer"
    ENROOT_CURL_TIMEOUT = "enroot: curl transfer timeout"
    ENROOT_CURL_EMPTY_REPLY = "enroot: curl empty reply from server"
    ENROOT_DNS_RESOLUTION_FAILED = "enroot: DNS resolution failed"
    ENROOT_JSON_PARSE_FAILED = "enroot: JSON parse failed"
    # Filesystem / storage errors. Path-aware rules emit different symptoms
    # depending on whether the affected path is user-managed (/home/*, /lustre/*)
    # or system-managed (everything else).
    ENROOT_ROOTFS_LOCK_CONTENTION = "enroot: rootfs lock contention"
    ENROOT_NOT_ENOUGH_SPACE_USER = "enroot: not enough space (user path)"
    ENROOT_NOT_ENOUGH_SPACE_SYSTEM = "enroot: not enough space (system path)"
    ENROOT_INVALID_IMAGE_FORMAT = "enroot: invalid image format"
    PYXIS_FILE_ALREADY_EXISTS_EPHEMERAL = "pyxis: file already exists (ephemeral)"
    PYXIS_MKDIR_PERMISSION_DENIED_USER = "pyxis: mkdir permission denied (user path)"
    PYXIS_MKDIR_PERMISSION_DENIED_SYSTEM = "pyxis: mkdir permission denied (system path)"
    UNSQUASHFS_FS_CORRUPTION = "unsquashfs: file system corruption"
    UNSQUASHFS_DIR_SCAN_PERMISSION_USER = "unsquashfs: dir_scan permission denied (user path)"
    UNSQUASHFS_DIR_SCAN_PERMISSION_SYSTEM = "unsquashfs: dir_scan permission denied (system path)"
    UNSQUASHFS_OOM_KILLED = "unsquashfs: killed by OOM"
    MKSQUASHFS_DISK_FULL_USER = "mksquashfs: disk full (user path)"
    MKSQUASHFS_DISK_FULL_SYSTEM = "mksquashfs: disk full (system path)"
    UNSQUASHFS_WRITE_FAILED_USER = "unsquashfs: write failed (user path)"
    UNSQUASHFS_WRITE_FAILED_SYSTEM = "unsquashfs: write failed (system path)"
    # Namespace / kernel / exec errors
    PYXIS_NAMESPACE_CREATE_FAILED = "pyxis: failed to create namespace"
    ENROOT_CAP_SET_FAILED = "enroot: failed to set capabilities"
    ENROOT_SWITCHROOT_EXEC_FAILED = "enroot: switchroot exec failed"
    # Pyxis-direct errors (from pyxis C code)
    PYXIS_CONTAINER_START_NONZERO = "pyxis: container start nonzero exit"
    PYXIS_COULDNT_CHDIR_WORKDIR = "pyxis: couldn't chdir to container workdir"
    # Pyxis arg-validation errors
    PYXIS_CONTAINER_IMAGE_MULTIPLE_TIMES = "pyxis: --container-image specified multiple times"
    PYXIS_MOUNT_SOURCE_INVALID = "pyxis: mount source invalid"
    PYXIS_CONFLICTING_OPTIONS = "pyxis: conflicting options"
    PYXIS_CONTAINER_NAME_EXISTS = "pyxis: container name already exists"
    SRUN_INVALID_PYXIS_ARG = "srun: invalid pyxis argument"
    # Lifecycle wrappers and cancellation markers
    PYXIS_FAILED_TO_CREATE_CONTAINER_FS = "pyxis: failed to create container filesystem"
    PYXIS_CANCELLED_SIGTERM = "pyxis: cancelled by SIGTERM"
    # Shared memory / peer-task cascades
    PYXIS_CASCADE_FROM_PEER_TASK = "pyxis: cascade from peer task"
    NO_CUDA_RUNTIME_FOUND = "No CUDA runtime is found"
    NCCL_INVALID_NIC_SELECTION = "NCCL invalid NIC selection"
    TERMINATE_CALLED = "terminate called"
    FATAL = "fatal"
    NO_SUCH_FILE_OR_DIRECTORY = "No such file or directory"
    FILE_WRITE_FAILED = "file write failed"
    PYTHON_EXCEPTION = "Python Exception"
    VALUE_ERROR_TASKS_NOT_FOUND = "ValueError: Tasks not found"
    EOF_ERROR_RAN_OUT_OF_INPUT = "EOFError: Ran out of input"
    HTTP_CLIENT_REMOTE_DISCONNECTED = "http.client.RemoteDisconnected"
    ERROR_PYXIS = "error: pyxis"
    EXIT_CODE_SIGNAL = "exit code signal"
    SIGKILL_HANDLER_KILLING_SUBPROCESS = "sigkill_handler] Killing subprocess"
    PLACEHOLDER = "placeholder"

    # Hardware Symptoms
    ROW_REMAP = "Row Remap"
    PAGE_RETIREMENT = "Page Retirement"
    ECC_ERROR = "ECC Error"
    IB_DEVICE_SETTINGS_MISMATCH_CX7 = "IB device settings mismatch (CX7)"
    IB_FIRMWARE_OUTDATED = "IB firmware outdated"
    PCIE_DEVICE_MISSING = "PCIe device missing"
    PCIE_BUS_DEGRADED = "PCIe bus degraded"

    GSP_RM_ALLOC_FAILED = "GspRmAlloc failed"
    SILENT_DATA_CORRUPTION = "Silent Data Corruption"
    ERROR_RUNNING_NVIDIA_SMI = "error running nvidia-smi"
    RDMA_FABRIC_FLAP = "RDMA fabric flap"
    IBV_OPEN_DEVICE_FAILED = "ibv_open_device failed"
    EC2_INSTANCE_RETIREMENT = "ec2 instance retirement"
    GPU_UNAVAILABLE = "gpu unavailable"
    GPU_IDLE = "gpu_idle"
    POWER_SUPPLY_UNIT_FAILURE = "Power Supply Unit failure"
    NO_ROUTE_TO_HOST = "No route to host"
    IB_ERROR_12_VENDOR_129 = "IB Error-12 Vendor-129"
    IB_ERROR_12 = "IB Error-12"
    IB_ERROR_12_RETRY_EXC_ERR = "IB Error-12 (IBV_WC_RETRY_EXC_ERR)"
    IB_ERROR_12_VENDOR_129_RETRY_EXC_ERR = "IB Error-12 Vendor-129 (IBV_WC_RETRY_EXC_ERR)"
    IB_ERROR_5_WR_FLUSH_ERR = "IB Error-5 (IBV_WC_WR_FLUSH_ERR)"
    IB_ERROR_GENERIC = "IB Error-"
    EFA_ERROR_GENERIC = "EFA Error-"
    EFA_ERROR_CONNECTION_RESET = "EFA Error-connection-reset"
    EFA_ERROR_107 = "EFA Error-107"

    # Network Symptoms
    ROUTE_UNHEALTHY = "Route Unhealthy"
    NVLINK_INACTIVE = "NVLink inactive"
    NVLINK_ERROR_COUNTERS = "NVLink error counters"
    LINK_DOWN_EAST_WEST = "link down(east-west)"
    NODE_DOWN_UFM = "node down reported by ufm"
    LINK_DOWN_NVLINK = "link down(nvlink)"
    LINK_DOWN_NVLINK_PHYSICAL_LAYER = "link down(nvlink physical layer)"
    FM_LOST_CONNECTION_TO_GPU = "Fabric manager lost connection to GPU"
    FM_LOST_CONNECTION_TO_ALL_GPUS = "Fabric manager lost connection to all GPUs"
    NETWORK_ADDRESS_CANNOT_BE_RETRIEVED = "network address cannot be retrieved"
    NS_NETWORK_CONNECTION_LOST = "NS network connection lost"
    IMEX_LOST_CONNECTION = "IMEX lost connection"
    PROMETHEUS_LOST_CONNECTION = "prometheus lost connection"
    CALL_TO_IBV_MODIFY_QP_FAILED_TIMEOUT_OVERWRITE = (
        "Call to ibv_modify_qp failed: eno 110 timeout(Overwrite NCCL parameters)"
    )
    CALL_TO_IBV_MODIFY_QP_FAILED_TIMEOUT = "Call to ibv_modify_qp failed: eno 110 timeout"
    CALL_TO_IBV_MODIFY_QP_FAILED_INVALID_ARGUMENT_MISMATCHED = (
        "Call to ibv_modify_qp failed: eno 22 invalid argument(Mismatched GID)"
    )
    CALL_TO_IBV_MODIFY_QP_FAILED_INVALID_ARGUMENT = "Call to ibv_modify_qp failed: eno 22 invalid argument"
    CALL_TO_IBV_MODIFY_QP_FAILED_RESOURCE_UNAVAILABLE = "Call to ibv_modify_qp failed: eno 11 resource unavailable"
    CALL_TO_IBV_MODIFY_QP_FAILED = "Call to ibv_modify_qp failed"
    CALL_TO_IBV_SET_ECE_FAILED = "Call to ibv_set_ece failed"
    EAP_AUTHENTICATION_FAILED = "EAP authentication failed"
    EAP_AUTHENTICATION_RECOVERY = "EAP authentication recovery"
    CONNECTION_CLOSED = "Connection closed"
    CONNECTION_RESET_BY_PEER = "Connection reset by peer"
    SOFTWARE_CAUSED_CONNECTION_ABORT = "Software caused connection abort"
    MELLANOX_LINKS_ARE_DOWN = "Mellanox links are down"
    GLOO_CONNECT_FULL_MESH_FAILED = "Gloo connectFullMesh failed"

    # CUDA Error Symptoms
    CUDA_ERROR_ECC_UNCORRECTABLE = "cudaErrorECCUncorrectable"
    CUDA_ERROR_UNKNOWN = "cudaErrorUnknown"
    CUDA_ERROR_NVLINK_UNCORRECTABLE = "cudaErrorNvlinkUncorrectable"
    CUDA_ERROR_LAUNCH_FAILURE = "cudaErrorLaunchFailure"
    CUDA_ERROR_ILLEGAL_ADDRESS = "cudaErrorIllegalAddress"
    CUDA_ERROR_INVALID_RESOURCE_HANDLE = "cudaErrorInvalidResourceHandle"
    CUDA_ERROR_INVALID_VALUE = "cudaErrorInvalidValue"
    CUDA_ERROR_OUT_OF_MEMORY = "cudaError out of memory"
    CUDA_ERROR_DEVICES_UNAVAILABLE = "cudaErrorDevicesUnavailable"
    CUDA_ERROR_NOT_PERMITTED = "cudaErrorNotPermitted"
    CUDA_ERROR_GENERIC = "cudaError"
    CUDA_ERROR_CONTAINED = "cudaErrorContained"
    LOCAL_CATASTROPHIC_ERROR = "local catastrophic error"

    # NCCL Symptoms
    NCCL_INTERNAL_ERROR = "ncclInternalError"
    UNCONVERTIBLE_NCCL_TYPE = "Unconvertible NCCL type"
    NCCL_WARN_CUDA_FAILURE = "NCCL WARN Cuda failure"
    NCCL_UNHANDLED_CUDA_ERROR = "ncclUnhandledCudaError"
    NCCL_SYSTEM_ERROR = "ncclSystemError"
    NCCL_INVALID_ARGUMENT = "ncclInvalidArgument"
    NCCL_INVALID_USAGE = "ncclInvalidUsage"
    NCCL_REMOTE_ERROR = "ncclRemoteError"
    NCCL_TIMEOUT = "ncclTimeout"
    PYTORCH_NCCL_WATCHDOG = "PyTorch NCCL watchdog"
    NCCL_LATE_ARRIVAL = "NCCL Late Arrival"

    # Driver Symptoms
    NV_ERR_INSUFFICIENT_RESOURCES = "NV_ERR_INSUFFICIENT_RESOURCES"
    NV_ERR_INVALID_STATE = "NV_ERR_INVALID_STATE"
    NV_ERR_INVALID_STATE_NV_GPU_OPS = "NV_ERR_INVALID_STATE nv_gpu_ops.c"
    KERNEL_INVALID_MEMORY_ACCESS = "kernel invalid memory access"
    KERNEL_OOPS = "kernel oops"
    FAILED_TO_COLLECT_NVLINK_STATUS_INFO = "Failed to collect nvlink status info"
    NV_ERR_FABRIC_MANAGER_NOT_PRESENT = "NV_ERR_FABRIC_MANAGER_NOT_PRESENT"
    NV_ERR_GENERIC = "NV_ERR_"
    DRIVER_BUG = "DriverBug-"
    IMEX_DAEMON_SHUTDOWN = "imex-daemon-shutdown"
    COLLECT_NVLINK_STATUS_FAILURE = "collect-nvlink-status-failure"
    UPDATE_RX_DETECT_LINK_MASK_FAILURE = "update-rx-detect-link-mask-failure"
    KNVLINK_UPDATE_POST_RX_DETECT_LINK_MASK_IMPL = (
        "knvlinkUpdatePostRxDetectLinkMask_IMPL: Failed to update Rx Detect Link mask!"
    )

    # Kernel Symptoms
    BAD_KERNEL_STACK_FRAME_POINTER_NSYS = "bad kernel stack frame pointer in nsys"
    BAD_KERNEL_STACK_FRAME_POINTER_CUDAGPUINFODUMP = "bad kernel stack frame pointer in CudaGpuInfoDump"
    NUMA_NODE_MEMORY_EXHAUSTION = "NUMA node memory exhaustion"
    BAD_KERNEL_STACK_FRAME_POINTER = "bad kernel stack frame pointer"
    BAD_KERNEL_STACK_FRAME_POINTER_PYTHON = "bad kernel stack frame pointer in python"
    GENERAL_PROTECTION_FAULT = "general protection fault"
    CE_DEADLOCK_DETECTED = "CE deadlock DETECTED"
    USER_BUFFER_TIMEOUT = "UserBuffer timeout"
    SEGMENTATION_FAULT = "Segmentation fault"
    BUS_ERROR = "bus error"

    # Storage Symptoms
    LUSTRE_ERROR = "LustreError"
    LUSTRE_NETWORK_CONNECTIVITY_ERROR = "Lustre network connectivity error"
    ISCSI_DETECTED_CONN_ERROR = "iSCSI detected conn error"
    STALE_FILE_HANDLE = "Stale file handle"
    LUSTRE_CLIENT_EVICTED = "Lustre Client Evicted"
    S3_EXCEPTION = "S3 Exception"
    DATASET_CACHE_WRITE_FAILURE = "Dataset cache write failure"

    # Log Symptoms
    LOG_UNRECOGNIZED = "log unrecognized"
    LOG_UNRECOGNIZED_FROM_INTERACTIVE_JOB = "log unrecognized from interactive job"
    LOG_FILE_NOT_FOUND = "log file not found"
    LOG_PERMISSION_ERROR = "log permission error"
    LOG_FILE_TOO_LARGE = "log file too large"

    # Performance Symptoms
    PERFORMANCE_PHY_EFFECTIVE_ERRORS_HIGH = "performance_phy_effective_errors_high"
    EFFECTIVE_BER_HIGH = "effective_ber_high"
    PLR_TX_BW_LOSS_HIGH_1M_5PCT = "plr_tx_bw_loss_high_1m_5pct"
    PLR_TX_BW_LOSS_HIGH_5M_1PCT = "plr_tx_bw_loss_high_5m_1pct"
    PLR_TX_BW_LOSS_HIGH_10M_0P5PCT = "plr_tx_bw_loss_high_10m_0p5pct"
    PLR_RX_BW_LOSS_HIGH_1M_5PCT = "plr_rx_bw_loss_high_1m_5pct"
    PLR_RX_BW_LOSS_HIGH_5M_1PCT = "plr_rx_bw_loss_high_5m_1pct"
    PLR_RX_BW_LOSS_HIGH_10M_0P5PCT = "plr_rx_bw_loss_high_10m_0p5pct"
    HIST15_HIGH = "hist15_high"
    PERFORMANCE_PORT_XMIT_WAIT_BW_LOSS_HOST_1M_5PCT = "performance_port_xmit_wait_bw_loss_host_1m_5pct"
    PERFORMANCE_PORT_XMIT_WAIT_BW_LOSS_HOST_1M_1PCT = "performance_port_xmit_wait_bw_loss_host_1m_1pct"
    PERFORMANCE_PORT_XMIT_WAIT_BW_LOSS_HOST_1M_0P5PCT = "performance_port_xmit_wait_bw_loss_host_1m_0p5pct"
    PERFORMANCE_PORT_XMIT_WAIT_BW_LOSS_HOST_1M_10PCT = "performance_port_xmit_wait_bw_loss_host_1m_10pct"
    PERFORMANCE_PORT_XMIT_WAIT_BW_LOSS_HOST_1M_50PCT = "performance_port_xmit_wait_bw_loss_host_1m_50pct"
    PERFORMANCE_PORT_XMIT_WAIT_BW_LOSS_HOST_1M_GT0 = "performance_port_xmit_wait_bw_loss_host_1m_gt0"
    PERFORMANCE_PORT_XMIT_WAIT_BW_LOSS_HOST_5M_5PCT = "performance_port_xmit_wait_bw_loss_host_5m_5pct"
    PERFORMANCE_PORT_XMIT_WAIT_BW_LOSS_HOST_5M_1PCT = "performance_port_xmit_wait_bw_loss_host_5m_1pct"
    PERFORMANCE_PORT_XMIT_WAIT_BW_LOSS_HOST_5M_0P5PCT = "performance_port_xmit_wait_bw_loss_host_5m_0p5pct"
    PERFORMANCE_PORT_XMIT_WAIT_BW_LOSS_HOST_5M_10PCT = "performance_port_xmit_wait_bw_loss_host_5m_10pct"
    PERFORMANCE_PORT_XMIT_WAIT_BW_LOSS_HOST_5M_50PCT = "performance_port_xmit_wait_bw_loss_host_5m_50pct"
    PERFORMANCE_PORT_XMIT_WAIT_BW_LOSS_HOST_5M_GT0 = "performance_port_xmit_wait_bw_loss_host_5m_gt0"

    # Severity Symptoms
    NON_FATAL = "non-fatal"

    # Miscellaneous Symptoms
    JSON_DECODE_ERROR = "JSONDecodeError"
    CMD_AEMON_NOT_READY_FOR_SCHEDULING = "CMDaemon: not ready for scheduling"
    REQUIRED_SERVICE_NOT_RUNNING = "Required service not running"
    FAILED_TO_IMPORT_DOCKER_IMAGE = "failed to import docker image"
    ADDRESS_ALREADY_IN_USE = "Address already in use"
    IO_ERROR = "IO error"
    PMIX_COLLECTIVE_TIMEOUT = "pmix collective timeout"
    PYTHON_BUG = "Python bug"
    BROKEN_PIPE = "BrokenPipeError"
    NFS_NOT_RESPONDING = "nfs not responding"
    NFS_SERVER_FAILURE = "nfs server failure"


# Informational or catalog-designated ignored XIDs that should never contribute
# to failure attribution.
INFORMATIONAL_XIDS = frozenset(
    [
        RegisteredSymptoms.XID_63_INFOROM_DRAM_RETIREMENT_EVENT.value,
        RegisteredSymptoms.XID_108_NVLINK_REMOTE_TRANSLATION_ERROR.value,
        RegisteredSymptoms.XID_162_PSHC_REENGAGED.value,
        RegisteredSymptoms.XID_163_PSHC_DISENGAGED.value,
    ]
)


def resolve_registered_symptom(symptom_id: str) -> RegisteredSymptoms:
    """Resolve a raw symptom id string to a RegisteredSymptoms enum.

    Resolution: exact enum match on the value.
    Any id that cannot be resolved is treated as unregistered.
    """
    clean = symptom_id.strip()

    try:
        return RegisteredSymptoms(clean)
    except ValueError:
        # Generic prefix matching for symptoms with dynamic suffixes.
        # XID sub-symptoms (e.g. XID-149.21-NETIR_LINK_DOWN-RECEIEVED_TS1-0)
        # are dynamically generated from syslog decoding and cannot all be
        # enumerated; unregistered variants fall back to XID_GENERIC here.
        prefix_symptoms = (
            RegisteredSymptoms.XID_GENERIC,
            RegisteredSymptoms.BAD_KERNEL_STACK_FRAME_POINTER,
            RegisteredSymptoms.NV_ERR_GENERIC,
            RegisteredSymptoms.RUNTIME_ERROR_GENERIC,
            RegisteredSymptoms.CUDA_ERROR_GENERIC,
            RegisteredSymptoms.IB_ERROR_GENERIC,
            RegisteredSymptoms.EFA_ERROR_GENERIC,
        )

        for prefix in prefix_symptoms:
            if clean.startswith(prefix.value):
                return prefix

        raise ValueError(f"{symptom_id!r} is not a known RegisteredSymptoms id")
