# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import re
from dataclasses import dataclass
from typing import Optional

from attribution_lib.attr_sub_class import AttributionSubClassEnum


# --------------------------------------------------------------------------- #
#  Error code structure                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class ErrorCode:
    error_class: AttributionSubClassEnum
    suggestion: str


# --------------------------------------------------------------------------- #
#  Non-program error codes                                                   #
# --------------------------------------------------------------------------- #
_CUDA_NON_PROGRAM_ERRORS = [3, 36, 100, 103, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812]

# --------------------------------------------------------------------------- #
#  Complete CUDA error mappings                                             #
# --------------------------------------------------------------------------- #
CUDA_ERROR_MAP = {
    # 1
    "cudaErrorInvalidValue": ErrorCode(
        AttributionSubClassEnum.CUDA_INVALID_VALUE,
        "One or more parameters passed to the API call is not within an acceptable range of values.",
    ),
    # 2
    "cudaErrorMemoryAllocation": ErrorCode(
        AttributionSubClassEnum.CUDA_MEMORY_ALLOCATION,
        "Unable to allocate enough memory or resources for the requested operation.",
    ),
    # 3
    "cudaErrorInitializationError": ErrorCode(
        AttributionSubClassEnum.CUDA_INITIALIZATION_ERROR, "CUDA driver and runtime could not be initialized."
    ),
    # 4
    "cudaErrorCudartUnloading": ErrorCode(
        AttributionSubClassEnum.CUDA_CUDART_UNLOADING,
        "CUDA Runtime API call cannot be executed during process shutdown after the CUDA driver has been unloaded.",
    ),
    # 5
    "cudaErrorProfilerDisabled": ErrorCode(
        AttributionSubClassEnum.CUDA_PROFILER_DISABLED,
        "Profiler not initialized. This can happen when the application is running with external profiling tools like visual profiler.",
    ),
    # 9
    "cudaErrorInvalidConfiguration": ErrorCode(
        AttributionSubClassEnum.CUDA_INVALID_CONFIGURATION,
        "Kernel launch is requesting resources that the current device cannot satisfy. Requesting more shared memory per block than the device supports will trigger this error, as will requesting too many threads or blocks. See cudaDeviceProp for more device limitations.",
    ),
    # 12
    "cudaErrorInvalidPitchValue": ErrorCode(
        AttributionSubClassEnum.CUDA_INVALID_PITCH_VALUE, "One or more pitch-related parameters are out of range."
    ),
    # 13
    "cudaErrorInvalidSymbol": ErrorCode(
        AttributionSubClassEnum.CUDA_INVALID_SYMBOL, "The symbol name/identifier passed to the API call is invalid."
    ),
    # 16
    "cudaErrorInvalidHostPointer": ErrorCode(
        AttributionSubClassEnum.CUDA_INVALID_HOST_POINTER, "A host pointer passed to the API call is not valid."
    ),
    # 17
    "cudaErrorInvalidDevicePointer": ErrorCode(
        AttributionSubClassEnum.CUDA_INVALID_DEVICE_POINTER, "A device pointer passed to the API call is not valid."
    ),
    # 18
    "cudaErrorInvalidTexture": ErrorCode(
        AttributionSubClassEnum.CUDA_INVALID_TEXTURE, "The texture passed to the API call is invalid."
    ),
    # 19
    "cudaErrorInvalidTextureBinding": ErrorCode(
        AttributionSubClassEnum.CUDA_INVALID_TEXTURE_BINDING, "The texture binding is invalid."
    ),
    # 20
    "cudaErrorInvalidChannelDescriptor": ErrorCode(
        AttributionSubClassEnum.CUDA_INVALID_CHANNEL_DESCRIPTOR,
        "The channel descriptor passed to the API call is invalid.",
    ),
    # 21
    "cudaErrorInvalidMemcpyDirection": ErrorCode(
        AttributionSubClassEnum.CUDA_INVALID_MEMCPY_DIRECTION,
        "The direction of the memcpy passed to the API call is invalid.",
    ),
    # 26
    "cudaErrorInvalidFilterSetting": ErrorCode(
        AttributionSubClassEnum.CUDA_INVALID_FILTER_SETTING,
        "This indicates that a non-float texture was being accessed with linear filtering. This is not supported by CUDA.",
    ),
    # 27
    "cudaErrorInvalidNormSetting": ErrorCode(
        AttributionSubClassEnum.CUDA_INVALID_NORM_SETTING,
        "This indicates that an attempt was made to read a non-float texture as a normalized float. This is not supported by CUDA.",
    ),
    # 34
    "cudaErrorStubLibrary": ErrorCode(
        AttributionSubClassEnum.CUDA_STUB_LIBRARY,
        "CUDA driver that the application has loaded is a stub library. Applications that run with the stub rather than a real driver loaded will result in CUDA API returning this error.",
    ),
    # 36
    "cudaErrorCallRequiresNewerDriver": ErrorCode(
        AttributionSubClassEnum.CUDA_CALL_REQUIRES_NEWER_DRIVER,
        "The API call requires a newer CUDA driver than the one currently installed.",
    ),
    # 37
    "cudaErrorInvalidSurface": ErrorCode(
        AttributionSubClassEnum.CUDA_INVALID_SURFACE, "The surface passed to the API call is not a valid surface."
    ),
    # 43
    "cudaErrorDuplicateVariableName": ErrorCode(
        AttributionSubClassEnum.CUDA_DUPLICATE_VARIABLE_NAME,
        "Multiple global or constant variables (across separate CUDA source files) share the same string name.",
    ),
    # 44
    "cudaErrorDuplicateTextureName": ErrorCode(
        AttributionSubClassEnum.CUDA_DUPLICATE_TEXTURE_NAME,
        "Multiple textures (across separate CUDA source files) share the same string name.",
    ),
    # 45
    "cudaErrorDuplicateSurfaceName": ErrorCode(
        AttributionSubClassEnum.CUDA_DUPLICATE_SURFACE_NAME,
        "Multiple surfaces (across separate CUDA source files) share the same string name.",
    ),
    # 46
    "cudaErrorDevicesUnavailable": ErrorCode(
        AttributionSubClassEnum.CUDA_DEVICES_UNAVAILABLE,
        "All CUDA devices are busy or unavailable at the current time.",
    ),
    # 49
    "cudaErrorIncompatibleDriverContext": ErrorCode(
        AttributionSubClassEnum.CUDA_INCOMPATIBLE_DRIVER_CONTEXT,
        "The current context is not compatible with the CUDA Runtime. May occur if using CUDA Runtime/Driver interoperability with a driver context.",
    ),
    # 52
    "cudaErrorMissingConfiguration": ErrorCode(
        AttributionSubClassEnum.CUDA_MISSING_CONFIGURATION,
        "The device function being invoked (via cudaLaunchKernel()) was not configured with cudaConfigureCall().",
    ),
    # 53
    "cudaErrorPriorLaunchFailure": ErrorCode(
        AttributionSubClassEnum.CUDA_PRIOR_LAUNCH_FAILURE,
        "Deprecated. Indicates that a previous kernel launch failed. Used for device emulation of kernel launches in CUDA 3.1 or earlier.",
    ),
    # 65
    "cudaErrorLaunchMaxDepthExceeded": ErrorCode(
        AttributionSubClassEnum.CUDA_LAUNCH_MAX_DEPTH_EXCEEDED,
        "Device runtime grid launch exceeded the maximum supported nested grid depth.",
    ),
    # 66
    "cudaErrorLaunchFileScopedTex": ErrorCode(
        AttributionSubClassEnum.CUDA_LAUNCH_FILE_SCOPED_TEX,
        "Grid launch did not occur due to unsupported file-scoped textures with the device runtime.",
    ),
    # 67
    "cudaErrorLaunchFileScopedSurf": ErrorCode(
        AttributionSubClassEnum.CUDA_LAUNCH_FILE_SCOPED_SURF,
        "Grid launch did not occur due to unsupported file-scoped surfaces with the device runtime.",
    ),
    # 68
    "cudaErrorSyncDepthExceeded": ErrorCode(
        AttributionSubClassEnum.CUDA_SYNC_DEPTH_EXCEEDED,
        "cudaDeviceSynchronize call from device runtime failed due to grid depth exceeding cudaLimitDevRuntimeSyncDepth.",
    ),
    # 69
    "cudaErrorLaunchPendingCountExceeded": ErrorCode(
        AttributionSubClassEnum.CUDA_LAUNCH_PENDING_COUNT_EXCEEDED,
        "Device runtime grid launch failed due to exceeding cudaLimitDevRuntimePendingLaunchCount.",
    ),
    # 98
    "cudaErrorInvalidDeviceFunction": ErrorCode(
        AttributionSubClassEnum.CUDA_INVALID_DEVICE_FUNCTION,
        "The requested device function does not exist or is not compiled for the proper device architecture.",
    ),
    # 100
    "cudaErrorNoDevice": ErrorCode(AttributionSubClassEnum.CUDA_NO_DEVICE, "No CUDA-capable devices were detected."),
    # 101
    "cudaErrorInvalidDevice": ErrorCode(
        AttributionSubClassEnum.CUDA_INVALID_DEVICE,
        "The device ordinal supplied by the user does not correspond to a valid CUDA device.",
    ),
    # 102
    "cudaErrorDeviceNotLicensed": ErrorCode(
        AttributionSubClassEnum.CUDA_DEVICE_NOT_LICENSED, "The device doesn't have a valid Grid License."
    ),
    # 103
    "cudaErrorSoftwareValidityNotEstablished": ErrorCode(
        AttributionSubClassEnum.CUDA_SOFTWARE_VALIDITY_NOT_ESTABLISHED,
        "Indicates a failure in CUDA runtime or driver self-tests, failing to establish the validity of either.",
    ),
    # 127
    "cudaErrorStartupFailure": ErrorCode(
        AttributionSubClassEnum.CUDA_STARTUP_FAILURE, "An internal startup failure in the CUDA runtime."
    ),
    # 200
    "cudaErrorInvalidKernelImage": ErrorCode(
        AttributionSubClassEnum.CUDA_INVALID_KERNEL_IMAGE, "The device kernel image is invalid."
    ),
    # 201
    "cudaErrorDeviceUninitialized": ErrorCode(
        AttributionSubClassEnum.CUDA_DEVICE_UNINITIALIZED,
        "No context is bound to the current thread, or an invalid context handle was passed to an API call.",
    ),
    # 205
    "cudaErrorMapBufferObjectFailed": ErrorCode(
        AttributionSubClassEnum.CUDA_MAP_BUFFER_OBJECT_FAILED, "The buffer object could not be mapped."
    ),
    # 206
    "cudaErrorUnmapBufferObjectFailed": ErrorCode(
        AttributionSubClassEnum.CUDA_UNMAP_BUFFER_OBJECT_FAILED, "The buffer object could not be unmapped."
    ),
    # 207
    "cudaErrorArrayIsMapped": ErrorCode(
        AttributionSubClassEnum.CUDA_ARRAY_IS_MAPPED, "The specified array is currently mapped and cannot be destroyed."
    ),
    # 208
    "cudaErrorAlreadyMapped": ErrorCode(AttributionSubClassEnum.CUDA_ALREADY_MAPPED, "The resource is already mapped."),
    # 209
    "cudaErrorNoKernelImageForDevice": ErrorCode(
        AttributionSubClassEnum.CUDA_NO_KERNEL_IMAGE_FOR_DEVICE, "No suitable kernel image is available for the device."
    ),
    # 210
    "cudaErrorAlreadyAcquired": ErrorCode(
        AttributionSubClassEnum.CUDA_ALREADY_ACQUIRED, "A resource has already been acquired."
    ),
    # 211
    "cudaErrorNotMapped": ErrorCode(AttributionSubClassEnum.CUDA_NOT_MAPPED, "A resource is not mapped."),
    # 212
    "cudaErrorNotMappedAsArray": ErrorCode(
        AttributionSubClassEnum.CUDA_NOT_MAPPED_AS_ARRAY, "A mapped resource is unavailable for access as an array."
    ),
    # 213
    "cudaErrorNotMappedAsPointer": ErrorCode(
        AttributionSubClassEnum.CUDA_NOT_MAPPED_AS_POINTER, "A mapped resource is unavailable for access as a pointer."
    ),
    # 214
    "cudaErrorEccUncorrectable": ErrorCode(
        AttributionSubClassEnum.CUDA_ECC_UNCORRECTABLE, "An uncorrectable ECC error was detected during execution."
    ),
    # 215
    "cudaErrorUnsupportedLimit": ErrorCode(
        AttributionSubClassEnum.CUDA_UNSUPPORTED_LIMIT,
        "The cudaLimit passed to the API call is unsupported by the active device.",
    ),
    # 216
    "cudaErrorDeviceAlreadyInUse": ErrorCode(
        AttributionSubClassEnum.CUDA_DEVICE_ALREADY_IN_USE,
        "An exclusive-thread device is already in use by a different thread.",
    ),
    # 217
    "cudaErrorPeerAccessUnsupported": ErrorCode(
        AttributionSubClassEnum.CUDA_PEER_ACCESS_UNSUPPORTED, "P2P access is not supported across the given devices."
    ),
    # 218
    "cudaErrorInvalidPtx": ErrorCode(
        AttributionSubClassEnum.CUDA_INVALID_PTX,
        "A PTX compilation failed; the runtime may fall back to compiling PTX if necessary.",
    ),
    # 219
    "cudaErrorInvalidGraphicsContext": ErrorCode(
        AttributionSubClassEnum.CUDA_INVALID_GRAPHICS_CONTEXT, "An error with the OpenGL or DirectX context."
    ),
    # 220
    "cudaErrorNvlinkUncorrectable": ErrorCode(
        AttributionSubClassEnum.CUDA_NVLINK_UNCORRECTABLE,
        "An uncorrectable NVLink error was detected during execution.",
    ),
    # 221
    "cudaErrorJitCompilerNotFound": ErrorCode(
        AttributionSubClassEnum.CUDA_JIT_COMPILER_NOT_FOUND, "The PTX JIT compiler library was not found."
    ),
    # 222
    "cudaErrorUnsupportedPtxVersion": ErrorCode(
        AttributionSubClassEnum.CUDA_UNSUPPORTED_PTX_VERSION,
        "The provided PTX was compiled with an unsupported toolchain.",
    ),
    # 223
    "cudaErrorJitCompilationDisabled": ErrorCode(
        AttributionSubClassEnum.CUDA_JIT_COMPILATION_DISABLED, "The JIT compilation was disabled."
    ),
    # 224
    "cudaErrorUnsupportedExecAffinity": ErrorCode(
        AttributionSubClassEnum.CUDA_UNSUPPORTED_EXEC_AFFINITY,
        "The provided execution affinity is unsupported by the device.",
    ),
    # 225
    "cudaErrorUnsupportedDevSideSync": ErrorCode(
        AttributionSubClassEnum.CUDA_UNSUPPORTED_DEV_SIDE_SYNC,
        "The PTX JIT contains an unsupported call to cudaDeviceSynchronize.",
    ),
    # 300
    "cudaErrorInvalidSource": ErrorCode(
        AttributionSubClassEnum.CUDA_INVALID_SOURCE, "The device kernel source is invalid."
    ),
    # 301
    "cudaErrorFileNotFound": ErrorCode(
        AttributionSubClassEnum.CUDA_FILE_NOT_FOUND, "The specified file was not found."
    ),
    # 302
    "cudaErrorSharedObjectSymbolNotFound": ErrorCode(
        AttributionSubClassEnum.CUDA_SHARED_OBJECT_SYMBOL_NOT_FOUND, "A link to a shared object failed to resolve."
    ),
    # 303
    "cudaErrorSharedObjectInitFailed": ErrorCode(
        AttributionSubClassEnum.CUDA_SHARED_OBJECT_INIT_FAILED, "Initialization of a shared object failed."
    ),
    # 304
    "cudaErrorOperatingSystem": ErrorCode(AttributionSubClassEnum.CUDA_OPERATING_SYSTEM, "An OS call failed."),
    # 400
    "cudaErrorInvalidResourceHandle": ErrorCode(
        AttributionSubClassEnum.CUDA_INVALID_RESOURCE_HANDLE, "A resource handle passed to the API call was invalid."
    ),
    # 401
    "cudaErrorIllegalState": ErrorCode(
        AttributionSubClassEnum.CUDA_ILLEGAL_STATE,
        "A resource required by the API call is not in a valid state to perform the requested operation.",
    ),
    # 402
    "cudaErrorLossyQuery": ErrorCode(
        AttributionSubClassEnum.CUDA_LOSSY_QUERY,
        "An introspection attempt would discard semantically important information.",
    ),
    # 500
    "cudaErrorSymbolNotFound": ErrorCode(
        AttributionSubClassEnum.CUDA_SYMBOL_NOT_FOUND, "A named symbol was not found."
    ),
    # 600
    "cudaErrorNotReady": ErrorCode(
        AttributionSubClassEnum.CUDA_NOT_READY, "Asynchronous operations previously issued have not completed yet."
    ),
    # 700
    "cudaErrorIllegalAddress": ErrorCode(
        AttributionSubClassEnum.CUDA_ILLEGAL_ADDRESS,
        "The device encountered a load or store instruction on an invalid memory address.",
    ),
    # 701
    "cudaErrorLaunchOutOfResources": ErrorCode(
        AttributionSubClassEnum.CUDA_LAUNCH_OUT_OF_RESOURCES,
        "A launch did not occur because it did not have appropriate resources.",
    ),
    # 702
    "cudaErrorLaunchTimeout": ErrorCode(
        AttributionSubClassEnum.CUDA_LAUNCH_TIMEOUT, "The device kernel took too long to execute."
    ),
    # 703
    "cudaErrorLaunchIncompatibleTexturing": ErrorCode(
        AttributionSubClassEnum.CUDA_LAUNCH_INCOMPATIBLE_TEXTURING,
        "A kernel launch used an incompatible texturing mode.",
    ),
    # 704
    "cudaErrorPeerAccessAlreadyEnabled": ErrorCode(
        AttributionSubClassEnum.CUDA_PEER_ACCESS_ALREADY_ENABLED,
        "Re-enable peer addressing on a context that already had peer addressing enabled.",
    ),
    # 705
    "cudaErrorPeerAccessNotEnabled": ErrorCode(
        AttributionSubClassEnum.CUDA_PEER_ACCESS_NOT_ENABLED,
        "cudaDeviceDisablePeerAccess() is trying to disable peer addressing which has not been enabled yet.",
    ),
    # 708
    "cudaErrorSetOnActiveProcess": ErrorCode(
        AttributionSubClassEnum.CUDA_SET_ON_ACTIVE_PROCESS,
        "cudaSetValidDevices(), cudaSetDeviceFlags(), or similar function called after initializing CUDA runtime with non-device management operations.",
    ),
    # 709
    "cudaErrorContextIsDestroyed": ErrorCode(
        AttributionSubClassEnum.CUDA_CONTEXT_IS_DESTROYED,
        "The context current to the calling thread has been destroyed or is a primary context not yet initialized.",
    ),
    # 710
    "cudaErrorAssert": ErrorCode(
        AttributionSubClassEnum.CUDA_ASSERT,
        "An assert triggered in device code during kernel execution; the device cannot be used again until the process is terminated and relaunched.",
    ),
    # 711
    "cudaErrorTooManyPeers": ErrorCode(
        AttributionSubClassEnum.CUDA_TOO_MANY_PEERS,
        "Hardware resources required to enable peer access have been exhausted.",
    ),
    # 712
    "cudaErrorHostMemoryAlreadyRegistered": ErrorCode(
        AttributionSubClassEnum.CUDA_HOST_MEMORY_ALREADY_REGISTERED,
        "The memory range passed to cudaHostRegister() has already been registered.",
    ),
    # 713
    "cudaErrorHostMemoryNotRegistered": ErrorCode(
        AttributionSubClassEnum.CUDA_HOST_MEMORY_NOT_REGISTERED,
        "The pointer passed to cudaHostUnregister() does not correspond to any currently registered memory region.",
    ),
    # 714
    "cudaErrorHardwareStackError": ErrorCode(
        AttributionSubClassEnum.CUDA_HARDWARE_STACK_ERROR,
        "An error in the call stack was encountered during kernel execution, possibly due to stack corruption or exceeding the stack size limit.",
    ),
    # 715
    "cudaErrorIllegalInstruction": ErrorCode(
        AttributionSubClassEnum.CUDA_ILLEGAL_INSTRUCTION,
        "The device encountered an illegal instruction during kernel execution.",
    ),
    # 716
    "cudaErrorMisalignedAddress": ErrorCode(
        AttributionSubClassEnum.CUDA_MISALIGNED_ADDRESS,
        "A load or store instruction encountered an unaligned memory address.",
    ),
    # 717
    "cudaErrorInvalidAddressSpace": ErrorCode(
        AttributionSubClassEnum.CUDA_INVALID_ADDRESS_SPACE,
        "A kernel instruction attempted to operate on a memory address outside the allowed address spaces (global, shared, or local).",
    ),
    # 718
    "cudaErrorInvalidPc": ErrorCode(
        AttributionSubClassEnum.CUDA_INVALID_PC,
        "The device encountered an invalid program counter during kernel execution.",
    ),
    # 719
    "cudaErrorLaunchFailure": ErrorCode(
        AttributionSubClassEnum.CUDA_LAUNCH_FAILURE,
        "An exception occurred on the device while executing a kernel. This leaves the process in an inconsistent state.",
    ),
    # 720
    "cudaErrorCooperativeLaunchTooLarge": ErrorCode(
        AttributionSubClassEnum.CUDA_COOPERATIVE_LAUNCH_TOO_LARGE,
        "The number of blocks launched per grid for a cooperative kernel exceeds the maximum supported by the device's multiprocessors.",
    ),
    # 800
    "cudaErrorNotPermitted": ErrorCode(
        AttributionSubClassEnum.CUDA_NOT_PERMITTED, "The attempted operation is not permitted."
    ),
    # 801
    "cudaErrorNotSupported": ErrorCode(
        AttributionSubClassEnum.CUDA_NOT_SUPPORTED,
        "The attempted operation is not supported on the current system or device.",
    ),
    # 802
    "cudaErrorSystemNotReady": ErrorCode(
        AttributionSubClassEnum.CUDA_SYSTEM_NOT_READY, "The system is not yet ready to start any CUDA work."
    ),
    # 803
    "cudaErrorSystemDriverMismatch": ErrorCode(
        AttributionSubClassEnum.CUDA_SYSTEM_DRIVER_MISMATCH,
        "There is a mismatch between the display driver and the CUDA driver.",
    ),
    # 804
    "cudaErrorCompatNotSupportedOnDevice": ErrorCode(
        AttributionSubClassEnum.CUDA_COMPAT_NOT_SUPPORTED_ON_DEVICE,
        "The system was upgraded for forward compatibility, but the hardware detected by CUDA does not support this configuration.",
    ),
    # 805
    "cudaErrorMpsConnectionFailed": ErrorCode(
        AttributionSubClassEnum.CUDA_MPS_CONNECTION_FAILED,
        "The MPS client failed to connect to the MPS control daemon or the MPS server.",
    ),
    # 806
    "cudaErrorMpsRpcFailure": ErrorCode(
        AttributionSubClassEnum.CUDA_MPS_RPC_FAILURE,
        "The remote procedural call between the MPS server and the MPS client failed.",
    ),
    # 807
    "cudaErrorMpsServerNotReady": ErrorCode(
        AttributionSubClassEnum.CUDA_MPS_SERVER_NOT_READY,
        "The MPS server is not ready to accept new MPS client requests. This can happen when the MPS server is recovering from a fatal failure.",
    ),
    # 808
    "cudaErrorMpsMaxClientsReached": ErrorCode(
        AttributionSubClassEnum.CUDA_MPS_MAX_CLIENTS_REACHED,
        "Hardware resources needed to create an MPS client have been exhausted.",
    ),
    # 809
    "cudaErrorMpsMaxConnectionsReached": ErrorCode(
        AttributionSubClassEnum.CUDA_MPS_MAX_CONNECTIONS_REACHED,
        "Hardware resources required for device connections have been exhausted.",
    ),
    # 810
    "cudaErrorMpsClientTerminated": ErrorCode(
        AttributionSubClassEnum.CUDA_MPS_CLIENT_TERMINATED,
        "The MPS client has been terminated by the server. To continue using CUDA, the process must be terminated and relaunched.",
    ),
    # 811
    "cudaErrorCdpNotSupported": ErrorCode(
        AttributionSubClassEnum.CUDA_CDP_NOT_SUPPORTED,
        "CUDA Dynamic Parallelism is being used, but the current configuration (e.g., MPS) does not support it.",
    ),
    # 812
    "cudaErrorCdpVersionMismatch": ErrorCode(
        AttributionSubClassEnum.CUDA_CDP_VERSION_MISMATCH,
        "The program contains an unsupported interaction between different versions of CUDA Dynamic Parallelism.",
    ),
    # 900
    "cudaErrorStreamCaptureUnsupported": ErrorCode(
        AttributionSubClassEnum.CUDA_STREAM_CAPTURE_UNSUPPORTED,
        "The operation is not permitted when the stream is capturing.",
    ),
    # 901
    "cudaErrorStreamCaptureInvalidated": ErrorCode(
        AttributionSubClassEnum.CUDA_STREAM_CAPTURE_INVALIDATED,
        "The current capture sequence on the stream has been invalidated due to a previous error.",
    ),
    # 902
    "cudaErrorStreamCaptureMerge": ErrorCode(
        AttributionSubClassEnum.CUDA_STREAM_CAPTURE_MERGE,
        "The operation would have resulted in a merge of two independent capture sequences.",
    ),
    # 903
    "cudaErrorStreamCaptureUnmatched": ErrorCode(
        AttributionSubClassEnum.CUDA_STREAM_CAPTURE_UNMATCHED, "The capture was not initiated in this stream."
    ),
    # 904
    "cudaErrorStreamCaptureUnjoined": ErrorCode(
        AttributionSubClassEnum.CUDA_STREAM_CAPTURE_UNJOINED,
        "The capture sequence contains a fork that was not joined to the primary stream.",
    ),
    # 905
    "cudaErrorStreamCaptureIsolation": ErrorCode(
        AttributionSubClassEnum.CUDA_STREAM_CAPTURE_ISOLATION,
        "A dependency would have been created that crosses the capture sequence boundary, which is disallowed.",
    ),
    # 906
    "cudaErrorStreamCaptureImplicit": ErrorCode(
        AttributionSubClassEnum.CUDA_STREAM_CAPTURE_IMPLICIT,
        "The operation would have resulted in a disallowed implicit dependency on a current capture sequence from cudaStreamLegacy.",
    ),
    # 907
    "cudaErrorCapturedEvent": ErrorCode(
        AttributionSubClassEnum.CUDA_CAPTURED_EVENT,
        "The operation is not permitted on an event last recorded in a capturing stream.",
    ),
    # 908
    "cudaErrorStreamCaptureWrongThread": ErrorCode(
        AttributionSubClassEnum.CUDA_STREAM_CAPTURE_WRONG_THREAD,
        "A capture sequence initiated without cudaStreamCaptureModeRelaxed was ended in a different thread.",
    ),
    # 909
    "cudaErrorTimeout": ErrorCode(AttributionSubClassEnum.CUDA_TIMEOUT, "The wait operation has timed out."),
    # 910
    "cudaErrorGraphExecUpdateFailure": ErrorCode(
        AttributionSubClassEnum.CUDA_GRAPH_EXEC_UPDATE_FAILURE,
        "Graph update was not performed due to constraints specific to instantiated graph update.",
    ),
    # 911
    "cudaErrorExternalDevice": ErrorCode(
        AttributionSubClassEnum.CUDA_EXTERNAL_DEVICE,
        "An async error has occurred in an external device outside of CUDA. This leaves the process in an inconsistent state.",
    ),
    # 912
    "cudaErrorInvalidClusterSize": ErrorCode(
        AttributionSubClassEnum.CUDA_INVALID_CLUSTER_SIZE,
        "A kernel launch error occurred due to cluster misconfiguration.",
    ),
    # 913
    "cudaErrorFunctionNotLoaded": ErrorCode(
        AttributionSubClassEnum.CUDA_FUNCTION_NOT_LOADED,
        "A function handle is not loaded when calling an API that requires a loaded function.",
    ),
    # 914
    "cudaErrorInvalidResourceType": ErrorCode(
        AttributionSubClassEnum.CUDA_INVALID_RESOURCE_TYPE,
        "One or more resources passed in are not valid resource types for the operation.",
    ),
    # 915
    "cudaErrorInvalidResourceConfiguration": ErrorCode(
        AttributionSubClassEnum.CUDA_INVALID_RESOURCE_CONFIGURATION,
        "One or more resources are insufficient or non-applicable for the operation.",
    ),
    # 999
    "cudaErrorUnknown": ErrorCode(AttributionSubClassEnum.CUDA_UNKNOWN, "An unknown internal error has occurred."),
}

# --------------------------------------------------------------------------- #
#  Numeric to string error code mapping                                     #
# --------------------------------------------------------------------------- #
CUDA_NUMERIC_ERROR_MAP = {
    1: "cudaErrorInvalidValue",
    2: "cudaErrorMemoryAllocation",
    3: "cudaErrorInitializationError",
    4: "cudaErrorCudartUnloading",
    5: "cudaErrorProfilerDisabled",
    9: "cudaErrorInvalidConfiguration",
    12: "cudaErrorInvalidPitchValue",
    13: "cudaErrorInvalidSymbol",
    16: "cudaErrorInvalidHostPointer",
    17: "cudaErrorInvalidDevicePointer",
    18: "cudaErrorInvalidTexture",
    19: "cudaErrorInvalidTextureBinding",
    20: "cudaErrorInvalidChannelDescriptor",
    21: "cudaErrorInvalidMemcpyDirection",
    26: "cudaErrorInvalidFilterSetting",
    27: "cudaErrorInvalidNormSetting",
    34: "cudaErrorStubLibrary",
    36: "cudaErrorCallRequiresNewerDriver",
    37: "cudaErrorInvalidSurface",
    43: "cudaErrorDuplicateVariableName",
    44: "cudaErrorDuplicateTextureName",
    45: "cudaErrorDuplicateSurfaceName",
    46: "cudaErrorDevicesUnavailable",
    49: "cudaErrorIncompatibleDriverContext",
    52: "cudaErrorMissingConfiguration",
    53: "cudaErrorPriorLaunchFailure",
    65: "cudaErrorLaunchMaxDepthExceeded",
    66: "cudaErrorLaunchFileScopedTex",
    67: "cudaErrorLaunchFileScopedSurf",
    68: "cudaErrorSyncDepthExceeded",
    69: "cudaErrorLaunchPendingCountExceeded",
    98: "cudaErrorInvalidDeviceFunction",
    100: "cudaErrorNoDevice",
    101: "cudaErrorInvalidDevice",
    102: "cudaErrorDeviceNotLicensed",
    103: "cudaErrorSoftwareValidityNotEstablished",
    127: "cudaErrorStartupFailure",
    200: "cudaErrorInvalidKernelImage",
    201: "cudaErrorDeviceUninitialized",
    205: "cudaErrorMapBufferObjectFailed",
    206: "cudaErrorUnmapBufferObjectFailed",
    207: "cudaErrorArrayIsMapped",
    208: "cudaErrorAlreadyMapped",
    209: "cudaErrorNoKernelImageForDevice",
    210: "cudaErrorAlreadyAcquired",
    211: "cudaErrorNotMapped",
    212: "cudaErrorNotMappedAsArray",
    213: "cudaErrorNotMappedAsPointer",
    214: "cudaErrorEccUncorrectable",
    215: "cudaErrorUnsupportedLimit",
    216: "cudaErrorDeviceAlreadyInUse",
    217: "cudaErrorPeerAccessUnsupported",
    218: "cudaErrorInvalidPtx",
    219: "cudaErrorInvalidGraphicsContext",
    220: "cudaErrorNvlinkUncorrectable",
    221: "cudaErrorJitCompilerNotFound",
    222: "cudaErrorUnsupportedPtxVersion",
    223: "cudaErrorJitCompilationDisabled",
    224: "cudaErrorUnsupportedExecAffinity",
    225: "cudaErrorUnsupportedDevSideSync",
    300: "cudaErrorInvalidSource",
    301: "cudaErrorFileNotFound",
    302: "cudaErrorSharedObjectSymbolNotFound",
    303: "cudaErrorSharedObjectInitFailed",
    304: "cudaErrorOperatingSystem",
    400: "cudaErrorInvalidResourceHandle",
    401: "cudaErrorIllegalState",
    402: "cudaErrorLossyQuery",
    500: "cudaErrorSymbolNotFound",
    600: "cudaErrorNotReady",
    700: "cudaErrorIllegalAddress",
    701: "cudaErrorLaunchOutOfResources",
    702: "cudaErrorLaunchTimeout",
    703: "cudaErrorLaunchIncompatibleTexturing",
    704: "cudaErrorPeerAccessAlreadyEnabled",
    705: "cudaErrorPeerAccessNotEnabled",
    708: "cudaErrorSetOnActiveProcess",
    709: "cudaErrorContextIsDestroyed",
    710: "cudaErrorAssert",
    711: "cudaErrorTooManyPeers",
    712: "cudaErrorHostMemoryAlreadyRegistered",
    713: "cudaErrorHostMemoryNotRegistered",
    714: "cudaErrorHardwareStackError",
    715: "cudaErrorIllegalInstruction",
    716: "cudaErrorMisalignedAddress",
    717: "cudaErrorInvalidAddressSpace",
    718: "cudaErrorInvalidPc",
    719: "cudaErrorLaunchFailure",
    720: "cudaErrorCooperativeLaunchTooLarge",
    800: "cudaErrorNotPermitted",
    801: "cudaErrorNotSupported",
    802: "cudaErrorSystemNotReady",
    803: "cudaErrorSystemDriverMismatch",
    804: "cudaErrorCompatNotSupportedOnDevice",
    805: "cudaErrorMpsConnectionFailed",
    806: "cudaErrorMpsRpcFailure",
    807: "cudaErrorMpsServerNotReady",
    808: "cudaErrorMpsMaxClientsReached",
    809: "cudaErrorMpsMaxConnectionsReached",
    810: "cudaErrorMpsClientTerminated",
    811: "cudaErrorCdpNotSupported",
    812: "cudaErrorCdpVersionMismatch",
    900: "cudaErrorStreamCaptureUnsupported",
    901: "cudaErrorStreamCaptureInvalidated",
    902: "cudaErrorStreamCaptureMerge",
    903: "cudaErrorStreamCaptureUnmatched",
    904: "cudaErrorStreamCaptureUnjoined",
    905: "cudaErrorStreamCaptureIsolation",
    906: "cudaErrorStreamCaptureImplicit",
    907: "cudaErrorCapturedEvent",
    908: "cudaErrorStreamCaptureWrongThread",
    909: "cudaErrorTimeout",
    910: "cudaErrorGraphExecUpdateFailure",
    911: "cudaErrorExternalDevice",
    912: "cudaErrorInvalidClusterSize",
    913: "cudaErrorFunctionNotLoaded",
    914: "cudaErrorInvalidResourceType",
    915: "cudaErrorInvalidResourceConfiguration",
    999: "cudaErrorUnknown",
}

# Build set of non-program error classes
CUDA_ERROR_NOT_PROGRAM_ERROR_CLASS = set()
for err_num in _CUDA_NON_PROGRAM_ERRORS:
    if err_num in CUDA_NUMERIC_ERROR_MAP:
        error_name = CUDA_NUMERIC_ERROR_MAP[err_num]
        if error_name in CUDA_ERROR_MAP:
            CUDA_ERROR_NOT_PROGRAM_ERROR_CLASS.add(CUDA_ERROR_MAP[error_name].error_class)

# --------------------------------------------------------------------------- #
#  Regex patterns                                                           #
# --------------------------------------------------------------------------- #
CudaErrorPattern = re.compile(
    r"(?:(\d+)\s*\(?([Cc]uda.*?[Ee]rror\w+)\)|\b([Cc]uda\w+)\b)|(?:[Cc]uda.*?[Ee]rror[\s\(\) ]*(\d+):?)"
)

# Specific CUDA error patterns
ErrCudaSetMaxSharedMemPattern = re.compile(r"this gpu does not have enough shared[- ]?memory", re.IGNORECASE)
ErrCudaInsufficientDriverPattern = re.compile(
    r"no nvidia driver|nvidia driver[a-z ]*too old|cuda.*error[ ]*\(35\)|(cuda.*error)?insufficient[ ]?driver",
    re.IGNORECASE,
)
ErrCudaInitializationErrorPattern = re.compile(
    r"cuda(error)?[ ]?driver[ ]?initialization[ ]?(failed)?|cuda.*error[ ]*\(3\)", re.IGNORECASE
)
ErrCudaErrorPattern = re.compile(r"cuda.*error|unexpected error from cudagetdevicecount", re.IGNORECASE)

ErrCudaSetMaxSharedMem = AttributionSubClassEnum.CUDA_SET_MAX_SHARED_MEM
ErrCudaSystemNotReady = AttributionSubClassEnum.CUDA_SYSTEM_NOT_READY
ErrCudaInsufficientDriver = AttributionSubClassEnum.CUDA_INSUFFICIENT_DRIVER
ErrCudaInitializationError = AttributionSubClassEnum.CUDA_INITIALIZATION_ERROR
ErrCudaError = AttributionSubClassEnum.CUDA_ERROR

# Suggestions
ErrCudaSetMaxSharedMemSuggestion = "This GPU does not have enough shared-memory."
ErrCudaInsufficientDriverSuggestion = "cudaErrorInsufficientDriver(35): NVIDIA driver is either missing or too old."
ErrCudaInitializationErrorSuggestion = "cudaErrorInitializationError(3): CUDA driver initialization failed."
ErrCudaErrorSuggestion = "The error is from CUDA library. Please check the returned error message for the error reason."


# --------------------------------------------------------------------------- #
#  Helper functions                                                          #
# --------------------------------------------------------------------------- #
def detect_cuda_error(line: str) -> Optional[ErrorCode]:
    """Parse the input text to find a single CUDA error pattern and return it in a structured format."""

    # Find the first match for the pattern
    match = CudaErrorPattern.search(line)
    if not match:
        return None

    matches = CudaErrorPattern.findall(line)
    for match_groups in matches:
        for submatch in match_groups:
            if not submatch:
                continue

            # Check if a numeric error code is matched
            try:
                numeric_value = int(submatch)
                # Look up the error code in CUDA_NUMERIC_ERROR_MAP
                if numeric_value in CUDA_NUMERIC_ERROR_MAP:
                    error_name = CUDA_NUMERIC_ERROR_MAP[numeric_value]
                    if error_name in CUDA_ERROR_MAP:
                        return CUDA_ERROR_MAP[error_name]
            except ValueError:
                # Check string error code
                if submatch in CUDA_ERROR_MAP:
                    return CUDA_ERROR_MAP[submatch]

    return None


def is_cuda_error(line: str, line_lower: str) -> tuple[AttributionSubClassEnum, str, bool]:
    """
    Check if a line contains a CUDA error and classify it.

    Returns:
        Tuple of (error_class, suggestion, is_program_error)
    """

    if ErrCudaSetMaxSharedMemPattern.search(line_lower):
        return (ErrCudaSetMaxSharedMem, f"{ErrCudaSetMaxSharedMemSuggestion} Error message: {line}", True)

    if ErrCudaInitializationErrorPattern.search(line_lower):
        return (ErrCudaInitializationError, f"{ErrCudaInitializationErrorSuggestion} Error message: {line}", False)

    if ErrCudaInsufficientDriverPattern.search(line_lower):
        return (ErrCudaInsufficientDriver, f"{ErrCudaInsufficientDriverSuggestion} Error message: {line}", False)

    # Try to detect specific CUDA error
    cuda_error = detect_cuda_error(line)
    if cuda_error:
        if cuda_error.error_class in CUDA_ERROR_NOT_PROGRAM_ERROR_CLASS:
            return cuda_error.error_class, cuda_error.suggestion, False
        return cuda_error.error_class, cuda_error.suggestion, True

    # Generic CUDA error fallback
    if ErrCudaErrorPattern.search(line_lower):
        return (ErrCudaError, f"{ErrCudaErrorSuggestion} Error message: {line}", True)

    return AttributionSubClassEnum.UNDETERMINED, "", False
