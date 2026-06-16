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
from typing import Any, NamedTuple, Optional

from attribution_lib.attr_sub_class import AttributionSubClassEnum

from .lepton_cuda_errors import is_cuda_error
from .lepton_nccl_errors import is_nccl_error

# --------------------------------------------------------------------------- #
#  Regex roots                                                               #
# --------------------------------------------------------------------------- #
_ERRORS = r"error|errors|exception|fail|failed|failure|fault"
_ERRORS_CAP = r"Error|Errors|Exception|Fail|Failure|Fault"
_ERRORS_ALL_CAP = r"ERROR|ERRORS|EXCEPTION|FAIL|FAILURE|FAULT"

_SIGS = r"sig(fpe|ill|segv|bus|abrt|trap|emt|sys|term|int|quit|kill|hup)"
_SIGS_ALL_CAP = r"SIG(FPE|ILL|SEGV|BUS|ABRT|TRAP|EMT|SYS|TERM|INT|QUIT|KILL|HUP)"

# --------------------------------------------------------------------------- #
#  Core patterns                                                             #
# --------------------------------------------------------------------------- #
_error_patterns = "|".join(
    [
        rf"\b({_ERRORS}|invalidresponse|traceback)[_,]?\b",
        rf"\b([A-Z][a-z]*)*[_]*({_ERRORS_CAP}|InvalidResponse|Traceback)[,]?\b",
        rf"\b[A-Z_]*({_ERRORS_ALL_CAP}|INVALID_RESPONSE|TRACEBACK)[A-Z_,=]*\b",
        rf"\b({_SIGS})s?\b",
        rf"\b({_SIGS_ALL_CAP})s?\b",
    ]
)
ErrorPatterns = re.compile(_error_patterns)

_check_configuration_pattern = re.compile(r"check your[a-z ]*configuration")
_error_variables_pattern = re.compile(rf"[a-z0-9_]+_({_ERRORS})")

_exclude_error_patterns = "|".join(
    [
        rf"\b(lib|ignore_|ignore )({_ERRORS}|{_SIGS})",
        rf"\b({_ERRORS})s?\.(json|jpeg|jpg|yaml|yml|png|pdf|gif|svg|wav|mp3|mp4|avi|mov)\b",
        rf"\b({_ERRORS})s?_[a-z0-9\-_ ]*\b",
        rf"[/]+[a-z\-_]*({_ERRORS})[a-z\-_\/]*",
        r"error parsing requirements for",
        r"-[-]?(verbose|verbosity) [a-z|]*error[a-z|]*",
        r"couldn't find ffmpeg or avconv - defaulting to ffmpeg",
    ]
)
ExcludeErrorPatterns = re.compile(_exclude_error_patterns, re.IGNORECASE)
ExcludeNCCLFileNotFoundPattern = re.compile(
    r"nccl info[a-z0-9/=: ]*(libnccl-net\.so|libnccl-tuner\.so|libnccl-profiler\.so)[a-z0-9/=: ]*no such file or directory",
    re.IGNORECASE,
)

ErrEthNetworkConnection = AttributionSubClassEnum.ETHNET_CONNECTION
ErrEthNetworkAddrInUse = AttributionSubClassEnum.ETHNET_ADDR_IN_USE
ErrEthNetworkDNSUnresolve = AttributionSubClassEnum.ETHNET_DNS_UNRESOLVE

ErrFilesystemNoSpace = AttributionSubClassEnum.FILESYSTEM_NO_SPACE
ErrFilesystemFileNameTooLong = AttributionSubClassEnum.FILESYSTEM_FILENAME_TOO_LONG
ErrFilesystemMissFileDir = AttributionSubClassEnum.FILESYSTEM_MISS_FILE_DIR
ErrFilesystemFileTooLarge = AttributionSubClassEnum.FILESYSTEM_FILE_TOO_LARGE
ErrFileDescriptorNotSeekable = AttributionSubClassEnum.FILE_DESCRIPTOR_NOT_SEEKABLE

ErrPythonImport = AttributionSubClassEnum.PYTHON_IMPORT
ErrLibUndefinedSymbol = AttributionSubClassEnum.LIB_UNDEFINED_SYMBOL
ErrPythonModuleNotFound = AttributionSubClassEnum.PYTHON_MODULE_NOT_FOUND
ErrCudnnParamNullPointer = AttributionSubClassEnum.CUDNN_PARAM_NULL_POINTER
ErrNullPointer = AttributionSubClassEnum.NULL_POINTER
ErrPILInputToBig = AttributionSubClassEnum.PIL_INPUT_TOO_BIG
ErrPytorchShapeMismatch = AttributionSubClassEnum.PYTORCH_SHAPE_MISMATCH
ErrIndexOutOfBound = AttributionSubClassEnum.INDEX_OUT_OF_BOUND

ErrGPUOutOfMemory = AttributionSubClassEnum.GPU_OUT_OF_MEMORY
ErrCublasAllocFail = AttributionSubClassEnum.CUBLAS_ALLOC_FAIL
ErrCudnnDeviceMemAllocFail = AttributionSubClassEnum.CUDNN_DEVICE_MEM_ALLOC_FAIL
ErrCudnnHostMemAllocFail = AttributionSubClassEnum.CUDNN_HOST_MEM_ALLOC_FAIL

ErrProgramErrorSignalRecv = AttributionSubClassEnum.PROGRAM_ERROR_SIGNAL_RECV
ErrTerminationSignalRecv = AttributionSubClassEnum.TERMINATION_SIGNAL_RECV
ErrSignalRecv = AttributionSubClassEnum.SIGNAL_RECV

ErrCudaSetMaxSharedMem = AttributionSubClassEnum.CUDA_SET_MAX_SHARED_MEM
ErrCudaSystemNotReady = AttributionSubClassEnum.CUDA_SYSTEM_NOT_READY
ErrCudaInsufficientDriver = AttributionSubClassEnum.CUDA_INSUFFICIENT_DRIVER
ErrCudaInitializationError = AttributionSubClassEnum.CUDA_INITIALIZATION_ERROR
ErrCudaError = AttributionSubClassEnum.CUDA_ERROR

ErrProgram = AttributionSubClassEnum.PROGRAM
ErrGPUInfra = AttributionSubClassEnum.GPU_INFRA
ErrUnknown = AttributionSubClassEnum.UNDETERMINED

# --------------------------------------------------------------------------- #
#  Suggestions                                                               #
# --------------------------------------------------------------------------- #
ErrPythonImportSuggestion = (
    "Please check the import module name for python package. If it's a 'from' list "
    "(from ... import ...), make sure the import exists."
)
ErrPythonModuleNotFoundSuggestion = (
    "Please check the import module name. Make sure the module is installed and the name is spelled correctly."
)
ErrLibUndefinedSymbolSuggestion = "A library contains undefined symbol. Please check whether the relevant libraries and their dependent libraries are installed."
ErrCudnnParamNullPointerSuggestion = "A null pointer is passed to a cudnn library function. Please check the input argument to the cudnn library function."
ErrNullPointerSuggestion = "Program dereferences a null pointer. Please check the memory accesses."
ErrPILInputTooBigSuggestion = (
    "Increase the limit by setting PIL.Image.MAX_IMAGE_PIXELS = {pixels}. "
    "If the input is trusted, the check can be disabled by setting PIL.Image.MAX_IMAGE_PIXELS = None"
)
ErrIndexOutOfBoundSuggestion = "Please check the index."

# CUDA / memory
_COMMON_REDUCE_GPU_MEM = (
    "Try to reduce GPU memory usage, e.g. reduce the batch size. "
    "Try to free up GPU memory before making new allocations. "
)
ErrGPUOutOfMemorySuggestion = "Device is out of memory. " + _COMMON_REDUCE_GPU_MEM
ErrCublasAllocFailSuggestion = "Cublas failed to allocate memory. " + _COMMON_REDUCE_GPU_MEM
ErrCudnnDeviceMemAllocFailSuggestion = "Cudnn failed to allocate device memory. " + _COMMON_REDUCE_GPU_MEM
ErrCudnnHostMemAllocFailSuggestion = "Cudnn failed to allocate host memory. Try to increase the available CPU memory."
ErrCudaMemAllocFailSuggestion = (
    "CUDA memory allocation failed. "
    + _COMMON_REDUCE_GPU_MEM
    + "If CUDA tries to allocate host memory, try to increase the available CPU memory."
)
ErrPytorchShapeMismatchSuggestion = "Shape mismatch."
ErrProgramErrorSignalRecvSuggestion = "Program killed by the OS. There's a serious error."
ErrTerminationSignalRecvSuggestion = "Receiving a termination signal. Please check program log for the root cause."
ErrSignalRecvSuggestion = "Program terminated because it received an unhandled signal. Please check the log message."
ErrProgramSuggestion = "Program exits abnormally."

# File system & network
ErrFilesystemNoSpaceSuggestion = "Storage/disk/file system is full."
ErrFilesystemFileNameTooLongSuggestion = "File/directory name is too long. Please shorten the file/directory name."
ErrFilesystemMissFileOrDirSuggestionDefault = (
    "File or directory doesn't exist. Please check the path of the file or directory."
)
ErrFilesystemFileTooLargeSuggestion = "The file is too large. Please split the file."
ErrEthNetworkConnectionSuggestion = "Ethernet is disconnected. The remote host might be down."
ErrEthNetworkAddrInUseSuggestion = "The socket address is already in use elsewhere."
ErrEthNetworkDNSUnresolveSuggestion = ""
ErrGPUInfraSuggestion = "Detected GPU infra problem. Please contact support."
ErrUnknownSuggestion = "Please check the log for the error reasons."

# --------------------------------------------------------------------------- #
#  Dedicated regex helpers                                                   #
# --------------------------------------------------------------------------- #
ErrNullPointerPattern = re.compile(r"null[ ]*pointer[ ]*exception|segmentation[ ]*fault")
ErrExtractPixelPattern = re.compile(r"\(([0-9]*) pixels\)")
ErrPytorchShapeMismatchPattern = re.compile(r"(shape|size)[ ]+mismatch")

ErrProgramErrorSignalRecvPattern = re.compile(r"\bsig(?:fpe|ill|segv|bus|abrt|trap|emt|sys)\b")
ErrTerminationSignalRecvPattern = re.compile(r"\bsig(?:term|int|quit|kill|hup)\b")
ErrSignalRecvPattern = re.compile(r"signal[0-9 ]*")

# File-system patterns (matched on lower-case lines)
ErrFilesystemNoSpacePattern = re.compile(
    "|".join(
        [
            r"(storage|disk|filesystem)[a-z ]*full",
            r"(filesystem|disk) quota exceeded",
            r"no space left on device",
        ]
    )
)
ErrFilesystemFileNameTooLongPattern = re.compile(r"(file|host)[ ]*name too long")
ErrFilesystemFileTooLargePattern = re.compile(r"file[ ]*too[ ]*large")
ErrFilesystemMissFileOrDirPattern = re.compile(
    "|".join(
        [
            r"no[ ]*such[ ]*file[ ]*or[ ]*directory",
            r"(file|directory|sh|df)[:a-z0-9\./'\" ]*not[ ]*found",
            r"(unable to|couldn't|could not|cannot) (open|find)",
            r"path[ ]*does[ ]*not[ ]*exist",
        ]
    ),
    re.IGNORECASE,
)
KeyNotFoundPattern = re.compile(r"key [a-z0-9_'\.]* not found")

# GPU / CUDA memory alloc
ErrGPUOutOfMemoryPattern = re.compile(
    r"(device|cuda|gpu|rocm)[a-z :.']*out[ ]*of[ ]*memory|out[ ]*of[ ]*memory[a-z :.]*(device|cuda|gpu|rocm)",
    re.IGNORECASE,
)
ErrCudaMemAllocFailPattern = re.compile(r"CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED")

# Network
ErrEthNetworkConnectionPattern = re.compile(
    r"connection[a-z ]*(close|error|refuse|reset|abort)|(host|network)[a-z ]*(unreachable|down)|no route to host|drop(ped)? connection",
    re.IGNORECASE,
)
ErrEthNetworkAddrInUsePattern = re.compile(r"address[ a-zA-Z]*in[ ]*use", re.IGNORECASE)
ErrEthNetworkDNSUnresolvePattern = re.compile(
    r"name[ ]*or[ ]*service[ ]*not[ ]*known|host[ ]*name[ ]*lookup[ ]*failure",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
#  Helper utilities                                                          #
# --------------------------------------------------------------------------- #
def with_error_message(suggestion: str, line: str) -> str:
    """Append the concrete log line to the human suggestion."""
    return f"{suggestion} Error message: {line}"


# --------------------------------------------------------------------------- #
#  Primitive detectors                                                       #
# --------------------------------------------------------------------------- #
def is_no_such_file_and_directory_error(line: str) -> bool:
    """Special-case filter for benign 'no such file' lines."""
    if ErrFilesystemMissFileOrDirPattern.search(line):
        if any(
            s in line
            for s in (
                ".triton/autotune",
                "NCCL INFO NET/Plugin",
                "command not found",
                "cudaErrorSharedObjectSymbolNotFound",
            )
        ) or KeyNotFoundPattern.search(line):
            return False
        return True
    return False


def found_error(line: str) -> bool:
    return bool(ErrorPatterns.search(line)) or is_no_such_file_and_directory_error(line)


def should_exclude_line(line_lower: str) -> bool:
    """Return True if the line looks like an ignorable error string."""
    return any(
        [
            ExcludeErrorPatterns.search(line_lower),
            ExcludeNCCLFileNotFoundPattern.search(line_lower),
            "error executing" in line_lower,
            "exceptiongroup" in line_lower,
            "pip's dependency resolver" in line_lower,
            (_check_configuration_pattern.search(line_lower) and "fails" in line_lower),
            (_error_variables_pattern.search(line_lower) and "environment variable" in line_lower),
        ]
    )


# --------------------------------------------------------------------------- #
#  Categorisation helpers                                                    #
# --------------------------------------------------------------------------- #
def is_allocation_failure(line: str, line_lower: str) -> tuple[AttributionSubClassEnum, str, bool]:
    if ErrGPUOutOfMemoryPattern.search(line_lower):
        return (
            ErrGPUOutOfMemory,
            with_error_message(ErrGPUOutOfMemorySuggestion, line),
            True,
        )
    if "CUBLAS_STATUS_ALLOC_FAILED" in line:
        return (
            ErrCublasAllocFail,
            with_error_message(ErrCublasAllocFailSuggestion, line),
            True,
        )
    if "CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED" in line:
        return (
            ErrCudnnDeviceMemAllocFail,
            with_error_message(ErrCudnnDeviceMemAllocFailSuggestion, line),
            True,
        )
    if "CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED" in line:
        return (
            ErrCudnnHostMemAllocFail,
            with_error_message(ErrCudnnHostMemAllocFailSuggestion, line),
            True,
        )
    return AttributionSubClassEnum.UNDETERMINED, "", False


def is_signal_error(line: str, line_lower: str) -> tuple[AttributionSubClassEnum, str, bool, bool]:
    if ErrProgramErrorSignalRecvPattern.search(line_lower):
        return (
            ErrProgramErrorSignalRecv,
            with_error_message(ErrProgramErrorSignalRecvSuggestion, line),
            True,
            True,
        )
    if ErrTerminationSignalRecvPattern.search(line_lower):
        return (
            ErrTerminationSignalRecv,
            with_error_message(ErrTerminationSignalRecvSuggestion, line),
            False,
            False,
        )
    if ErrSignalRecvPattern.search(line_lower):
        return (
            ErrSignalRecv,
            with_error_message(ErrSignalRecvSuggestion, line),
            True,
            False,
        )
    return AttributionSubClassEnum.UNDETERMINED, "", False, False


def is_file_system_error(line: str, line_lower: str) -> tuple[AttributionSubClassEnum, str, bool]:
    if ErrFilesystemNoSpacePattern.search(line_lower):
        return (
            ErrFilesystemNoSpace,
            with_error_message(ErrFilesystemNoSpaceSuggestion, line),
            True,
        )
    if ErrFilesystemFileNameTooLongPattern.search(line_lower):
        return (
            ErrFilesystemFileNameTooLong,
            with_error_message(ErrFilesystemFileNameTooLongSuggestion, line),
            True,
        )
    if is_no_such_file_and_directory_error(line):
        return (
            ErrFilesystemMissFileDir,
            with_error_message(ErrFilesystemMissFileOrDirSuggestionDefault, line),
            True,
        )
    if ErrFilesystemFileTooLargePattern.search(line_lower):
        return (
            ErrFilesystemFileTooLarge,
            with_error_message(ErrFilesystemFileTooLargeSuggestion, line),
            True,
        )
    return AttributionSubClassEnum.UNDETERMINED, "", False


def is_eth_network_error(line: str, line_lower: str) -> tuple[AttributionSubClassEnum, str, bool]:
    if ErrEthNetworkConnectionPattern.search(line_lower):
        return (
            ErrEthNetworkConnection,
            with_error_message(ErrEthNetworkConnectionSuggestion, line),
            True,
        )
    if ErrEthNetworkAddrInUsePattern.search(line_lower):
        return (
            ErrEthNetworkAddrInUse,
            with_error_message(ErrEthNetworkAddrInUseSuggestion, line),
            True,
        )
    if ErrEthNetworkDNSUnresolvePattern.search(line_lower):
        return (
            ErrEthNetworkDNSUnresolve,
            with_error_message(ErrEthNetworkDNSUnresolveSuggestion, line),
            True,
        )
    return AttributionSubClassEnum.UNDETERMINED, "", False


def is_program_error(line_lower: str) -> bool:
    return not (
        "during handling of the above exception, another exception occurred" in line_lower
        or "the above exception was the direct cause of the following exception" in line_lower
    ) and any(kw in line_lower for kw in ("error", "exception", "assertion", "fail", "invalidresponse"))


def is_python_lib_error(line: str, line_lower: str) -> tuple[AttributionSubClassEnum, str]:
    if "ImportError" in line:
        return ErrPythonImport, with_error_message(ErrPythonImportSuggestion, line)
    if "ModuleNotFound" in line:
        return ErrPythonModuleNotFound, with_error_message(ErrPythonModuleNotFoundSuggestion, line)
    if "undefined symbol" in line:
        return ErrLibUndefinedSymbol, with_error_message(ErrLibUndefinedSymbolSuggestion, line)
    if "index out of bound" in line_lower:
        return ErrIndexOutOfBound, with_error_message(ErrIndexOutOfBoundSuggestion, line)
    if "PIL.Image.DecompressionBomb" in line:
        pixel_match = ErrExtractPixelPattern.search(line)
        pixel = pixel_match.group(1) if pixel_match else "unknown"
        return ErrPILInputToBig, ErrPILInputTooBigSuggestion.format(pixels=pixel)
    return AttributionSubClassEnum.UNDETERMINED, ""


def is_null_pointer_error(line: str, line_lower: str) -> tuple[AttributionSubClassEnum, str, bool]:
    if "cudnn_status_bad_param_null_pointer" in line_lower:
        return (
            ErrCudnnParamNullPointer,
            with_error_message(ErrCudnnParamNullPointerSuggestion, line),
            True,
        )
    if ErrNullPointerPattern.search(line_lower):
        return (
            ErrNullPointer,
            with_error_message(ErrNullPointerSuggestion, line),
            True,
        )
    return AttributionSubClassEnum.UNDETERMINED, "", False


# --------------------------------------------------------------------------- #
#  Main classification                                                       #
# --------------------------------------------------------------------------- #
def classify_line(
    line: str,
) -> tuple[AttributionSubClassEnum, str, bool, bool]:
    """
    Parameters
    ----------
    line : str
        Raw log line

    Returns
    -------
    err_class : str
        One of the ERR_* constants
    suggestion : str
        Human-readable suggestion (may include original message)
    precise_error : bool
        True  → err_class pinpoints a concrete failure cause
        False → err_class is generic (ErrUnknown / ErrCuda etc.)
    is_program_error : bool
        True  → program side error
        False → infra / environment level error
    """
    line_lower = line.lower()

    # GPU infra NVML
    if "nvml_error_" in line or "nvmLError_" in line:
        return ErrGPUInfra, with_error_message(ErrGPUInfraSuggestion, line), True, False

    # Python import & misc
    err_cls, sugg = is_python_lib_error(line, line_lower)
    if err_cls is not AttributionSubClassEnum.UNDETERMINED:
        return err_cls, sugg, True, True

    # Shape mismatch
    if ErrPytorchShapeMismatchPattern.search(line_lower):
        return (
            ErrPytorchShapeMismatch,
            with_error_message(ErrPytorchShapeMismatchSuggestion, line),
            True,
            True,
        )

    # Null-pointer deref
    err_cls, sugg, ok = is_null_pointer_error(line, line_lower)
    if ok:
        return err_cls, sugg, True, True

    # Memory allocation failures
    err_cls, sugg, ok = is_allocation_failure(line, line_lower)
    if ok:
        return err_cls, sugg, True, True

    # File system
    err_cls, sugg, ok = is_file_system_error(line, line_lower)
    if ok:
        return err_cls, sugg, True, True

    err_cls, sugg, is_prog_err = is_cuda_error(line, line_lower)
    if err_cls is not AttributionSubClassEnum.UNDETERMINED:
        return err_cls, sugg, True, is_prog_err

    err_cls, sugg, ok, is_prog_err = is_nccl_error(line)
    if ok:
        return err_cls, sugg, True, is_prog_err

    # Generic program error
    if is_program_error(line_lower):
        return ErrProgram, with_error_message(ErrProgramSuggestion, line), True, True

    # Signals
    err_cls, sugg, ok, is_prog_err = is_signal_error(line, line_lower)
    if ok:
        return err_cls, sugg, True, is_prog_err

    # Infra-level net errors
    err_cls, sugg, ok = is_eth_network_error(line, line_lower)
    if ok:
        return err_cls, sugg, True, False

    # Fallback – unknown
    return ErrUnknown, ErrUnknownSuggestion, False, False


class LeptonSymptom(NamedTuple):
    errClass: AttributionSubClassEnum
    suggestion: str
    preciseError: bool
    isProgramError: bool

    def to_event_payload(self) -> dict[str, Any]:
        return self._asdict()

    def to_node_symptom(self) -> str:
        return self.errClass.to_short_string()


def check_line(line: str) -> Optional[LeptonSymptom]:
    if should_exclude_line(line.lower()):
        return None

    errClass, suggestion, preciseError, isProgramError = classify_line(line)
    if errClass is not AttributionSubClassEnum.UNDETERMINED:
        return LeptonSymptom(errClass, suggestion, preciseError, isProgramError)
    return None
