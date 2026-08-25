"""
Compatibility shims for Apple Silicon on macOS < 14.

macOS 13's Metal backend is missing several integer ops that current
transformers/torch assume. Each shim here exists because a specific call in the
generation path raises on this machine; they are no-ops on macOS 14+, CUDA and
CPU, so it is safe to apply them unconditionally at import time.
"""
import platform

import torch


def _is_macos_13_mps() -> bool:
    if not torch.backends.mps.is_available():
        return False
    if platform.system() != "Darwin":
        return False
    try:
        major = int(platform.mac_ver()[0].split(".")[0])
    except (ValueError, IndexError):
        return False
    return major < 14


def generation_device() -> str:
    """
    Best device for autoregressive generation.

    macOS 13's Metal backend fails inside generate() with an `mps_matmul`
    "incompatible dimensions / invalid shape" graph error (an abort, not a Python
    exception, so it cannot be caught). CPU generation of a 0.5B model is a few
    seconds per response, so it is the correct default here. Training is
    unaffected and still runs on MPS.
    """
    if torch.cuda.is_available():
        return "cuda"
    if _is_macos_13_mps():
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def apply_mps_compat_patches() -> bool:
    """
    Patch transformers' `isin_mps_friendly` to evaluate on CPU.

    `generate()` compares eos against pad via torch.isin, which on macOS 13 MPS
    raises "isin_Tensor_Tensor_out only works on floating types ... Received
    dtype: Long". The tensors involved are a handful of token ids, so running the
    comparison on CPU costs nothing.

    Returns True if patches were applied.
    """
    if not _is_macos_13_mps():
        return False

    from transformers import pytorch_utils

    original = pytorch_utils.isin_mps_friendly

    def isin_cpu_fallback(elements, test_elements):
        elements_is_mps = torch.is_tensor(elements) and elements.device.type == "mps"
        test_is_mps = torch.is_tensor(test_elements) and test_elements.device.type == "mps"
        if elements_is_mps or test_is_mps:
            device = elements.device if elements_is_mps else test_elements.device
            elements_cpu = elements.cpu() if torch.is_tensor(elements) else elements
            test_cpu = test_elements.cpu() if torch.is_tensor(test_elements) else test_elements
            return torch.isin(elements_cpu, test_cpu).to(device)
        return original(elements, test_elements)

    pytorch_utils.isin_mps_friendly = isin_cpu_fallback

    # Several transformers modules do `from ..pytorch_utils import
    # isin_mps_friendly`, which binds the original function object into their own
    # namespace at import time. Patching the source module alone leaves those
    # bindings pointing at the unpatched version (stopping_criteria is one such
    # caller), so rebind every module that holds a reference.
    import sys

    for module in list(sys.modules.values()):
        if module is None or not getattr(module, "__name__", "").startswith("transformers"):
            continue
        if getattr(module, "isin_mps_friendly", None) is original:
            module.isin_mps_friendly = isin_cpu_fallback

    return True
