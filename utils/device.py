import os
import warnings

import torch

_torch_npu_imported = False


def _try_import_torch_npu():
    """Import torch_npu once so the npu backend registers with PyTorch."""
    global _torch_npu_imported
    if _torch_npu_imported:
        return True
    try:
        import torch_npu  # noqa: F401
        _torch_npu_imported = True
        return True
    except ImportError:
        return False


def _npu_available():
    return hasattr(torch, "npu") and callable(
        getattr(torch.npu, "is_available", None)
    ) and torch.npu.is_available()


def _npu_device_index():
    raw = os.environ.get("ASCEND_DEVICE_ID", "0")
    try:
        return int(raw)
    except (TypeError, ValueError):
        warnings.warn(
            f"Invalid ASCEND_DEVICE_ID={raw!r}, using 0",
            stacklevel=2,
        )
        return 0


def _initialize_npu_device():
    """Force-register npu:N (see lipku/LiveTalking#574) then return that device."""
    index = _npu_device_index()
    device = torch.device(f"npu:{index}")
    try:
        torch.npu.set_device(device)
        dummy = torch.tensor([1.0], device=device)
        torch.npu.synchronize()
        torch.npu.empty_cache()
        del dummy
        return device
    except Exception as e:
        warnings.warn(f"NPU initialization failed, falling back: {e}")
        return None


def initialize_device():
    """Pick an inference device: NPU, then CUDA, then MPS, then CPU."""
    _try_import_torch_npu()
    if _npu_available():
        npu = _initialize_npu_device()
        if npu is not None:
            return npu
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_type(device):
    """Return the backend name (cuda / npu / mps / cpu) without an index."""
    if isinstance(device, torch.device):
        return device.type
    text = str(device)
    return text.split(":", 1)[0]


def is_accelerator(device):
    return device_type(device) in ("cuda", "npu", "mps")


def load_checkpoint(path, device):
    """Load a .pth onto ``device``. Accelerators use map_location=device."""
    if is_accelerator(device):
        return torch.load(path, map_location=device)
    return torch.load(path, map_location=lambda storage, loc: storage)
