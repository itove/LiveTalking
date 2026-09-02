import importlib.metadata
import importlib.util
import os
import subprocess
import sys
import warnings

# PyTorch 2.5+ auto-loads torch_npu on `import torch` via entry points.
# That crashes with libhccl.so when CANN is not sourced. Load NPU ourselves.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch

_torch_npu_imported = False
_torch_npu_failed = False

_CANN_LOAD_HINT = (
    "torch_npu is installed but failed to load. Source CANN before starting: "
    "source /usr/local/Ascend/ascend-toolkit/set_env.sh"
)
_VERSION_HINT = (
    "torch and torch-npu major.minor must match "
    "(see https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md). "
    "A mismatch aborts with: Duplicated key 'pinned_reserve_segment_size_mb'."
)


def _version_major_minor(ver):
    core = str(ver).split("+", 1)[0]
    for sep in ("rc", "a", "b", "dev"):
        idx = core.find(sep)
        if idx != -1:
            core = core[:idx]
    parts = [p for p in core.split(".") if p]
    if len(parts) < 2:
        return core, "0"
    return parts[0], parts[1]


def _installed_torch_npu_version():
    for name in ("torch-npu", "torch_npu"):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _npu_stack_compatible():
    """Return (ok, warning_or_empty). Must not import torch_npu."""
    npu_ver = _installed_torch_npu_version()
    if not npu_ver:
        return False, "torch-npu is not installed"
    torch_ver = torch.__version__
    if _version_major_minor(torch_ver) != _version_major_minor(npu_ver):
        return False, (
            f"Skipping NPU: torch {torch_ver} is not compatible with "
            f"torch-npu {npu_ver}. {_VERSION_HINT}"
        )
    return True, ""


def _probe_torch_npu_import():
    """Import torch_npu in a child process.

    A torch/torch-npu mismatch can abort() in libtorch_npu.so during dlopen.
    That cannot be caught with try/except in this process.
    Set LIVETALKING_NPU_PROBE=0 to skip (faster, but a mismatch will coredump).
    """
    if os.environ.get("LIVETALKING_NPU_PROBE", "1") in ("0", "false", "False"):
        return True, ""
    env = os.environ.copy()
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import os; os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'; "
                "import torch; import torch_npu",
            ],
            env=env,
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, str(e)
    if result.returncode == 0:
        return True, ""
    err = (result.stderr or result.stdout).decode("utf-8", "replace").strip()
    return False, err or f"torch_npu probe exited {result.returncode}"


def _try_import_torch_npu():
    """Import torch_npu once so the npu backend registers with PyTorch."""
    global _torch_npu_imported, _torch_npu_failed
    if _torch_npu_imported:
        return True
    if _torch_npu_failed:
        return False
    if importlib.util.find_spec("torch_npu") is None:
        return False
    ok, reason = _npu_stack_compatible()
    if not ok:
        _torch_npu_failed = True
        warnings.warn(reason, stacklevel=2)
        return False
    ok, err = _probe_torch_npu_import()
    if not ok:
        _torch_npu_failed = True
        warnings.warn(
            f"{_CANN_LOAD_HINT} Probe import aborted ({err}). {_VERSION_HINT}",
            stacklevel=2,
        )
        return False
    try:
        import torch_npu  # noqa: F401
        _torch_npu_imported = True
        return True
    except Exception as e:
        _torch_npu_failed = True
        warnings.warn(f"{_CANN_LOAD_HINT} ({e})", stacklevel=2)
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
