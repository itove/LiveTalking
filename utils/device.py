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
    if sys.version_info >= (3, 14) and os.environ.get(
        "LIVETALKING_ALLOW_PY314_NPU", ""
    ) not in ("1", "true", "True"):
        return False, (
            f"Skipping NPU: Python {sys.version.split()[0]} makes libtorch_npu "
            "abort with duplicated allocator hooks "
            "('pinned_reserve_segment_size_mb'). Use Python 3.10–3.12, or set "
            "LIVETALKING_ALLOW_PY314_NPU=1 to probe anyway."
        )
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
    extra = ":".join(
        p for p in (
            "/usr/local/Ascend/driver/lib64/driver",
            "/usr/local/Ascend/driver/lib64/common",
            "/usr/local/Ascend/driver/lib64",
        ) if os.path.isdir(p)
    )
    if extra:
        env["LD_LIBRARY_PATH"] = extra + ":" + env.get("LD_LIBRARY_PATH", "")
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


def _summarize_npu_probe_error(err):
    """Keep startup logs to one line; the C++ abort dump is not actionable."""
    if "pinned_reserve_segment_size_mb" in err:
        torch_ver = getattr(torch, "__version__", "?")
        npu_ver = _installed_torch_npu_version() or "?"
        return (
            f"libtorch_npu aborted (Duplicated key pinned_reserve_segment_size_mb); "
            f"torch={torch_ver} torch-npu={npu_ver} python={sys.version.split()[0]}. "
            "Install a matching CPU torch + torch-npu pair on Python 3.10–3.12."
        )
    lines = [ln.strip() for ln in err.splitlines() if ln.strip()]
    short = " | ".join(lines[:2])
    return short[:400] if short else err[:400]


_HAL_SO_CANDIDATES = (
    "/usr/local/Ascend/driver/lib64/driver/libascend_hal.so",
    "/usr/local/Ascend/driver/lib64/common/libascend_hal.so",
    "/usr/local/Ascend/driver/lib64/libascend_hal.so",
)

_NPU_INVISIBLE_HINT = (
    "torch_npu imported but ascend_hal reports no devices. "
    "In this shell: source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh; "
    "npu-smi info; l /dev/davinci*; groups (usually need HwHiAiUser). "
    "Also check LD_LIBRARY_PATH includes /usr/local/Ascend/driver/lib64/driver. "
    "CANN 9.0.0 with HDK 24.1.rc2.2 is a known mismatch."
)


def _preload_ascend_hal():
    """dlopen libascend_hal.so by absolute path.

    Changing LD_LIBRARY_PATH inside Python does not affect the dynamic linker.
    npu-smi can work while torch_npu still reports 'Can't get ascend_hal device count'.
    """
    import ctypes

    loaded = []
    for path in _HAL_SO_CANDIDATES:
        if not os.path.isfile(path):
            continue
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            loaded.append(path)
            break
        except OSError as e:
            warnings.warn(f"Failed to preload {path}: {e}", stacklevel=2)
    return loaded


def _quiet_torch_npu_owner_warnings():
    """CANN is installed as root; torch_npu warns when the process user differs."""
    warnings.filterwarnings(
        "ignore",
        message=r".*owner does not match the current owner.*",
        category=UserWarning,
        module=r"torch_npu\.utils\.collect_env",
    )


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
            f"{_CANN_LOAD_HINT} Probe import aborted ({_summarize_npu_probe_error(err)}). {_VERSION_HINT}",
            stacklevel=2,
        )
        return False
    try:
        _quiet_torch_npu_owner_warnings()
        _preload_ascend_hal()
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


def _visible_npu_ids():
    """Physical ids from ASCEND_RT_VISIBLE_DEVICES or ASCEND_VISIBLE_DEVICES."""
    raw = os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or os.environ.get(
        "ASCEND_VISIBLE_DEVICES", ""
    )
    if not str(raw).strip():
        return None
    ids = []
    for part in str(raw).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        # npu-smi chip form "7" or "7.0"
        part = part.split(".", 1)[0]
        try:
            ids.append(int(part))
        except ValueError:
            warnings.warn(
                f"Ignoring invalid visible NPU id {part!r} in {raw!r}",
                stacklevel=2,
            )
    return ids or None


def _npu_device_index():
    """Logical index for torch.device('npu:N').

    ASCEND_DEVICE_ID is an index into the visible list when
    ASCEND_RT_VISIBLE_DEVICES / ASCEND_VISIBLE_DEVICES is set, otherwise it is
    the global device id. Masking to card 7 makes that card npu:0 — do not also
    set ASCEND_DEVICE_ID=1.
    """
    raw = os.environ.get("ASCEND_DEVICE_ID", "0")
    try:
        requested = int(raw)
    except (TypeError, ValueError):
        warnings.warn(
            f"Invalid ASCEND_DEVICE_ID={raw!r}, using 0",
            stacklevel=2,
        )
        requested = 0
    visible = _visible_npu_ids()
    if visible is not None:
        if requested < 0 or requested >= len(visible):
            warnings.warn(
                f"ASCEND_DEVICE_ID={requested} is out of range for visible "
                f"NPUs {visible} (from ASCEND_RT_VISIBLE_DEVICES / "
                f"ASCEND_VISIBLE_DEVICES). Using logical npu:0 "
                f"(physical {visible[0]}).",
                stacklevel=2,
            )
            return 0
        return requested
    return requested


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
    loaded = _try_import_torch_npu()
    if loaded and not _npu_available():
        count = 0
        try:
            count = torch.npu.device_count()
        except Exception:
            pass
        warnings.warn(
            f"{_NPU_INVISIBLE_HINT} torch.npu.device_count()={count}.",
            stacklevel=2,
        )
    if loaded and _npu_available():
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
