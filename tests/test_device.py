import os
import unittest
from unittest.mock import MagicMock, patch

import torch

import utils.device as device_mod


class FakeDevice:
    def __init__(self, spec):
        text = str(spec)
        if ":" in text:
            self.type, index = text.split(":", 1)
            self.index = int(index)
        else:
            self.type = text
            self.index = None

    def __eq__(self, other):
        return str(self) == str(other)

    def __str__(self):
        if self.index is not None:
            return f"{self.type}:{self.index}"
        return self.type

    def __repr__(self):
        return f"device(type='{self.type}'" + (
            f", index={self.index})" if self.index is not None else ")"
        )


class FakeNpu:
    def __init__(self, available=True, fail_init=False):
        self._available = available
        self._fail_init = fail_init
        self.set_device = MagicMock(side_effect=self._set_device)
        self.synchronize = MagicMock()
        self.empty_cache = MagicMock()
        self.get_device_name = MagicMock(return_value="Ascend910B3")

    def is_available(self):
        return self._available

    def _set_device(self, _device):
        if self._fail_init:
            raise RuntimeError("npu init failed")


class DeviceInitTestCase(unittest.TestCase):
    def setUp(self):
        device_mod._torch_npu_imported = False

    def test_prefers_npu_over_cuda(self):
        npu = FakeNpu(available=True)
        with patch.object(device_mod, "_try_import_torch_npu", return_value=True), \
             patch.object(device_mod.torch, "npu", npu, create=True), \
             patch.object(device_mod.torch, "device", FakeDevice), \
             patch.object(device_mod.torch, "tensor", MagicMock()), \
             patch.object(device_mod.torch.cuda, "is_available", return_value=True):
            chosen = device_mod.initialize_device()
        self.assertEqual(str(chosen), "npu:0")
        npu.set_device.assert_called_once()
        npu.synchronize.assert_called_once()
        npu.empty_cache.assert_called_once()

    def test_honors_ascend_device_id(self):
        npu = FakeNpu(available=True)
        with patch.dict(os.environ, {"ASCEND_DEVICE_ID": "1"}), \
             patch.object(device_mod, "_try_import_torch_npu", return_value=True), \
             patch.object(device_mod.torch, "npu", npu, create=True), \
             patch.object(device_mod.torch, "device", FakeDevice), \
             patch.object(device_mod.torch, "tensor", MagicMock()), \
             patch.object(device_mod.torch.cuda, "is_available", return_value=False):
            chosen = device_mod.initialize_device()
        self.assertEqual(str(chosen), "npu:1")

    def test_npu_init_failure_falls_back_to_cuda(self):
        npu = FakeNpu(available=True, fail_init=True)
        with patch.object(device_mod, "_try_import_torch_npu", return_value=True), \
             patch.object(device_mod.torch, "npu", npu, create=True), \
             patch.object(device_mod.torch, "device", FakeDevice), \
             patch.object(device_mod.torch.cuda, "is_available", return_value=True):
            with self.assertWarns(UserWarning):
                chosen = device_mod.initialize_device()
        self.assertEqual(str(chosen), "cuda")

    def test_cuda_when_npu_missing(self):
        with patch.object(device_mod, "_try_import_torch_npu", return_value=False), \
             patch.object(device_mod.torch, "npu", FakeNpu(available=False), create=True), \
             patch.object(device_mod.torch.cuda, "is_available", return_value=True):
            chosen = device_mod.initialize_device()
        self.assertEqual(device_mod.device_type(chosen), "cuda")

    def test_mps_when_cuda_and_npu_missing(self):
        fake_mps = MagicMock()
        fake_mps.is_available.return_value = True
        with patch.object(device_mod, "_try_import_torch_npu", return_value=False), \
             patch.object(device_mod.torch, "npu", FakeNpu(available=False), create=True), \
             patch.object(device_mod.torch.cuda, "is_available", return_value=False), \
             patch.object(device_mod.torch.backends, "mps", fake_mps, create=True):
            chosen = device_mod.initialize_device()
        self.assertEqual(device_mod.device_type(chosen), "mps")

    def test_cpu_fallback(self):
        fake_mps = MagicMock()
        fake_mps.is_available.return_value = False
        with patch.object(device_mod, "_try_import_torch_npu", return_value=False), \
             patch.object(device_mod.torch, "npu", FakeNpu(available=False), create=True), \
             patch.object(device_mod.torch.cuda, "is_available", return_value=False), \
             patch.object(device_mod.torch.backends, "mps", fake_mps, create=True):
            chosen = device_mod.initialize_device()
        self.assertEqual(str(chosen), "cpu")

    def test_missing_torch_npu_package_is_silent(self):
        device_mod._torch_npu_imported = False
        with patch.object(device_mod.importlib.util, "find_spec", return_value=None):
            self.assertFalse(device_mod._try_import_torch_npu())

    def test_torch_npu_hccl_failure_warns_and_returns_false(self):
        device_mod._torch_npu_imported = False
        real_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torch_npu" or name.startswith("torch_npu."):
                raise ImportError(
                    "libhccl.so: cannot open shared object file: No such file or directory"
                )
            return real_import(name, globals, locals, fromlist, level)

        with patch.object(device_mod.importlib.util, "find_spec", return_value=object()), \
             patch("builtins.__import__", fake_import):
            with self.assertWarns(UserWarning):
                self.assertFalse(device_mod._try_import_torch_npu())
        self.assertFalse(device_mod._torch_npu_imported)


class DeviceHelperTestCase(unittest.TestCase):
    def test_disables_backend_autoload(self):
        self.assertEqual(os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD"), "0")

    def test_is_accelerator(self):
        self.assertTrue(device_mod.is_accelerator(torch.device("cuda")))
        self.assertTrue(device_mod.is_accelerator(FakeDevice("npu:0")))
        self.assertTrue(device_mod.is_accelerator("mps"))
        self.assertFalse(device_mod.is_accelerator(torch.device("cpu")))
        self.assertFalse(device_mod.is_accelerator("cpu"))

    def test_load_checkpoint_maps_accelerator_to_device(self):
        target = FakeDevice("npu:0")
        with patch.object(device_mod.torch, "load", return_value={"ok": True}) as load:
            result = device_mod.load_checkpoint("wav2lip.pth", target)
        self.assertEqual(result, {"ok": True})
        load.assert_called_once_with("wav2lip.pth", map_location=target)

    def test_load_checkpoint_cpu_uses_storage_lambda(self):
        with patch.object(device_mod.torch, "load", return_value={"ok": True}) as load:
            result = device_mod.load_checkpoint("wav2lip.pth", torch.device("cpu"))
        self.assertEqual(result, {"ok": True})
        kwargs = load.call_args.kwargs
        self.assertTrue(callable(kwargs.get("map_location")))


if __name__ == "__main__":
    unittest.main()
