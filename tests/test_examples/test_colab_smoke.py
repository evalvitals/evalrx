"""The Colab smoke must distinguish completed checks from failed/partial runs."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SMOKE = Path(__file__).resolve().parents[2] / "tools/colab/jax_smoke.py"


def smoke_module():
    spec = importlib.util.spec_from_file_location("colab_smoke", SMOKE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_keeps_failure_even_when_later_steps_pass(tmp_path):
    module = smoke_module()
    path = tmp_path / "report.json"
    report = module.SmokeReport(path)

    def failed():
        raise RuntimeError("load failed")

    assert report.step("load", failed) is False
    assert json.loads(path.read_text())["status"] == "running"
    assert report.step("later", lambda: {"ok": True}) is True
    assert report.finish() == 1
    saved = json.loads(path.read_text())
    assert saved["status"] == "failed"
    assert saved["steps"][0]["error"] == "RuntimeError: load failed"


@pytest.mark.parametrize("load_error", [False, True])
def test_main_records_load_and_returns_an_exit_code(monkeypatch, tmp_path, load_error):
    torch = pytest.importorskip("torch")
    pytest.importorskip("PIL")
    import evalrx.models
    from evalrx.models.backends.jax import backend

    device = SimpleNamespace(device_kind="test CPU", memory_stats=lambda: {})
    fake_jax = SimpleNamespace(default_backend=lambda: "cpu", devices=lambda: [device],
                               local_devices=lambda: [device], process_count=lambda: 1)
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setattr(backend, "configure_jax_runtime", lambda device: None)

    class FakeModel:
        capabilities = frozenset()
        modalities = frozenset({"text", "image"})

        def load(self):
            if load_error:
                raise RuntimeError("checkpoint unavailable")

        def forward(self, *args, **kwargs):
            return SimpleNamespace(logits=torch.ones(2, 4), extras={"image_token_mask": torch.tensor([True])},
                                   token_type_map=SimpleNamespace(grids=[(1, 1, 1)]))

        def generate(self, inputs):
            return "red"

    monkeypatch.setattr(evalrx.models, "compose", lambda *a, **kw: FakeModel())
    output = tmp_path / "smoke.json"
    exit_code = smoke_module().main(["--device", "cpu", "--suite", "image", "--output", str(output)])
    data = json.loads(output.read_text())
    assert exit_code == int(load_error)
    assert data["status"] == ("failed" if load_error else "passed")
    assert [s["name"] for s in data["steps"]] == (["hardware", "load"] if load_error else ["hardware", "load", "image"])
