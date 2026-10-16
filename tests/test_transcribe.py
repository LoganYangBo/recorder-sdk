import os
import sys

from recorder import transcribe


def test_nvidia_dll_dirs_added_on_windows(tmp_path, monkeypatch):
    (tmp_path / "nvidia" / "cublas" / "bin").mkdir(parents=True)
    (tmp_path / "nvidia" / "cudnn" / "bin").mkdir(parents=True)
    (tmp_path / "nvidia" / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "platform", "win32")
    added = []
    monkeypatch.setattr(os, "add_dll_directory", added.append, raising=False)
    monkeypatch.setenv("PATH", "orig")
    sys.modules.pop("nvidia", None)

    transcribe._add_nvidia_dll_dirs()

    assert sorted(os.path.basename(os.path.dirname(d)) for d in added) == ["cublas", "cudnn"]
    assert os.environ["PATH"].endswith("orig") and "cublas" in os.environ["PATH"]
