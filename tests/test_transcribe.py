import os
import sys

import numpy as np

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


def test_whisper_passes_options_and_keeps_words(monkeypatch):
    calls = {}

    class W:
        def __init__(self, start, end, word, probability):
            self.start, self.end, self.word, self.probability = start, end, word, probability

    class S:
        start, end, text, avg_logprob, no_speech_prob = 0.0, 1.0, " hi Logan", -0.3, 0.02
        words = [W(0.0, 0.3, " hi", 0.9), W(0.3, 1.0, " Logan", 0.5)]

    class Info:
        language, duration = "en", 1.0

    class FakeModel:
        def transcribe(self, path, **kw):
            calls.update(kw)
            return iter([S()]), Info()

    t = transcribe.WhisperTranscriber(model="small", hotwords="Logan", initial_prompt="Hi Logan.")
    t._model = FakeModel()
    monkeypatch.setattr(t, "load_audio", lambda path: np.zeros(16000, np.float32))
    out = t.transcribe("a.wav")

    assert calls["hotwords"] == "Logan" and calls["initial_prompt"] == "Hi Logan."
    assert calls["vad_filter"] is True  # one second: not batched
    assert calls["word_timestamps"] is True
    (seg,) = out.segments
    assert seg.text == "hi Logan" and seg.avg_logprob == -0.3
    assert [w.word for w in seg.words] == ["hi", "Logan"]
    assert out.params["model"] == "small" and out.params["hotwords"] == "Logan"
    assert "initial_prompt" in out.params and "language" not in out.params


def test_long_audio_is_batched_with_vocab_hotwords(monkeypatch):
    import types

    calls = {}

    class Info:
        language, duration = "en", 120.0

    class FakeBatched:
        def __init__(self, model):
            calls["model"] = model

        def transcribe(self, audio, **kw):
            calls.update(kw)
            return iter([]), Info()

    fake_module = types.ModuleType("faster_whisper")  # CI runs without faster-whisper
    fake_module.BatchedInferencePipeline = FakeBatched
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    vocab = ["Mia"]
    t = transcribe.WhisperTranscriber(hotwords="PyTorch", vocab=lambda: " ".join(vocab))
    t._model = object()
    out = t.transcribe(np.zeros(16000 * 120, np.float32))
    assert calls["batch_size"] == transcribe.DEFAULT_BATCH
    assert calls["without_timestamps"] is False  # sentence-sized segments for speaker labels
    assert calls["hotwords"] == "PyTorch Mia"
    assert out.params["model"] == "large-v3-turbo" and out.params["batch_size"] == 8
    vocab.append("su7")  # read again for the next file
    t.transcribe(np.zeros(16000 * 120, np.float32))
    assert calls["hotwords"] == "PyTorch Mia su7"
