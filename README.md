# recorder-sdk

Transcription pipeline SDK for an all-day wearable recorder (XIAO nRF52840 Sense + microSD, synced over USB).
For now it runs on any audio file, e.g. phone recordings.

Pipeline: audio file → faster-whisper → speaker label (me / other, stub) → SQLite.
Files are deduplicated by content hash, so re-running on the same folder is cheap.

## Install

```bash
pip install -e '.[whisper,dev]'
```

GPU: faster-whisper needs CUDA 12 + cuDNN 9 (see the faster-whisper README). `--device auto` uses the GPU when available.

## CLI

```bash
recorder transcribe ~/recordings/            # a file or a folder
recorder transcribe a.m4a --model small --device cpu --compute-type int8
recorder list
recorder show 1
recorder search "hello"
```

`--db` picks the SQLite file (default `recorder.db`).

## Python

```python
from recorder import Pipeline, Store, WhisperTranscriber, FolderSource

with Store("recorder.db") as store:
    pipe = Pipeline(WhisperTranscriber(model="large-v3"), store)
    pipe.process_source(FolderSource("~/recordings"))
    for rec, seg in store.search("hello"):
        print(rec.path, seg.start, seg.text)
```

## Extension points (not implemented yet)

- `SpeakerLabeler` (`speaker.py`): set `Segment.speaker` to `"me"` / `"other"` using a voiceprint. Default `NoopLabeler` leaves it `None`.
- `AudioSource` (`sync.py`): where files come from. `FolderSource` today; a USB device source later, which also fills `started_at`.

## Tests

```bash
pytest
```
