# recorder-sdk

[![CI](https://github.com/LoganYangBo/recorder-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/LoganYangBo/recorder-sdk/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

A Python SDK that turns hours of everyday audio into a searchable, speaker-labelled transcript
database, plus a local web UI and an API for apps built on top of it.

It is the software half of an all-day wearable recorder (XIAO nRF52840 Sense + microSD). It works
on any audio file today, such as phone recordings.

```
device / folder ──► sync ──► faster-whisper ──► speaker labels ──► word alignment ──► SQLite
                     │        (word timings,     (me / enrolled     (wav2vec2,          │
                     │         confidence)        people / other)    ~20 ms)            ▼
                     └─► library (Ogg Opus)                                   Recorder API · web UI · CLI
```

## Contents

- [Features](#features)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Syncing the recorder](#syncing-the-recorder)
- [Speaker identification](#speaker-identification)
- [Web UI](#web-ui)
- [Python API](#python-api)
- [CLI reference](#cli-reference)
- [Data model](#data-model)
- [Architecture](#architecture)
- [Development](#development)

## Features

- **Transcription** with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), with
  per-word timestamps and confidence, initial prompts and hotwords, on GPU or CPU.
- **Speaker labels** from voiceprints (SpeechBrain ECAPA): each segment is `me`, an enrolled
  person, or `other`.
- **Device sync** over USB or Bluetooth, plus live transcription over Bluetooth. Files are
  deduplicated by content hash, compressed to Ogg Opus (about 8x smaller than WAV) and processed
  one at a time. A file that fails is logged and retried
  without blocking the rest.
- **Wall-clock time** for every recording, taken from file names or metadata, so you can ask
  "what did I say on Tuesday afternoon".
- **Full-text search** in any language (SQLite FTS5, trigram tokenizer).
- **Pronunciation review**: low-confidence words get correction candidates, IPA and a phoneme
  diff. Forced alignment cuts the original audio, which you can compare with a dictionary
  recording or a synthesised female or male voice.
- **Local web UI** to browse, play, search, import, record live, enroll speakers and review words.
- **Stable `Recorder` API** for apps, phrased in days, people and words rather than tables.
- **Versioned schema**: existing databases migrate in place on open.

## Installation

Requires Python 3.10+. The core depends only on NumPy; features are opt-in extras.

```bash
pip install "recorder[whisper,speaker,learn] @ git+https://github.com/LoganYangBo/recorder-sdk@v0.3.0"
```

For development, from a clone:

```bash
pip install -e ".[whisper,speaker,learn,tts,dev]"
```

| Extra     | Adds                                                     | Pulls in                         |
|-----------|----------------------------------------------------------|----------------------------------|
| `whisper` | Transcription                                            | faster-whisper, PyAV             |
| `cuda`    | CUDA 12 cuBLAS and cuDNN 9 from pip, for the GPU         | nvidia-cublas-cu12, nvidia-cudnn |
| `speaker` | Speaker labels and forced alignment                      | SpeechBrain, torchaudio (torch)  |
| `learn`   | IPA, sound-alike candidates, phoneme diffs               | cmudict                          |
| `tts`     | Synthesised female and male reference voices             | kokoro-onnx                      |
| `device`  | Setting the recorder's clock over USB                    | pyserial                         |
| `dev`     | Tests and linting                                        | pytest, ruff                     |

**GPU.** faster-whisper needs CUDA 12 cuBLAS and cuDNN 9. If they aren't installed system-wide
(typical on Windows: `cublas64_12.dll is not found`), install the `cuda` extra. On Windows the SDK
adds those DLL folders to the search path itself. `--device auto` (the default) uses the GPU when
one is available.

**Models** download on first use: Whisper and the ECAPA speaker model from Hugging Face (about
80 MB for ECAPA), the MMS_FA aligner through torchaudio (about 1.2 GB), and Kokoro from GitHub
releases (about 340 MB, into `models/kokoro/` next to the database).

**Windows PowerShell 5** doesn't support `&&`, so run the commands one per line.

## Quick start

```bash
recorder enroll my_voice.m4a          # 1-2 minutes of only you talking; once
recorder transcribe ~/recordings/     # a file or a folder
recorder list
recorder show 1 --turns
recorder search "ship friday" --speaker me
recorder ui                           # http://127.0.0.1:8765
```

`recorder --db PATH COMMAND` picks the database (default `recorder.db`).

## Syncing the recorder

```bash
recorder sync                 # the plugged-in recorder, once
recorder sync --watch         # wait for it, and sync every time it is plugged in
recorder sync D:\phone-recs   # or any folder
recorder sync --delete        # free the device's card once files are safely synced
recorder status               # what was synced, and what failed

recorder sync --ble --watch --delete   # over Bluetooth: every 10 minutes when it's in range
recorder live                 # transcribe what the recorder hears, live, over Bluetooth
recorder device               # its state, battery and recordings waiting, over Bluetooth
```

Over Bluetooth (the `device` extra, which installs bleak) a sync sets the recorder's clock,
downloads the finished recordings to `inbox/` next to the database (resuming an interrupted
download), and imports them like any other source; `--delete` then removes them from the card.

### Device convention

The recorder mounts as a USB drive:

```
<drive>/recorder-device.json       {"id": "..."}  marks the drive as a recorder
<drive>/REC/20261016T091500.wav    one file per recording, named by its local start time
<drive>/REC/20261016/091500.wav    the same, for firmware limited to 8.3 names
```

The firmware in [`firmware/`](firmware/README.md) writes the second form, as IMA ADPCM WAV
(4:1), which PyAV decodes like any WAV. The device has no battery-backed clock: every Bluetooth
connection sets it, `recorder settime` sets it over USB serial, and `recorder sync --watch`
does so whenever the device is plugged in (all need the `device` extra).

`find_devices()` looks for the marker on Windows drive letters and under `/media`, `/run/media`,
`/mnt` and `/Volumes`. Other timestamp formats work too (`20261016_091500`,
`2026-10-16 09.15.00`, ISO 8601 with an offset), and phone recordings fall back to their
`creation_time` metadata. Times are stored as local wall-clock time without an offset. Run
`recorder dates` to fill them in for recordings imported before this existed.

### What a sync does

For each file:

1. Hash the file. If it is already in the database, it's done.
2. Copy it into the library as `recordings/YYYY/MM/DD/<name>.opus`. WAV is re-encoded to mono
   16 kHz Ogg Opus at 32 kbps (`--keep-wav` skips that). The copy goes through a `.part` file,
   so an interrupted copy never looks complete.
3. Transcribe, label speakers, align and analyse unsure words, as `transcribe` does.
4. Record the outcome in the `files` table. A failing file (corrupt, unreadable) is retried on
   the next sync, up to 3 times (`--retry` resets the limit).
5. With `--delete`, remove the file from the device, only once its copy is in both the library
   and the database.

## Speaker identification

Each segment is labelled with the enrolled person whose voiceprint matches it best, or `other`
when nobody matches well enough. You are `me`.

```bash
recorder enroll my_voice.m4a                                      # you
recorder enroll --name alice --recording 3 --segments 4,7,10-12   # Alice, from her segments
recorder enroll --name bob bob.m4a                                # Bob, from a file of him alone
recorder speakers                                                 # list; --forget NAME
recorder show 3 --scores                                          # best match and similarity
recorder relabel --rescore                                        # re-embed after enrolling
recorder relabel --threshold 0.4                                  # re-apply a threshold (instant)
```

The default threshold is 0.3. Check `--scores` on a recording you know and adjust it.
`[other ~alice +0.21]` means the best match was Alice, but below the threshold. Enrolling under
an existing name replaces that voiceprint, so pass all of a person's segments at once; 30 s or
more of their speech works best.

## Web UI

```bash
recorder ui                                   # http://127.0.0.1:8765
recorder ui --model large-v3 --hotwords "PyTorch Kubernetes"
```

The server listens on localhost only. The interface is in English or Chinese, following the
browser's language; the button at the top right switches it. Whisper and the speaker model run
one job at a time in a background queue, which the page lists under Background jobs.

- **Browse and play.** Recordings are grouped by day with their start times. Click a time or a
  word to play from there. Switch between segments and merged speaker turns, and search.
- **Import.** Click Import or drag files onto the page. Files are copied into the library and
  transcribed in the background.
- **Record live.** Transcribes the microphone as you talk and saves a normal recording when you
  stop.
- **Manage.** Rename a recording, set its start time, or delete it (the audio file goes too if it
  is in the library).
- **Speakers.** Lists and deletes voiceprints and tunes the threshold. Tick segments in a
  transcript to enroll that person.
- **Unsure words** are underlined in orange. Click one for up to 4 candidates, each with its IPA:
  Whisper re-decoding that spot at higher temperatures, plus dictionary words one or two sounds
  away. Pick one, type another, or confirm the original. Candidates and word times are worked out
  once per recording in the background, so the popup opens instantly.
- **Notebook.** Every corrected or confirmed word, with its IPA against what Whisper heard and
  the sounds that differ (e.g. /θ/ said as /s/). Play your original audio, cut to the aligned
  word, next to a reference: a human recording from Wiktionary (via dictionaryapi.dev), or an
  American female or male voice (Kokoro-82M). Export Markdown downloads the list.

## Python API

Apps (English practice, daily review, ...) read through `Recorder`. Its methods answer in terms of
days, people and words, so apps don't depend on database tables that may change between
versions.

```python
from recorder import Recorder

with Recorder("recorder.db") as rec:
    for day in rec.days():  # dates with recordings
        print(day.date, day.seconds, day.speakers)
    for u in rec.my_utterances(day="2026-10-16"):  # what I said, as speaker turns
        print(u.at, u.text)
    rec.utterances(speaker="alice", since="2026-10-01")  # anyone's, any range
    rec.search("ship friday", speaker="me")  # full-text, any language
    for w in rec.unsure_words(speaker="me"):  # words to practise
        print(w.word, w.candidates, w.sentence, w.at)
    rec.corrections()  # the notebook
```

| Method | Returns |
|--------|---------|
| `recordings(since, until, day)` | `Recording` objects that started in the range |
| `days()` | `Day(date, recordings, seconds, speakers)` per day, oldest first |
| `utterances(speaker, day, since, until, recording)` | `Utterance` turns in time order |
| `my_utterances(**filters)` | `utterances(speaker="me", ...)` |
| `search(query, speaker, since, until, day, limit)` | Matching segments as `Utterance` |
| `unsure_words(speaker, since, until, day, recording, below, include_reviewed)` | `UnsureWord` with candidates |
| `corrections(include_confirmed)` | `Correction` entries, newest first |

Time arguments accept `"2026-10-16"`, `"2026-10-16T09:00:00"`, `date` or `datetime`. Ranges are
half-open, `[since, until)`. `Utterance.at` is the wall-clock time, or `None` when the
recording's start time is unknown; undated results sort last.

Ingesting from code:

```python
from recorder import Pipeline, Store, Syncer, WhisperTranscriber, find_devices

with Store("recorder.db") as store:
    pipe = Pipeline(WhisperTranscriber(model="large-v3"), store)
    for device in find_devices():
        for result in Syncer(store, pipe, "recordings").sync(device):
            print(result.path, result.status, result.error)
```

### Extension points

| Interface | Module | Implementations |
|-----------|--------|-----------------|
| `Transcriber` | `transcribe.py` | `WhisperTranscriber` |
| `SpeakerLabeler` | `speaker.py` | `VoiceprintLabeler` (takes any `Embedder`, e.g. `EcapaEmbedder`), `NoopLabeler` |
| `AudioSource` | `sync.py` | `FolderSource`, `DeviceSource` (Bluetooth sync downloads into a folder first) |

## CLI reference

| Command | Purpose |
|---------|---------|
| `transcribe PATH...` | Transcribe files or folders. `--model`, `--device`, `--compute-type`, `--language`, `--prompt`, `--hotwords`, `--reprocess`, `--no-suggest` |
| `sync [SOURCE]` | Sync the device or a folder. `--ble`, `--watch`, `--interval`, `--library`, `--delete`, `--keep-wav`, `--retry` |
| `live` | Live transcription from the recorder over Bluetooth. `--save` keeps it as a recording |
| `device` | The recorder's state, battery and files, over Bluetooth |
| `settime` | Set the recorder's clock over USB |
| `status` | Synced and failed files |
| `dates [--force]` | Fill in start times from file names or metadata |
| `list` | Recordings with their start times |
| `show ID` | A transcript. `--turns`, `--scores` |
| `search QUERY` | Full-text search. `--speaker`, `--day`, `--limit` |
| `rename ID NAME` / `delete ID` | Rename a recording, or delete it (keeps the file) |
| `enroll` / `speakers` / `relabel` | Manage voiceprints and labels |
| `suggest [IDS]` | Analyse unsure words in older recordings |
| `ui` | Start the web UI |

Run `recorder COMMAND --help` for every option.

## Data model

One SQLite file. Its schema is versioned with `PRAGMA user_version` and migrated automatically
when a `Store` opens it.

| Table | Holds |
|-------|-------|
| `recordings` | Path, content hash, duration, `started_at`, display name, transcriber settings (`params`) |
| `segments` | Text, speaker, timings, `avg_logprob`, `no_speech_prob`; indexed by `segments_fts` |
| `words` | Per-word timings and probability, the original word if corrected, candidates, alignment state |
| `corrections` | Every word the user corrected or confirmed |
| `voiceprints` | Enrolled speakers' voiceprints |
| `files` | Per-source-file sync status, error and attempts |

`Recording.params` records the model, prompt and hotwords each transcript was made with, so you
can tell which recordings are worth re-running after a change.

## Architecture

| Module | Role |
|--------|------|
| `api.py` | `Recorder`: the read API for apps |
| `sync.py` | Audio sources and start-time detection |
| `clock.py` | Setting the recorder's clock over USB serial |
| `ble.py` | The recorder over Bluetooth: clock, file download, live audio |
| `adpcm.py` | Decoding the recorder's IMA ADPCM audio |
| `ingest.py` | `Syncer`: copy, compress, process and track each file |
| `pipeline.py` | Transcribe, label, store one file |
| `transcribe.py` | faster-whisper wrapper; alternative decodes for unsure words |
| `speaker.py` | Voiceprints and labelling |
| `align.py` | Forced alignment (MMS_FA) for word times |
| `learn.py` | IPA, sound-alikes, phoneme diffs (CMU dict) |
| `tts.py` | Kokoro reference voices |
| `ops.py` | Background analysis of unsure words |
| `store.py` | SQLite schema, migrations, queries |
| `web.py`, `web.html` | Local web UI (standard library HTTP server, no build step) |
| `live.py` | Live microphone transcription |
| `cli.py` | `recorder` command |

## Development

```bash
pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest
```

The tests use fakes for every model, so they need no GPU and no downloads. CI (GitHub Actions)
runs ruff and pytest on Ubuntu and Windows with Python 3.11 and 3.13 on every push and pull
request.
