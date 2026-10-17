# recorder-sdk

Transcription pipeline SDK for an all-day wearable recorder (XIAO nRF52840 Sense + microSD, synced over USB).
For now it runs on any audio file, e.g. phone recordings.

Pipeline: audio file → faster-whisper → speaker label (me / enrolled people / other, by voiceprint) → SQLite.
Files are deduplicated by content hash, so re-running on the same folder is cheap.

## Install

```bash
pip install -e '.[whisper,dev]'
```

GPU: faster-whisper needs CUDA 12 cuBLAS + cuDNN 9. If they aren't installed system-wide
(typical on Windows: `cublas64_12.dll is not found`), install them from pip:

```bash
pip install -e '.[whisper,cuda]'
```

On Windows the SDK adds those DLL folders to the search path automatically. `--device auto` uses the GPU when available.

Windows PowerShell 5 doesn't support `&&`; run commands one per line.

## CLI

```bash
recorder transcribe ~/recordings/            # a file or a folder
recorder transcribe a.m4a --model small --device cpu --compute-type int8
recorder transcribe ~/recordings/ --hotwords "Logan nRF52840 XIAO"   # names Whisper mishears
recorder transcribe ~/recordings/ --model large-v3 --reprocess       # redo files already in the db
recorder list
recorder show 1
recorder search "hello"
```

`--db` picks the SQLite file (default `recorder.db`).
`--prompt` passes Whisper an initial prompt (a sentence in your own style and vocabulary); `--hotwords` biases it towards specific terms.

## Data

Each segment stores its text, speaker, `avg_logprob` and `no_speech_prob`, plus per-word
timestamps and probabilities (`store.segments(id, words=True)`). Each recording stores the
transcriber settings it was made with in `Recording.params` (model, prompt, hotwords, ...),
so you can tell which recordings are worth re-running after a change.

The schema is versioned with `PRAGMA user_version` and migrated automatically when a `Store`
opens a database; existing databases are upgraded in place.

## Web UI

```bash
recorder ui                           # opens http://127.0.0.1:8765 in your browser
```

- Browse recordings, play audio from any segment or word (click the time or a word), switch
  between segments and merged speaker turns, see low-confidence words underlined, search.
- 导入 (or drag files onto the page): files are copied into the library folder
  (`recordings/` next to the database, `--library` to change) and transcribed one by one in
  the background, with speaker labels.
- 改名: give a recording a display name (`recorder rename ID "name"` in the CLI). The file on
  disk keeps its name.
- 删除: delete a recording with its transcript; the audio file is deleted too if it is in
  the library folder, otherwise kept (`recorder delete ID` in the CLI keeps the file).
- 实时录音: record from the microphone; the page shows the transcript and speakers as you
  talk, and saves a WAV in the library as a normal recording when you stop.
- 说话人: list and delete voiceprints, tune the threshold; tick segments in a transcript and
  enter a name to enroll that person.
- Unsure words (orange underline): click one to get up to 4 guesses at what was really said
  (Whisper re-decoding that spot at higher temperature, plus dictionary words one or two
  sounds away), each with its IPA. Pick one, type another, or confirm the original. The
  transcript is fixed in place and the pick is logged.
  Suggestions are worked out once per recording in the background (after an import, after
  a live recording, and on `recorder ui` start for older recordings) and stored, so the
  popup opens instantly. The CLI does the same after `recorder transcribe` (skip with
  `--no-suggest`); `recorder suggest [IDS]` catches up on older recordings.
  The same pass force-aligns those sentences (torchaudio's MMS_FA wav2vec2 model, ~1.2 GB
  downloaded on first use, comes with the `speaker` extra) for word times accurate to
  ~20 ms instead of Whisper's 0.1-0.3 s; "▶ 录音原声" plays just that word, cut on the
  server.
- 纠错本: every corrected word, with the intended word's IPA against what Whisper heard and
  the sounds that differ (e.g. /θ/ said as /s/), the standard pronunciation (a human
  recording from Wiktionary via dictionaryapi.dev, cached; the browser's own voice when
  offline or missing) next to your original audio. 导出 Markdown downloads the list.
  IPA and sound-alikes need `pip install -e '.[learn]'`.

`recorder ui --model large-v3 --hotwords "Logan"` sets the Whisper options for imports and
live recording. Whisper and the speaker model run one job at a time, so live transcription
pauses while an import is being transcribed. The server only listens on localhost.

## Who is speaking

Each segment is labelled with the enrolled person whose voiceprint it matches best (SpeechBrain
ECAPA; the ~80 MB model downloads from Hugging Face on first use), or `other` when nobody
matches well enough. You are `me`.

```bash
pip install -e '.[whisper,speaker]'
recorder enroll my_voice.m4a          # 1-2 min of only you talking; run once
recorder transcribe ~/recordings/     # labels segments once anyone is enrolled
recorder show 3 --scores              # numbered segments with best match and similarity
recorder enroll --name alice --recording 3 --segments 4,7,10-12   # Alice from her segments
recorder enroll --name bob bob.m4a    # or from a file of only Bob talking
recorder speakers                     # list enrolled people; --forget NAME to delete
recorder relabel --rescore            # re-embed all recordings after enrolling someone
recorder relabel --threshold 0.4      # re-apply a threshold to stored matches (instant)
recorder show 3 --turns               # merge consecutive segments into speaker turns
```

The default threshold is 0.3; check `--scores` on a recording you know and adjust. In
`--scores`, `[other ~alice +0.21]` means the best match was Alice but below the threshold.
Enrolling someone again under the same name replaces their voiceprint, so pass all their
segments at once; 30 s or more of their speech works best.

## Python

```python
from recorder import Pipeline, Store, WhisperTranscriber, FolderSource

with Store("recorder.db") as store:
    pipe = Pipeline(WhisperTranscriber(model="large-v3"), store)
    pipe.process_source(FolderSource("~/recordings"))
    for rec, seg in store.search("hello"):
        print(rec.path, seg.start, seg.text)
```

## Extension points

- `SpeakerLabeler` (`speaker.py`): sets `Segment.speaker`. `VoiceprintLabeler` takes any `Embedder`; `NoopLabeler` leaves it `None`.
- `AudioSource` (`sync.py`): where files come from. `FolderSource` today; a USB device source later, which also fills `started_at`.

## Tests

```bash
pytest
```
