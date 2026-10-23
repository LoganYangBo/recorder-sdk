"""The vocab: names and jargon from the user's life (people, products, places,
work terms) that Whisper would otherwise mishear: "Mia" heard as "me a",
"su7" as "SU 7". Every term is given to Whisper as a hotword, which biases
it towards that spelling.

Terms come from four places: added by hand (`recorder vocab add`), the names
of enrolled voiceprints, words the user corrected because Whisper misheard
them, and documents (`recorder vocab import notes.md`: names and unusual words
are picked out).

Whisper's prompt is short (about 220 tokens, shared with the initial prompt),
so only the newest terms that fit in `MAX_CHARS` are sent.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from .learn import _cmu, normalize

MAX_CHARS = 600  # roughly 150 tokens: leaves room for an initial prompt


def hotwords(terms: list[str], max_chars: int = MAX_CHARS) -> str | None:
    """The terms (newest first) that fit in `max_chars`, as one hotword string."""
    out, used = [], 0
    for t in terms:
        if used + len(t) + 1 > max_chars:
            break
        out.append(t)
        used += len(t) + 1
    return " ".join(out) or None


def from_store(db) -> Callable[[], str | None]:
    """A hotword source for WhisperTranscriber(vocab=...): reads the vocab of the
    database at `db` each time (its own connection, so any thread can call it)."""
    from .store import Store

    def get() -> str | None:
        with Store(db) as store:
            return hotwords([v["term"] for v in store.vocab()])

    return get


def _common(word: str) -> bool:
    """An everyday English word (in the CMU dictionary), lower-case as written."""
    d = _cmu()
    return d is not None and word.islower() and normalize(word) in d


def worth_keeping(term: str) -> bool:
    """Whether a corrected word is worth a hotword: a name, an acronym, something
    with digits, or a word not in the dictionary. Everyday words aren't: Whisper
    knows them, and they would crowd out the names."""
    term = term.strip()
    if not term:
        return False
    if " " in term or any(c.isdigit() for c in term) or not term.islower():
        return True
    d = _cmu()
    return d is None or normalize(term) not in d


_TOKEN = re.compile(r"[A-Za-z][\w'’.+-]*[\w+]|[A-Za-z]")
_SENTENCE_END = re.compile(r"[.!?。！？:\n]\s*$")


def candidates(text: str, known: set[str] | None = None) -> list[str]:
    """Terms worth adding from a document, in order of first appearance: capitalised
    words that don't merely start a sentence (names), runs of them ("Model Y"),
    acronyms, words with digits or inner capitals (su7, iPhone, PyTorch), and
    lower-case words missing from the dictionary (when cmudict is installed).
    Words already `known` (lower-cased) are skipped."""
    known = {k.lower() for k in known or ()}
    found: dict[str, None] = {}
    tokens = list(_TOKEN.finditer(text))
    run: list[str] = []

    def flush():
        if run:
            term = " ".join(run)
            if term.lower() not in known and len(term) > 1:
                found.setdefault(term, None)
            run.clear()

    for i, m in enumerate(tokens):
        word = m.group().rstrip(".")
        before = text[tokens[i - 1].end() : m.start()] if i else ""
        starts_sentence = i == 0 or bool(_SENTENCE_END.search(text[: m.start()][-3:] or "\n"))
        adjacent = i > 0 and before.strip() == "" and run
        special = any(c.isdigit() for c in word) or (
            any(c.isupper() for c in word[1:]) and not word.isupper()
        )
        acronym = len(word) >= 2 and word.isupper()
        capital = word[:1].isupper()
        if special or acronym or (capital and (not starts_sentence or adjacent)):
            if not adjacent:
                flush()
            run.append(word)
            continue
        flush()
        if word.islower() and len(word) > 3 and not _common(word) and _cmu() is not None:
            if word.lower() not in known:
                found.setdefault(word, None)
    flush()
    return list(found)
