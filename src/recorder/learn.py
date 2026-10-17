"""Help for words Whisper was unsure about: candidate corrections, phonetic
transcriptions (IPA, from the CMU Pronouncing Dictionary) and a phoneme-level
diff that shows how what was heard differs from the intended word.

The CMU dictionary comes from the optional `cmudict` package
(`pip install 'recorder[learn]'`); without it there is no IPA and no
sound-alike candidates, but Whisper's own alternatives still work.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import lru_cache

ARPABET_IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ", "B": "b",
    "CH": "tʃ", "D": "d", "DH": "ð", "EH": "ɛ", "ER": "ɝ", "EY": "eɪ", "F": "f",
    "G": "ɡ", "HH": "h", "IH": "ɪ", "IY": "i", "JH": "dʒ", "K": "k", "L": "l",
    "M": "m", "N": "n", "NG": "ŋ", "OW": "oʊ", "OY": "ɔɪ", "P": "p", "R": "ɹ",
    "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v",
    "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}  # fmt: skip
UNSTRESSED = {"AH": "ə", "ER": "ɚ"}  # reduced vowels

MAX_CANDIDATES = 4


def normalize(word: str) -> str:
    """'Friday.' -> 'friday'; keeps inner apostrophes (don't)."""
    return re.sub(r"^[^\w']+|[^\w']+$", "", word.strip().lower()).strip("'")


def recase(new: str, old: str) -> str:
    """Put `new` in place of `old`, keeping old's surrounding punctuation and
    capitalisation: recase('think', 'Sink,') -> 'Think,'.
    """
    m = re.match(r"^(\W*)(.*?)(\W*)$", old, re.S)
    lead, core, trail = m.groups() if m else ("", old, "")
    if core[:1].isupper() and new[:1].islower():
        new = new[:1].upper() + new[1:]
    return f"{lead}{new}{trail}"


@lru_cache(maxsize=1)
def _cmu() -> dict[str, list[list[str]]] | None:
    try:
        import cmudict
    except ImportError:
        return None
    return cmudict.dict()


def phonemes(word: str) -> list[str] | None:
    """ARPAbet phonemes with stress digits (first pronunciation), or None."""
    d = _cmu()
    prons = d.get(normalize(word)) if d else None
    return prons[0] if prons else None


def _bare(phones: list[str]) -> tuple[str, ...]:
    return tuple(p.rstrip("012") for p in phones)


def ipa_of(phone: str) -> str:
    base = phone.rstrip("012")
    if phone.endswith("0") and base in UNSTRESSED:
        return UNSTRESSED[base]
    return ARPABET_IPA.get(base, base.lower())


def ipa(word: str) -> str | None:
    phones = phonemes(word)
    return "".join(ipa_of(p) for p in phones) if phones else None


def _edit_ops(a: tuple[str, ...], b: tuple[str, ...]) -> list[tuple[str, int | None, int | None]]:
    """Levenshtein alignment of a -> b as (op, i, j) with op in same/sub/del/ins."""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(
                d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1])
            )
    ops, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            ops.append(("same" if a[i - 1] == b[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", None, j - 1))
            j -= 1
    return ops[::-1]


def _distance(a: tuple[str, ...], b: tuple[str, ...], limit: int) -> int:
    """Edit distance, giving up (returning limit + 1) once it must exceed limit."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        if min(cur) > limit:
            return limit + 1
        prev = cur
    return prev[-1]


def compare(target: str, heard: str) -> dict:
    """How `heard` differs from the intended `target`, phoneme by phoneme.

    Returns {target_ipa, heard_ipa, ops: [{op, target, heard}]}, where op is
    same / sub (target sound said as another) / del (target sound missing) /
    ins (extra sound). IPA fields are None for words the dictionary lacks.
    """
    t, h = phonemes(target), phonemes(heard)
    out = {"target_ipa": ipa(target), "heard_ipa": ipa(heard), "ops": []}
    if not t or not h:
        return out
    for op, i, j in _edit_ops(_bare(t), _bare(h)):
        out["ops"].append(
            {
                "op": op,
                "target": ipa_of(t[i]) if i is not None else None,
                "heard": ipa_of(h[j]) if j is not None else None,
            }
        )
    return out


@lru_cache(maxsize=1)
def _by_length() -> dict[int, list[tuple[tuple[str, ...], str]]]:
    index: dict[int, list] = {}
    for word, prons in (_cmu() or {}).items():
        if not word.isalpha():
            continue
        for p in prons:
            bare = _bare(p)
            index.setdefault(len(bare), []).append((bare, word))
    return index


def sound_alikes(
    word: str, rank: Callable[[str], int | None] | None = None, limit: int = MAX_CANDIDATES
) -> list[str]:
    """Dictionary words one phoneme away from `word` (then two, if none).

    `rank(w)` says how common a word is (lower = more common) or None to
    leave it out; without it, every dictionary word counts, which includes
    many obscure ones."""
    phones = phonemes(word)
    if not phones:
        return []
    target, me = _bare(phones), normalize(word)
    index = _by_length()
    for max_d in (1, 2):
        found: dict[str, tuple[int, int]] = {}
        for n in range(len(target) - max_d, len(target) + max_d + 1):
            for bare, w in index.get(n, ()):
                if w == me or w in found:
                    continue
                dist = _distance(target, bare, max_d)
                if dist > max_d:
                    continue
                r = rank(w) if rank else 0
                if r is not None:
                    found[w] = (dist, r)
        if found:
            return sorted(found, key=lambda w: (found[w], w))[:limit]
    return []


def candidates(
    heard: str,
    whisper_alternatives: list[str],
    rank: Callable[[str], int | None] | None = None,
    limit: int = MAX_CANDIDATES,
) -> list[str]:
    """Up to `limit` possible intended words for `heard`: Whisper's own
    alternatives for that spot first (in order, most frequent first), then
    dictionary words that sound alike. Can be empty."""
    me = normalize(heard)
    counts: dict[str, int] = {}
    for w in whisper_alternatives:
        w = normalize(w)
        if w and w != me:
            counts[w] = counts.get(w, 0) + 1
    out = sorted(counts, key=lambda w: -counts[w])
    for w in sound_alikes(heard, rank, limit):
        if w not in out:
            out.append(w)
    return out[:limit]
