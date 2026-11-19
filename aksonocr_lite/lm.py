"""Optional rescoring of the recognizer's CTC output with a character n-gram model and a word list.

The greedy decode keeps only the best class per frame. On degraded scans the right Thai cluster is often the
second or third candidate (ด/ต/ค, ช/ซ, a dropped or doubled tone mark), and a language model can tell which
candidate makes a word. This module re-reads only the uncertain positions of a line:

* only Thai-for-Thai substitutions and Thai insertions are ever proposed. Digits, Latin letters, punctuation and
  ฯ/ๆ are never changed and nothing is ever deleted, so amounts, codes and URLs come out exactly as the greedy
  decode read them;
* a change is kept only if it does not add out-of-dictionary characters, and - when it leaves that count
  unchanged (a real word swapped for another real word) - only if the combined score improves by `margin`;
* a line with fewer than `min_thai` Thai characters is not rescored at all: a two- or three-letter table cell
  or abbreviation gives the model no context, so it would only swap in whichever short string is more frequent.

Off by default. Enable with `Config(lm_path=...)`; build a model from your own text with
`python -m aksonocr_lite.lm build corpus.txt model.json.gz`.
"""
from __future__ import annotations

import gzip
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from functools import lru_cache

import numpy as np

_THAI = re.compile(r"[ก-ฺเ-๎]")
_THAI_RUN = re.compile(r"[ก-ฺเ-๎]+")


def _is_thai(cluster: str) -> bool:
    return bool(cluster) and bool(_THAI.search(cluster)) and "ฯ" not in cluster and "ๆ" not in cluster


class CharLM:
    """Interpolated Witten-Bell character n-gram. Whitespace is not modelled (Thai spacing is not lexical)."""

    def __init__(self, n: int = 6):
        self.n = n
        self.c: Counter = Counter()   # (history, char) -> count
        self.h: Counter = Counter()   # history -> count
        self.t: Counter = Counter()   # history -> distinct continuations
        self.vocab: set[str] = set()

    @property
    def start(self) -> str:
        return "^" * (self.n - 1)

    def train(self, text: str) -> None:
        text = "".join(unicodedata.normalize("NFC", text).replace("\u0e4d\u0e32", "\u0e33").split())
        text = self.start + text
        for i in range(self.n - 1, len(text)):
            ch = text[i]
            self.vocab.add(ch)
            for k in range(self.n):
                h = text[i - k:i]
                if self.c[(h, ch)] == 0:
                    self.t[h] += 1
                self.c[(h, ch)] += 1
                self.h[h] += 1

    def p(self, hist: str, ch: str) -> float:
        hist = hist[-(self.n - 1):]
        pr = 1.0 / (len(self.vocab) + 50)
        for k in range(len(hist) + 1):
            hh = hist[len(hist) - k:] if k else ""
            total = self.h.get(hh, 0)
            if total == 0:
                break
            distinct = self.t[hh]
            pr = (self.c.get((hh, ch), 0) + distinct * pr) / (total + distinct)
        return pr

    def logp(self, hist: str, text: str) -> tuple[float, str]:
        lp = 0.0
        for ch in text:
            if ch == " ":
                continue
            lp += math.log(self.p(hist, ch))
            hist = (hist + ch)[-(self.n - 1):]
        return lp, hist

    def save(self, path: str) -> None:
        grams = {h + ch: v for (h, ch), v in self.c.items()}
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump({"n": self.n, "grams": grams}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "CharLM":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            d = json.load(f)
        lm = cls(d["n"])
        for g, v in d["grams"].items():
            h, ch = g[:-1], g[-1]
            lm.c[(h, ch)] = v
            lm.h[h] += v
            lm.t[h] += 1
            lm.vocab.add(ch)
        return lm


def load_words(path: str | None = None) -> frozenset[str]:
    """One word per line; without a path, PyThaiNLP's word list if that package is installed."""
    if path:
        with open(path, encoding="utf-8") as f:
            words = [w.strip() for w in f]
    else:
        try:
            from pythainlp.corpus import thai_words
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("rescoring needs a Thai word list: pass lexicon_path=... or `pip install pythainlp`") from e
        words = thai_words()
    return frozenset(w for w in words if len(w) >= 2 and " " not in w)


class Rescorer:
    def __init__(self, lm: CharLM, words: frozenset[str], chars: list[str], alpha: float = 0.2, gate: float = 0.98,
                 margin: float = 2.0, topk: int = 6, blank_alt: float = 0.03, floor: float = 1e-4, beam: int = 8,
                 min_thai: int = 4):
        self.lm, self.words, self.chars = lm, words, chars
        self.alpha, self.gate, self.margin, self.min_thai = alpha, gate, margin, min_thai
        self.topk, self.blank_alt, self.floor, self.beam = topk, blank_alt, floor, beam
        self.maxw = max((len(w) for w in words), default=0)
        self.thai = np.array([_is_thai(c) for c in chars])
        self.oov = lru_cache(maxsize=100_000)(self._oov)

    # ------------------------------------------------------------ dictionary coverage
    def _oov_run(self, s: str) -> int:
        best = [0] * (len(s) + 1)
        for i in range(1, len(s) + 1):
            b = best[i - 1] + 1
            for L in range(2, min(self.maxw, i) + 1):
                if best[i - L] < b and s[i - L:i] in self.words:
                    b = best[i - L]
            best[i] = b
        return best[-1]

    def _oov(self, text: str) -> int:
        """Thai characters no segmentation into dictionary words can cover."""
        return sum(self._oov_run(r) for r in _THAI_RUN.findall(text))

    # ------------------------------------------------------------ lattice
    def positions(self, p: np.ndarray) -> tuple[list[list[tuple[str, float]]], bool]:
        """Greedy path as a list of candidate lists [(cluster, prob)], best first; and whether any is open."""
        ids, pr = p.argmax(-1), p.max(-1)
        chars, T, out, prev, any_open = self.chars, len(ids), [], 0, False
        t = 0
        while t < T:
            c = int(ids[t])
            if c == 0:
                if pr[t] < 1.0 - self.blank_alt:  # some other class may hold >= blank_alt
                    top = np.argsort(-p[t])[1:self.topk]
                    alts = [(chars[k], float(p[t, k])) for k in top if p[t, k] >= self.blank_alt and self.thai[k]]
                    alts = [x for x in alts if not ((t > 0 and chars[int(ids[t - 1])] == x[0])
                                                    or (t + 1 < T and chars[int(ids[t + 1])] == x[0]))]
                    if alts:
                        out.append([("", float(pr[t]))] + alts)
                        any_open |= pr[t] < self.gate
                prev = c
                t += 1
                continue
            if c != prev:
                t2 = t
                while t2 + 1 < T and int(ids[t2 + 1]) == c:
                    t2 += 1
                tb = t + int(np.argmax(pr[t:t2 + 1]))
                cand = [(chars[c], float(pr[tb]))]
                if self.thai[c] and pr[tb] < self.gate:
                    top = np.argsort(-p[tb])[1:self.topk]
                    cand += [(chars[k], float(p[tb, k])) for k in top if self.thai[k]]
                    any_open |= len(cand) > 1
                out.append(cand)
            prev = c
            t += 1
        return out, any_open

    def _decode(self, pos, gate: float) -> tuple[str, float]:
        lm, alpha = self.lm, self.alpha
        beams = [(0.0, lm.start, "")]
        for cand in pos:
            if cand[0][1] >= gate or len(cand) == 1:
                s_, p_ = cand[0]
                nb = []
                for sc, h, txt in beams:
                    l, h2 = lm.logp(h, s_)
                    nb.append((sc + math.log(max(p_, 1e-9)) + alpha * l, h2, txt + s_))
                beams = nb
                continue
            nb = []
            for sc, h, txt in beams:
                for s_, p_ in cand:
                    if p_ < self.floor:
                        continue
                    l, h2 = lm.logp(h, s_)
                    nb.append((sc + math.log(p_) + alpha * l, h2, txt + s_))
            nb.sort(key=lambda x: -x[0])
            seen, beams = set(), []
            for b in nb:
                key = (b[1], b[2][-12:])
                if key in seen:
                    continue
                seen.add(key)
                beams.append(b)
                if len(beams) >= self.beam:
                    break
        best = max(beams, key=lambda b: b[0])  # confident positions re-score the beams without re-sorting them
        return best[2], best[0]

    def choose(self, pos) -> str | None:
        """The rescored text, or None to keep the greedy reading."""
        greedy, gs = self._decode(pos, gate=-1.0)
        new, ns = self._decode(pos, gate=self.gate)
        if new == greedy:
            return None
        d = self.oov(new.replace(" ", "")) - self.oov(greedy.replace(" ", ""))
        if d > 0 or (d == 0 and ns - gs < self.margin):
            return None
        return new

    def __call__(self, p: np.ndarray) -> str | None:
        """p: (T, classes) posteriors of one line. -> corrected text, or None when the greedy reading stands."""
        pos, any_open = self.positions(p)
        if not any_open or sum(len(_THAI.findall(c[0][0])) for c in pos) < self.min_thai:
            return None
        return self.choose(pos)


def _main(argv: list[str]) -> int:
    if len(argv) != 4 or argv[1] != "build":
        print("usage: python -m aksonocr_lite.lm build corpus.txt model.json.gz", file=sys.stderr)
        return 2
    lm = CharLM()
    with open(argv[2], encoding="utf-8") as f:
        for block in re.split(r"\n\s*\n|\f", f.read()):  # a paragraph is one stream: line breaks fall mid-word in Thai
            lm.train(block)
    lm.save(argv[3])
    print(f"{lm.h['']} characters, {len(lm.c)} n-grams -> {argv[3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
