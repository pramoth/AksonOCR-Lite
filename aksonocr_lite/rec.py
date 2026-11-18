"""Thai/English text-line recognizer: PP-OCRv6-tiny (SVTR_LCNet) CTC with Thai cluster tokens.

Each output class is a Thai grapheme cluster (consonant + vowel/tone marks) or a single character,
so the greedy CTC decode is a plain lookup into `rec_charset.json`.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict

import cv2
import numpy as np

from .models import model_path, session

H, MAX_W = 48, 3200


def _prep(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    W = min(MAX_W, max(8, int(math.ceil(H * w / max(h, 1)))))
    x = cv2.resize(img, (W, H)).astype(np.float32) / 255.0
    return ((x - 0.5) / 0.5).transpose(2, 0, 1)


class Recognizer:
    """Batches crops of (nearly) equal width only. Right padding changes the SVTR neck's global attention:
    batching unequal widths together costs ~3 points of exact match, so each batch shares one `bucket`-rounded
    width. On CUDA the batch size is also rounded up to a power of two, because every new input shape costs
    a cuDNN algorithm search."""

    def __init__(self, device: str = "auto", budget: int = 3_000_000, flip: bool = True, flip_margin: float = 0.05,
                 bucket: int = 16):
        self.sess = session("rec.onnx", device)
        self.input = self.sess.get_inputs()[0].name
        self.chars = json.load(open(model_path("rec_charset.json"), encoding="utf-8"))["chars"]
        self.budget, self.flip, self.flip_margin, self.bucket = budget, flip, flip_margin, bucket
        self.gpu = self.sess.get_providers()[0] != "CPUExecutionProvider"

    def __call__(self, crops: list[np.ndarray]) -> list[tuple[str, float, float]]:
        """BGR crops -> [(text, confidence, char_confidence)]. With flip=True each crop is also read rotated 180
        degrees and the reading with the higher per-character log-confidence wins (catches upside-down lines)."""
        if not crops:
            return []
        if not self.flip:
            return self._run(crops)
        res = self._run(list(crops) + [cv2.rotate(c, cv2.ROTATE_180) for c in crops])
        n, out = len(crops), []
        for a, b in zip(res[:n], res[n:]):
            # compare on char_confidence: comparable across the two readings' different lengths
            use_b = b[0] and (not a[0] or math.log(max(b[2], 1e-12)) > math.log(max(a[2], 1e-12)) + self.flip_margin)
            out.append(b if use_b else a)
        return out

    def _run(self, crops):
        xs = [_prep(c) for c in crops]
        groups = defaultdict(list)
        for i, x in enumerate(xs):
            groups[int(math.ceil(x.shape[2] / self.bucket) * self.bucket)].append(i)
        out = [("", 0.0, 0.0)] * len(xs)
        for W, idx in groups.items():
            step = max(1, self.budget // (H * W))
            for s in range(0, len(idx), step):
                part = idx[s:s + step]
                n = 1 << (len(part) - 1).bit_length() if self.gpu else len(part)
                batch = np.zeros((n, 3, H, W), np.float32)
                for t, k in enumerate(part):
                    batch[t, :, :, : xs[k].shape[2]] = xs[k]
                p = self.sess.run(None, {self.input: batch})[0]
                ids, pr = p.argmax(-1), p.max(-1)
                for t, k in enumerate(part):
                    T = max(1, int(math.ceil(xs[k].shape[2] / W * p.shape[1])))  # skip timesteps over the padding
                    out[k] = self._decode(ids[t, :T], pr[t, :T])
        return out

    def _decode(self, ids, probs):
        """-> (text, product_confidence, char_confidence)

        `char_confidence` is the geometric mean of the per-character probabilities, computed in
        log space. Deriving it from the product afterwards does not work: the product underflows
        to exactly 0.0 as soon as one character is near-zero, and 0.0 ** (1/n) is still 0.0, so a
        line with one bad character becomes indistinguishable from a line that is entirely wrong.
        """
        txt, conf, prev = [], [], 0
        for c, q in zip(ids, probs):
            if c != prev and c != 0:
                txt.append(self.chars[c])
                conf.append(q)
            prev = c
        if not conf:
            return "", 0.0, 0.0
        lg = np.log(np.clip(np.asarray(conf, dtype=np.float64), 1e-12, 1.0))
        return "".join(txt), float(np.prod(conf)), float(np.exp(lg.mean()))
