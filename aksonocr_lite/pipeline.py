"""Page pipeline: (dewarp) -> page rotation -> layout + text lines -> tables -> reading-order merge -> Markdown."""
from __future__ import annotations

import re
import statistics
import time
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from . import schema
from .det import TextDetector, crop_quad
from .inputs import load_pages
from .layout import LayoutDetector
from .orient import LineOrientation, PageOrientation, rotate
from .rec import Recognizer
from .table import TableStructure, grid_to_html

HEADING_LABELS = {"doc_title", "paragraph_title"}
TABLE_LABELS = {"table"}
FIGURE_LABELS = {"image", "chart", "seal"}
FORMULA_LABELS = {"display_formula", "inline_formula", "algorithm"}  # read as OCR text (no formula model)


@dataclass
class Config:
    device: str = "auto"                 # auto | cpu | cuda: layout, detection, tables, dewarp
    small_device: str = "cpu"            # recognizer + orientation classifiers. These are tiny and see a new input
                                         # shape for almost every crop width; on CUDA each new shape costs ~50 ms of
                                         # setup, so CPU is several times faster even on a GPU machine.
    dpi: int = 150                       # PDF render resolution
    rotate: bool = True                  # page orientation correction (0/90/180/270)
    dewarp: bool = False                 # flatten curled/photographed pages (needs the [dewarp] extra)
    tables: bool = True                  # TATR table structure (off: tables are emitted as plain OCR text)
    table_min_score: float = 0.9         # layout confidence a table region needs to be parsed as a table
    table_ocr: bool = True               # re-detect text inside table/chart crops (tighter boxes per cell)
    det_limit_side_len: int = 1280
    det_thresh: float = 0.3
    det_box_thresh: float = 0.3
    det_unclip_ratio: float = 2.0
    table_det_unclip_ratio: float = 1.2
    layout_threshold: float = 0.1
    line_flip: bool = True               # also read every line rotated 180 degrees, keep the more confident one
    rec_budget: int = 3_000_000          # recognizer batch size in pixels
    timings: bool = True
    reading_order: str = "rows"          # rows: group blocks into rows by vertical overlap, read each left to
                                         # right. layout: trust the layout model's own `order` (the 1.0.0
                                         # behaviour). The layout head mis-sequences grids - on a 3x3 figure
                                         # grid it emitted column 1 -> column 3 -> column 2 on every row.
    row_overlap: float = 0.3             # y-overlap (fraction of the shorter block) to count as the same row
    keep_short_captions: bool = False    # emit a figure's text even when it is a single short line. Off keeps
                                         # 1.0.0 behaviour, which discards it silently.
    extra: dict = field(default_factory=dict)


class Converter:
    def __init__(self, config: Config | None = None, **kw):
        self.cfg = config or Config(**kw)
        c = self.cfg
        self.layout = LayoutDetector(c.device, threshold=c.layout_threshold)
        self.det = TextDetector(c.device, limit_side_len=c.det_limit_side_len, thresh=c.det_thresh,
                                box_thresh=c.det_box_thresh, unclip_ratio=c.det_unclip_ratio)
        self.line_ori = LineOrientation(c.small_device)
        self.rec = Recognizer(c.small_device, budget=c.rec_budget, flip=c.line_flip)
        self.page_ori = PageOrientation(c.small_device) if c.rotate else None
        self.tatr = TableStructure(c.device) if c.tables else None
        self.dewarper = None
        if c.dewarp:
            from .dewarp import Dewarper
            self.dewarper = Dewarper(c.device)

    # ---------------------------------------------------------------- public API
    def convert(self, src, max_pages: int | None = None, pages: list[int] | None = None) -> Iterator[schema.Page]:
        """Yield one Page per input page (path, bytes, PIL image or BGR numpy array)."""
        for i, img in enumerate(load_pages(src, dpi=self.cfg.dpi, max_pages=max_pages, pages=pages)):
            yield self.page(img, index=i)

    def ocr(self, img: np.ndarray) -> list[schema.Line]:
        """Text lines only (no layout), in top-to-bottom order."""
        return [schema.Line(**l) for l in self._lines(img, self.cfg.det_unclip_ratio)]

    def page(self, img: np.ndarray, index: int = 0) -> schema.Page:
        t, T = time.perf_counter(), {}

        def tick(name):
            nonlocal t
            now = time.perf_counter()
            T[name] = round(now - t, 4)
            t = now

        dewarped = False
        if self.dewarper is not None:
            img, dewarped = self.dewarper(img)
            tick("dewarp")
        angle = 0
        if self.page_ori is not None:
            angle = self.page_ori(img)
            img = rotate(img, angle)
            tick("rotate")
        h, w = img.shape[:2]
        lines = self._lines(img, self.cfg.det_unclip_ratio)
        tick("ocr")
        blocks = [dict(label=b.label, score=b.score, bbox=b.bbox, order=b.order) for b in self.layout(img)]
        blocks = [b for b in blocks if b["label"] not in TABLE_LABELS or b["score"] >= self.cfg.table_min_score]
        for b in blocks:
            if b["label"] in FORMULA_LABELS:
                b["label"] = "text"
        tick("layout")

        if self.cfg.table_ocr:
            lines = self._region_ocr(img, lines, [b for b in blocks if b["label"] in TABLE_LABELS | {"chart"}])
            tick("table_ocr")
        lines = filter_bad_lines(lines)

        tables = [b for b in blocks if b["label"] in TABLE_LABELS] if self.tatr is not None else []
        rest = [b for b in blocks if b not in tables]
        table_lines = {id(b): [] for b in tables}
        free = []
        for i, l in enumerate(lines):
            c = _center(l["bbox"])
            hits = [b for b in tables if _inside(c, b["bbox"])]
            if hits:
                table_lines[id(max(hits, key=lambda b: _overlap(l["bbox"], b["bbox"])))].append(i)
            else:
                free.append(i)
        for b in tables:
            b["html"] = self._table_html(img, b, [lines[i] for i in table_lines[id(b)]])
            b["_lines"] = table_lines[id(b)]
        tick("tables")

        if not rest and free:
            rest = [dict(label="text", score=1.0, bbox=[0.0, 0.0, float(w), float(h)], order=0.0)]
            blocks.append(rest[0])
        assign_lines(lines, free, rest)
        for b in rest:
            b["_lines"].sort(key=lambda i: (round(lines[i]["bbox"][1] / 8), lines[i]["bbox"][0]))
        promote_text_rich_figures(rest, lines, w * h)

        out_blocks, md = [], []
        seq = (row_major(blocks, self.cfg.row_overlap) if self.cfg.reading_order == "rows"
               else sorted(blocks, key=lambda b: b["order"]))
        for k, b in enumerate(seq):
            idx = b.get("_lines", [])
            text = "\n".join(lines[i]["text"] for i in (sorted(idx, key=lambda i: (round(lines[i]["bbox"][1] / 8), lines[i]["bbox"][0]))) if lines[i]["text"])
            md_part = block_markdown(b, text, len(idx), self.cfg.keep_short_captions)
            if md_part:
                md.append(md_part)
            out_blocks.append(schema.Block(id=k, label=b["label"], bbox=[round(v, 1) for v in b["bbox"]],
                                           score=round(b["score"], 4), order=b["order"], text=text,
                                           html=b.get("html") or None, line_ids=list(idx)))
        tick("merge")
        return schema.Page(index=index, width=w, height=h, rotation=angle, dewarped=dewarped,
                           markdown=format_html_tables("\n\n".join(md)), lines=[schema.Line(**l) for l in lines],
                           blocks=out_blocks, timings=T if self.cfg.timings else {})

    # ---------------------------------------------------------------- stages
    def _lines(self, img, unclip_ratio):
        quads = self.det(img, unclip_ratio=unclip_ratio)
        crops = [crop_quad(img, q) for q in quads]
        if crops:
            for k, up in enumerate(self.line_ori(crops)):
                if up:
                    crops[k] = np.ascontiguousarray(crops[k][::-1, ::-1])
        out = []
        for q, (text, conf, cconf) in zip(quads, self.rec(crops)):
            if not text:
                continue
            xs, ys = q[:, 0], q[:, 1]
            out.append(dict(text=text, confidence=round(conf, 4), char_confidence=round(cconf, 4),
                            poly=[[float(x), float(y)] for x, y in q],
                            bbox=[float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]))
        return out

    def _region_ocr(self, img, lines, regions):
        """Re-run detection + recognition inside each table/chart crop with a tighter unclip ratio, so text in
        neighbouring cells is not merged into one page-level box; the crop's lines replace the page lines whose
        centre falls in that region."""
        if not regions:
            return lines
        H, W = img.shape[:2]
        owner = {}
        for i, l in enumerate(lines):
            c = _center(l["bbox"])
            hits = [r for r in regions if _inside(c, r["bbox"])]
            if hits:
                owner[i] = id(max(hits, key=lambda r: _overlap(l["bbox"], r["bbox"])))
        replaced, extra = set(), []
        for r in regions:
            x0, y0, x1, y1 = [max(0, int(v)) for v in r["bbox"]]
            x1, y1 = min(W, x1), min(H, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            got = self._lines(np.ascontiguousarray(img[y0:y1, x0:x1]), self.cfg.table_det_unclip_ratio)
            if not got:
                continue
            for l in got:
                l["bbox"] = [l["bbox"][0] + x0, l["bbox"][1] + y0, l["bbox"][2] + x0, l["bbox"][3] + y0]
                l["poly"] = [[p[0] + x0, p[1] + y0] for p in l["poly"]]
            replaced.add(id(r))
            extra += got
        return [l for i, l in enumerate(lines) if owner.get(i) not in replaced] + extra

    def _table_html(self, img, block, tlines):
        H, W = img.shape[:2]
        x0, y0, x1, y1 = [max(0, int(v)) for v in block["bbox"]]
        x1, y1 = min(W, x1), min(H, y1)
        if x1 <= x0 or y1 <= y0:
            return ""
        cells, _ = self.tatr(np.ascontiguousarray(img[y0:y1, x0:x1]))
        crop_lines = [dict(text=l["text"], bbox=[l["bbox"][0] - x0, l["bbox"][1] - y0, l["bbox"][2] - x0, l["bbox"][3] - y0])
                      for l in tlines]
        return grid_to_html(cells, crop_lines)


# -------------------------------------------------------------------- merge helpers
def _center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def _inside(pt, b, pad=2):
    return b[0] - pad <= pt[0] <= b[2] + pad and b[1] - pad <= pt[1] <= b[3] + pad


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _overlap(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    return 0.0 if x1 <= x0 or y1 <= y0 else (x1 - x0) * (y1 - y0)


def assign_lines(lines, idxs, regions):
    """Each line goes to the region containing its centre (largest overlap, non-figure, smallest region first);
    lines no region contains go to the nearest region, so no detected text is dropped."""
    for r in regions:
        r.setdefault("_lines", [])
    for i in idxs:
        lb = lines[i]["bbox"]
        c = _center(lb)
        hits = [r for r in regions if _inside(c, r["bbox"])]
        if hits:
            la = max(1.0, _area(lb))
            best = max(hits, key=lambda r: (_overlap(lb, r["bbox"]) / la, r["label"] not in FIGURE_LABELS, -_area(r["bbox"])))
        elif regions:
            def dist(r):
                b = r["bbox"]
                dx, dy = max(b[0] - c[0], 0.0, c[0] - b[2]), max(b[1] - c[1], 0.0, c[1] - b[3])
                return (dx * dx + dy * dy) ** 0.5
            best = min(regions, key=dist)
        else:
            continue
        best["_lines"].append(i)


def promote_text_rich_figures(blocks, lines, page_area):
    """Slides and screenshots are often labelled `image`: keep their text by relabelling text-rich ones."""
    for b in blocks:
        if b["label"] not in FIGURE_LABELS or b["label"] == "chart":
            continue
        idx = b.get("_lines", [])
        text = "\n".join(lines[i]["text"] for i in idx).strip()
        if len(idx) >= 3 or len(text) >= 50 or (_area(b["bbox"]) >= 0.18 * page_area and idx and len(text) >= 15):
            b["label"] = "text"


def row_major(blocks, min_overlap=0.3):
    """Reading order: group blocks into rows, then read each row left to right.

    Two blocks share a row when their y ranges actually overlap by `min_overlap` of the shorter
    one -- not when their centres are close. Two stacked footnote lines have centres only ~60 px
    apart but no overlap at all; a centre-distance rule pulls them into one row and then orders
    them by a 1 px difference in x (measured: 5.85% -> 12.12% CER on a landscape table page).

    Replaces trusting the layout model's `order`, which mis-sequences multi-column rows
    (measured on a 3x3 figure grid: 26.21% -> 18.42% CER, with no change on text or table pages).
    """
    if not blocks:
        return []
    rows: list[list[dict]] = []
    for b in sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0])):
        y0, y1 = b["bbox"][1], b["bbox"][3]
        for r in rows:
            ry0 = min(x["bbox"][1] for x in r)
            ry1 = max(x["bbox"][3] for x in r)
            ov = min(y1, ry1) - max(y0, ry0)
            if ov > 0 and ov >= min_overlap * max(1.0, min(y1 - y0, ry1 - ry0)):
                r.append(b)
                break
        else:
            rows.append([b])
    rows.sort(key=lambda r: min(x["bbox"][1] for x in r))
    return [b for r in rows for b in sorted(r, key=lambda b: b["bbox"][0])]


def block_markdown(b, text, n_lines, keep_short_captions=False):
    label = b["label"]
    if label in TABLE_LABELS:
        return b.get("html") or text
    if label in FIGURE_LABELS:
        marker = "<figure>\n</figure>"
        if text and (keep_short_captions or n_lines >= 2 or len(text) >= 40 or b["score"] < 0.85):
            return marker + "\n" + text
        return marker
    if not text:
        return ""
    if label in HEADING_LABELS:
        return "## " + text
    return text


_TABLE_TAG_RE = re.compile(r"(<table\b[^>]*>|</table>|<tr\b[^>]*>|</tr>|<t[dh]\b[^>]*>.*?</t[dh]>)\s*", re.I | re.S)


def format_html_tables(md):
    if "<table" in md.lower():
        md = _TABLE_TAG_RE.sub(lambda m: m.group(1).strip() + "\n", md)
    return "\n".join(l for l in md.splitlines() if l.strip())


# -------------------------------------------------------------------- detection filter
def filter_bad_lines(lines, overlap_min=0.5, band=0.45):
    """Drop lines that are both tiny (vs the page median) and nested in the top/bottom band of a bigger line:
    the detector boxing a Thai vowel or tone mark separately."""
    if not lines:
        return lines
    boxes = [l["bbox"] for l in lines]
    med_h = statistics.median(max(1.0, b[3] - b[1]) for b in boxes)
    med_a = statistics.median(_area(b) for b in boxes)
    min_h, min_a = max(10, med_h * 0.32), max(36.0, med_a * 0.08)
    small = [(b[3] - b[1]) < min_h or _area(b) < min_a for b in boxes]
    drop = set()
    for i, inner in enumerate(boxes):
        if not small[i]:
            continue
        ia, text = _area(inner), (lines[i]["text"] or "").strip()
        for j, outer in enumerate(boxes):
            if i == j or ia >= _area(outer):
                continue
            ov = _overlap(inner, outer)
            if ia <= 0 or ov <= 0:
                continue
            nested = False
            if ov / ia >= overlap_min:
                bh = max(1.0, outer[3] - outer[1]) * band
                top = _overlap(inner, [outer[0], outer[1], outer[2], outer[1] + bh])
                bot = _overlap(inner, [outer[0], outer[3] - bh, outer[2], outer[3]])
                nested = top / ov >= 0.5 or bot / ov >= 0.5
            if not nested:
                yo = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1])) / max(1.0, min(inner[3] - inner[1], outer[3] - outer[1]))
                nested = yo >= 0.55 and ov / ia >= max(0.35, overlap_min - 0.15) and ia <= _area(outer) * 0.45 and len(text) <= 3
            if nested:
                drop.add(i)
                break
    return [l for i, l in enumerate(lines) if i not in drop]
