"""Output types."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Line:
    text: str
    confidence: float            # product of per-character probabilities: falls with line length
    bbox: list[float]            # x0, y0, x1, y1
    poly: list[list[float]]      # 4 points, clockwise from top-left
    char_confidence: float = 0.0
    """Geometric mean of the per-character probabilities -- threshold on this, not `confidence`.

    `confidence` is their product, so it drops with length even when quality does not: measured on
    scanned Thai reports the median is 0.97 at 3-5 characters but 0.54 at 21-40, while this stays
    ~0.99 in every length bucket. It also underflows to exactly 0.0 when one character is
    near-zero, which is why this is computed in log space at decode time rather than derived here.
    """


@dataclass
class Block:
    id: int
    label: str                   # PP-DocLayoutV3 label (text, paragraph_title, table, image, ...)
    bbox: list[float]
    score: float
    order: float
    text: str = ""               # OCR text of the block (newline between lines)
    html: str | None = None      # tables: <table> HTML
    line_ids: list[int] = field(default_factory=list)


@dataclass
class Page:
    index: int
    width: int
    height: int
    rotation: int                # counter-clockwise rotation applied before OCR (0/90/180/270)
    dewarped: bool
    markdown: str
    lines: list[Line]
    blocks: list[Block]
    timings: dict

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks if b.text and b.label not in ("image", "chart", "seal"))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["text"] = self.text
        return d
