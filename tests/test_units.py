"""Pure-python unit tests (no model weights needed)."""
import numpy as np

from aksonocr_lite.det import drop_mark_fragments, sorted_boxes
from aksonocr_lite.layout import Block, fill_missing_order
from aksonocr_lite.orient import rotate
from aksonocr_lite.pipeline import assign_lines, block_markdown, filter_bad_lines, format_html_tables, row_major
from aksonocr_lite.rec import Recognizer
from aksonocr_lite.table import grid_cells, grid_to_html


def q(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)


def test_sorted_boxes_row_then_column():
    out = sorted_boxes([q(200, 12, 300, 30), q(0, 10, 100, 30), q(0, 60, 100, 80)])
    assert [int(b[0][0]) for b in out] == [0, 200, 0]


def test_drop_mark_fragment_above_line():
    lines = [q(0, 100, 400, 130), q(0, 200, 400, 230), q(50, 92, 62, 100)]  # tiny box on top edge of line 1
    assert len(drop_mark_fragments(lines)) == 2


def test_fill_missing_order_no_ties():
    bs = [Block("text", 1, [0, 300, 10, 310], 1), Block("table", 1, [0, 10, 10, 20], None),
          Block("doc_title", 1, [0, 0, 10, 5], None)]
    fill_missing_order(bs)
    assert len({b.order for b in bs}) == 3
    assert sorted(bs, key=lambda b: b.order)[0].label == "doc_title"


def test_rotate_matches_pil_ccw():
    a = np.arange(6, dtype=np.uint8).reshape(2, 3, 1).repeat(3, 2)
    assert rotate(a, 90).shape[:2] == (3, 2)
    assert rotate(a, 90)[0, 0, 0] == a[0, 2, 0]  # top-right corner moves to top-left


def test_grid_cells_and_html_with_span():
    objs = [{"label": "table row", "score": .9, "bbox": [0, 0, 200, 20]},
            {"label": "table row", "score": .9, "bbox": [0, 20, 200, 40]},
            {"label": "table column", "score": .9, "bbox": [0, 0, 100, 40]},
            {"label": "table column", "score": .9, "bbox": [100, 0, 200, 40]},
            {"label": "table column header", "score": .9, "bbox": [0, 0, 200, 20]},
            {"label": "table spanning cell", "score": .9, "bbox": [0, 0, 200, 20]}]
    cells, score = grid_cells(objs)
    assert len(cells) == 3 and score > 0.8
    html = grid_to_html(cells, [{"text": "หัวตาราง", "bbox": [50, 2, 150, 18]},
                                {"text": "ก", "bbox": [10, 22, 40, 38]}, {"text": "1 < 2", "bbox": [110, 22, 150, 38]}])
    assert html == '<table><tr><th colspan="2">หัวตาราง</th></tr><tr><td>ก</td><td>1 &lt; 2</td></tr></table>'


def test_assign_lines_nearest_fallback():
    lines = [{"bbox": [10, 10, 50, 20]}, {"bbox": [500, 500, 520, 510]}]
    regions = [{"label": "text", "bbox": [0, 0, 100, 100]}]
    assign_lines(lines, [0, 1], regions)
    assert regions[0]["_lines"] == [0, 1]


def test_filter_drops_tiny_nested_mark():
    lines = [{"text": f"บรรทัด {i}", "bbox": [0, 40 * i, 400, 40 * i + 30]} for i in range(5)]
    lines.append({"text": "่", "bbox": [30, 1, 36, 8]})
    assert len(filter_bad_lines(lines)) == 5


def test_markdown_rendering():
    assert block_markdown({"label": "paragraph_title", "score": 1}, "หัวข้อ", 1) == "## หัวข้อ"
    assert block_markdown({"label": "image", "score": .95}, "x", 1) == "<figure>\n</figure>"
    assert format_html_tables("<table><tr><td>a</td></tr></table>") == "<table>\n<tr>\n<td>a</td>\n</tr>\n</table>"


def b(label, x0, y0, x1, y1):
    return {"label": label, "score": 1.0, "bbox": [x0, y0, x1, y1]}


def test_row_major_reads_a_grid_row_left_to_right():
    """The layout head sequences a 3-column figure grid as column 1 -> column 3 -> column 2."""
    row = [b("vision_footnote", 369, 1105, 585, 1161),
           b("vision_footnote", 1118, 1108, 1305, 1165),
           b("vision_footnote", 718, 1114, 951, 1173)]
    assert [int(x["bbox"][0]) for x in row_major(row)] == [369, 718, 1118]


def test_row_major_keeps_vertically_stacked_blocks_in_order():
    """Two footnote lines stacked under a table share an x but must not share a row.

    Their centres are only ~60px apart, so grouping rows by centre distance pulls them
    together and then orders them by a 1px difference in x, reversing the two lines.
    """
    stacked = [b("vision_footnote", 194, 1157, 567, 1209),
               b("vision_footnote", 193, 1222, 748, 1268)]
    assert [int(x["bbox"][1]) for x in row_major(stacked)] == [1157, 1222]


def test_row_major_orders_rows_top_to_bottom():
    out = row_major([b("text", 0, 200, 100, 230), b("text", 400, 10, 500, 40),
                     b("text", 0, 10, 100, 40)])
    assert [(int(x["bbox"][1]), int(x["bbox"][0])) for x in out] == [(10, 0), (10, 400), (200, 0)]


class _Rec:
    """Just enough of Recognizer for the pure CTC decode."""
    chars = ["", "ก", "ข", "ค"]
    _decode = Recognizer._decode


def test_char_confidence_is_comparable_across_lengths():
    """`confidence` is the product, so a long clean line scores below a short poor one."""
    long_clean = _Rec()._decode([1, 2, 1, 2] * 8, [0.97] * 32)
    short_poor = _Rec()._decode([1, 2], [0.5, 0.5])
    assert long_clean[1] < short_poor[1] + 0.2      # products say they are comparable
    assert long_clean[2] > 0.95 > short_poor[2]     # char confidence separates them


def test_char_confidence_survives_an_underflowing_product():
    text, prod, cconf = _Rec()._decode([1, 2] * 5, [0.99] * 9 + [0.0])
    assert len(text) == 10 and prod == 0.0          # one dead character zeroes the product
    assert cconf > 0.0                              # the log-space mean still ranks the line


def test_block_markdown_can_keep_a_short_caption():
    figure = {"label": "image", "score": .95}
    assert block_markdown(figure, "ข้าวแดงดอย", 1) == "<figure>\n</figure>"
    assert block_markdown(figure, "ข้าวแดงดอย", 1, True) == "<figure>\n</figure>\nข้าวแดงดอย"
