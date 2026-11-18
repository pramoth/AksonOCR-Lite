<p align="center">
  <img src="assets/AksonOCR-logo.svg" alt="AksonOCR" width="420">
</p>

<h1 align="center">AksonOCR-Lite</h1>

<p align="center">
  <a href="https://aksonocr.com/lite-demo"><img src="https://img.shields.io/badge/%E2%96%B6%20Try%20the%20live%20demo-aksonocr.com%2Flite--demo-2563eb?style=for-the-badge" alt="Try the live demo" height="36"></a>
</p>

<p align="center">
  <a href="https://github.com/Remtir/AksonOCR-Lite/actions/workflows/ci.yml"><img src="https://github.com/Remtir/AksonOCR-Lite/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue?style=flat-square" alt="License: Apache-2.0"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/ONNX%20Runtime-CPU%20%7C%20GPU-005CED?style=flat-square&logo=onnx&logoColor=white" alt="ONNX Runtime: CPU or GPU">
  <a href="https://huggingface.co/Remtir/AksonOCR-Lite"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20weights-Remtir%2FAksonOCR--Lite-FFD21E?style=flat-square" alt="Weights on Hugging Face"></a>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/language-Thai%20%2B%20English-6b7079?style=flat-square" alt="Thai + English">
  <a href="BENCHMARKS.md"><img src="https://img.shields.io/badge/MDPBench--th-85.6%20with%20dewarp-2ea44f?style=flat-square" alt="MDPBench-th 85.6 with dewarp"></a>
  <a href="BENCHMARKS.md"><img src="https://img.shields.io/badge/ThaiOCRBench%20DP-0.640-2ea44f?style=flat-square" alt="ThaiOCRBench Document parsing 0.640"></a>
  <a href="BENCHMARKS.md"><img src="https://img.shields.io/badge/speed-1.5%20s%2Fpage%20on%20CPU-6b7079?style=flat-square" alt="1.5 s per page on CPU"></a>
  <img src="https://img.shields.io/badge/weights-330%20MB-6b7079?style=flat-square" alt="330 MB of weights">
</p>

Fast, open-source **Thai document OCR** with a grapheme-cluster CTC recognizer. Turns PDFs and images into
Markdown, plain text or JSON with layout blocks, reading order, text-line boxes and tables. Runs on
**ONNX Runtime**, on CPU or GPU, with no PyTorch or Paddle install.

> [!TIP]
> **[Try it in your browser →](https://aksonocr.com/lite-demo)** Upload a page image or PDF (up to 7 MB) and see the
> Markdown, layout and tables. No install.

> **81M params, runs on CPU:** ~3× faster per page than 2B-param OCR VLMs on a GPU, and more accurate than VLMs
> on average across MDPBench, ThaiOCRBench and SEA-DocBench ([benchmarks](BENCHMARKS.md)).

```bash
pip install git+https://github.com/Remtir/AksonOCR-Lite
aksonocr invoice.pdf > invoice.md
```

<sub>The PyPI package (`pip install aksonocr-lite`) is coming soon; until then, install from GitHub as above.</sub>

- **Thai first:** the text recognizer is trained for Thai (with English), predicting whole grapheme clusters
  so stacked vowels and tone marks come out in the right order.
- **Document structure:** 25 layout classes (titles, paragraphs, tables, figures, headers/footers, ...) and
  reading order from PP-DocLayoutV3; tables become HTML with row/column spans.
- **Photos too:** page-orientation correction, upside-down line detection, and optional page dewarping.
- **Light:** about 330 MB of weights by default (≈620 MB with dewarping), fetched from the Hugging Face Hub on first run.

<p align="center">
  <img src="assets/viz/demo1.gif" alt="Layout blocks and text lines on a Thai government letter" width="32%">
  <img src="assets/viz/sample2.gif" alt="Layout blocks and text lines on a handwritten page photo" width="32%">
  <img src="assets/viz/sample1.gif" alt="Layout blocks and text lines on a financial statement" width="32%">
</p>
<p align="center"><sub>Layout blocks in reading order, table cells (TATR grid), then detected text lines. Made with <code>tools/visualize_gif.py</code>.</sub></p>

## Benchmarks

**MDPBench-th 85.6** with `--dewarp` (79.5 default) · **ThaiOCRBench DP 0.640** · **SEA-DocBench-th text NED 0.058**
· **1.5 s per page on CPU only**.

<p align="center">
  <img src="assets/charts/accuracy_light.png" alt="Accuracy of AksonOCR-Lite vs 12 other Thai OCR systems: average of MDPBench-th, ThaiOCRBench Document parsing and SEA-DocBench-th" width="100%">
</p>

<p align="center">
  <img src="assets/charts/speed_light.png" alt="Speed of AksonOCR-Lite vs three VLMs, average over three benchmarks: seconds per page one at a time, pages per second batched" width="100%">
</p>

Averages over MDPBench-th, ThaiOCRBench Document parsing and SEA-DocBench-th.
**[Full benchmark →](BENCHMARKS.md)** per-benchmark charts, default vs `--dewarp`, speed tables, text recognition
only.

## Requirements

| | |
|---|---|
| **Python** | 3.10 or newer (CI tests 3.10 and 3.13) |
| **OS** | Linux, macOS (Apple Silicon). Windows is untested so far. |
| **Hardware** | Any x86-64 or ARM64 CPU. No GPU needed; NVIDIA GPU optional (see [GPU](#gpu)). |
| **RAM** | about 1.3 GB peak for one page, about 2.8 GB with `--dewarp` |
| **Disk** | about 330 MB of model weights, plus 290 MB for `--dewarp`, in the Hugging Face cache (`~/.cache/huggingface`) |
| **Network** | only on first run, to download the weights. Use [`AKSONOCR_WEIGHTS`](#offline--custom-weights) for fully offline machines. |

No PyTorch, Paddle or system packages are needed. `pip` installs everything else: `onnxruntime`, `numpy`,
`opencv-python-headless`, `pillow`, `pypdfium2` (PDF rendering), `pyclipper`, `shapely` and `huggingface_hub`.

For the GPU build (`onnxruntime-gpu`): an NVIDIA driver with CUDA 12 and cuDNN 9, the versions current
`onnxruntime-gpu` wheels are built against.

## Usage

### Command line

```bash
aksonocr page.jpg                        # Markdown to stdout
aksonocr report.pdf --out out/           # out/report.md (every page of the PDF)
aksonocr scan.png -f json -o out/        # boxes, blocks, lines, confidences
aksonocr photo.jpg --dewarp              # flatten a photographed / curled page
aksonocr doc.pdf --device cpu --dpi 200 --max-pages 10
```

### Python

```python
import aksonocr_lite

conv = aksonocr_lite.load()                 # device="auto" | "cpu" | "cuda", dewarp=False, tables=True, ...
for page in conv.convert("report.pdf"):     # path, bytes, PIL.Image or BGR numpy array
    print(page.markdown)
    for block in page.blocks:               # reading order
        print(block.label, block.bbox, block.text or block.html)
    for line in page.lines:
        print(line.text, line.confidence, line.poly)

lines = conv.ocr(bgr_image)                  # text lines only, no layout
```

`Page` fields: `markdown`, `text`, `blocks` (`label`, `bbox`, `score`, `order`, `text`, `html`, `line_ids`),
`lines` (`text`, `confidence`, `bbox`, `poly`), `rotation`, `dewarped`, `width`, `height`, `timings`.
All coordinates are in the pixel space of the (rotated, dewarped) page image.

### GPU

The default install runs on CPU, on Linux and macOS (Apple Silicon). For CUDA, swap ONNX Runtime for its GPU build:

```bash
pip uninstall -y onnxruntime && pip install onnxruntime-gpu
```

`device="auto"` then picks CUDA. Inputs have variable shapes, so the CPU path is often just as fast for
single pages; the GPU pays off mostly with dewarping and large batches.

### Language-model rescoring (optional, off by default)

The greedy CTC decode keeps only the best class per frame. On degraded scans the right Thai cluster is often
the runner-up -- the confusions listed under [Limitations](#limitations) are exactly that -- and a language
model can tell which candidate makes a word. `lm_path` re-reads only the uncertain positions of each line:

```bash
# build a model from text in your own domain (plain UTF-8, paragraphs separated by blank lines)
python -m aksonocr_lite.lm build corpus.txt mymodel.json.gz
pip install pythainlp                       # the word list used as a guard
```

```python
conv = aksonocr_lite.load(lm_path="mymodel.json.gz")
```

Only Thai-for-Thai substitutions and Thai insertions are ever proposed, and nothing is ever deleted, so
digits, Latin letters and punctuation come out exactly as the greedy decode read them -- amounts, codes and
URLs are untouched. A change is kept only when it does not add characters that no dictionary word covers;
when a real word would be swapped for another real word, the combined score must also improve by
`lm_margin`. `lm_gate` (default 0.98) leaves confident clusters alone.

Measured on 5,566 lines of scanned Thai government reports the recognizer had not seen, against a
born-digital text layer as the reference: CER 1.248% -> 0.759%, whole lines correct 88.12% -> 90.68%, with
every digit of every currency amount unchanged. Rescoring costs 3-18% more time per page. A model built from
roughly a million characters of in-domain text is enough; the quality depends on that text matching your
documents, so the gain on other material will differ.

### Offline / custom weights

Set `AKSONOCR_WEIGHTS=/path/to/dir` to load models from a local folder instead of the Hub. It must contain
`layout.onnx det.onnx doc_ori.onnx textline_ori.onnx rec.onnx rec_charset.json tatr.onnx`, plus
`dewarp_mask.onnx dewarp_grid.onnx` for `--dewarp`.

## How it works

```
page image ─┬─ (dewarp: MonkeyOCRv2 preprocessor)            optional
            ├─ page orientation: PP-LCNet doc_ori             0/90/180/270
            ├─ layout + reading order: PP-DocLayoutV3        ┐
            ├─ text lines: PP-OCRv6_medium_det (DB)           │ run on the same page
            │    └─ line orientation (PP-LCNet textline_ori)  │
            │    └─ recognition: AksonOCR Thai CTC (+180° re-read, keeps the more confident reading)
            ├─ tables: re-detect text in the crop → TATR v1.1 grid → fill cells with OCR lines → HTML
            └─ merge: each line → the layout block holding its centre (nearest block as fallback),
                      blocks in reading order → Markdown
```

### Thai recognizer: grapheme-cluster CTC

Thai stacks vowels and tone marks on the consonant, so one glyph can be 2–3 code points (ผู้ = ผ + ู + ้).
Instead of emitting them one by one, the recognizer predicts the **whole cluster as one token**.

<p align="center">
  <img src="assets/viz/cluster_tokens.gif" alt="Step by step: the recognizer's real CTC output on a Thai line, spelled as character tokens (22 symbols, stacked marks as 2–3 symbols in one spot) and as cluster tokens (15 symbols, one per glyph column)" width="100%">
</p>
<p align="center"><sub>Real CTC output of the recognizer on a photographed line; made with <code>tools/cluster_gif.py</code>.</sub></p>

- **Correct order, always:** marks inside a cluster always come out in Unicode order, never swapped.
- **No impossible output:** only real clusters are in the vocabulary (2,149 of 2,455 classes).
- **Shorter sequence:** one token per glyph (15 vs 22 on the line above).
- **Tiny and fast:** 1.5M parameters, 88.6% exact match at 8 ms per line on one CPU thread
  ([comparison](BENCHMARKS.md#text-recognition-only)).

Details worth knowing:

- Text in tables is read by the Thai recognizer, not by the table model: TATR only provides the grid.
- The detector often boxes a lone Thai vowel or tone mark; such fragments are dropped before recognition.
- Lines are batched only with lines of nearly the same width. Padding changes the recognizer's
  attention and cost about 3 points of exact-match accuracy when widths were mixed freely.

## Limitations

- **Photographed pages are harder than digital ones.** On MDPBench-th the default pipeline scores 89.5 on digital
  pages but 76.2 on photos (curl, shadow, glare, perspective). `--dewarp` brings photos to 84.3 at about twice the
  weights and extra time per page; very dark, folded or blurred photos still lose text.
- **Handwriting is weak.** The recognizer was trained for printed Thai and English; handwritten notes and
  filled-in forms come out with many errors.
- **No formula or picture understanding.** Math is read as plain text lines (no LaTeX). Figures and charts
  become `<figure>` blocks holding only the text printed in them, with no description of the picture.
- **Dense multi-column layouts** such as newspapers often get the reading order or column split wrong.
- **Thai and English only.** Other scripts and vertical text are not supported.

## Models and licenses

| Stage | Model | License |
|---|---|---|
| Layout + reading order | [PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_onnx) | Apache-2.0 |
| Text detection | [PP-OCRv6_medium_det](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_onnx) | Apache-2.0 |
| Page / line orientation | [PP-LCNet_x1_0_doc_ori](https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_doc_ori_onnx), [textline_ori](https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_textline_ori_onnx) | Apache-2.0 |
| Text recognition | AksonOCR Thai recognizer (fine-tuned PP-OCRv6_tiny_rec), [model card](MODEL_CARD.md) | Apache-2.0 |
| Table structure | [Table Transformer v1.1-all](https://huggingface.co/microsoft/table-transformer-structure-recognition-v1.1-all) (Microsoft) | MIT |
| Dewarping (optional) | [MonkeyOCRv2](https://github.com/Yuliang-Liu/MonkeyOCRv2) document preprocessor | Apache-2.0 |

All weights download from one Hugging Face repo, [Remtir/AksonOCR-Lite](https://huggingface.co/Remtir/AksonOCR-Lite);
the PaddlePaddle models there are unmodified copies of the official ONNX exports linked above.
The code is Apache-2.0 ([LICENSE](LICENSE)); third-party attributions are in [NOTICE](NOTICE). Every model
is permissively licensed, so AksonOCR-Lite can be used commercially. PDFs are rendered with pypdfium2
(Apache-2.0/BSD), not PyMuPDF (AGPL).

## Development

```bash
pip install -e .[dev]
pytest -m "not e2e"          # unit tests, no weights needed
pytest                       # + end-to-end test (downloads weights)
```

`tools/` holds the scripts that produced the ONNX files (they need Paddle/PyTorch and are not part of the
package): `export_tatr.py`, `export_dewarp.py`.

## Acknowledgements

AksonOCR-Lite stands on open work from many teams:

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) (PaddlePaddle): PP-DocLayoutV3, PP-OCRv6 detection and
  the tiny recognizer the Thai model was fine-tuned from, PP-LCNet orientation models, and the DB / CTC
  post-processing this code ports.
- [Table Transformer](https://github.com/microsoft/table-transformer) (Microsoft) for table structure.
- [MonkeyOCRv2](https://github.com/Yuliang-Liu/MonkeyOCRv2) for the page dewarping preprocessor.
- [ONNX Runtime](https://onnxruntime.ai) and [pypdfium2](https://github.com/pypdfium2-team/pypdfium2), which make
  a framework-free install possible.
- The benchmark authors: [MDPBench](https://github.com/Yuliang-Liu/MultimodalOCR) (with the OmniDocBench
  evaluator), [ThaiOCRBench](https://github.com/scb-10x/ThaiOCRBench) and
  [SEA-Vision / SEA-DocBench](https://huggingface.co/datasets/xingranzhao/SEA-Vision).

## Citation

```bibtex
@software{aksonocr_lite,
  author = {Jaknamon, Teetouch},
  title  = {AksonOCR-Lite: Thai Document OCR with Grapheme-Cluster CTC Recognition},
  version = {1.0.0},
  year   = {2026},
  url    = {https://github.com/Remtir/AksonOCR-Lite}
}
```
