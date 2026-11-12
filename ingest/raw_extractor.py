"""Step 2: extract raw text with positions and fonts from every page.

Turns PyMuPDF's nested dicts into typed models, so later steps never touch
PyMuPDF output directly. No cleanup happens here: headers, footers,
hyphenation and Unicode are step 3.

Run from the project root:
    python -m ingest.raw_extractor
"""

import sys
import time
from collections.abc import Sequence

import pymupdf
from loguru import logger

from ingest.errors import ExtractionError, IngestionError
from ingest.loader import open_pdf
from ingest.sources import PAPERS
from schemas.ingestion import BBox, PaperSource, RawBlock, RawDocument, RawLine, RawPage, RawSpan


# Kept: whitespace as written, and only text inside the visible page area.
# Left out on purpose:
#   TEXT_PRESERVE_LIGATURES      -> "ﬁ" comes out as "fi", matching what people type
#   TEXT_PRESERVE_IMAGES         -> image bytes are never loaded; only text is needed
#   TEXT_CID_FOR_UNKNOWN_UNICODE -> unmappable glyphs become U+FFFD, which can be
#                                   counted, instead of wrong-looking letters
#   TEXT_DEHYPHENATE             -> joining split words is decided in step 3
TEXT_FLAGS = pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_MEDIABOX_CLIP

# Hundredths of a point are far below anything visible.
_PRECISION = 2

_TEXT_BLOCK = 0
_UNMAPPED_CHAR = "\ufffd"


def _bbox(values: Sequence[float]) -> BBox:
    x0, y0, x1, y1 = (round(v, _PRECISION) for v in values)
    return (x0, y0, x1, y1)


def _span(span: dict) -> RawSpan | None:
    if not span["text"]:
        return None

    return RawSpan(
        text=span["text"],
        bbox=_bbox(span["bbox"]),
        font=span["font"],
        size=round(span["size"], _PRECISION),
        flags=span["flags"],
        color=span["color"],
    )


def _line(line: dict) -> RawLine | None:
    spans = [s for s in map(_span, line["spans"]) if s is not None]

    # Lines that are only whitespace carry no content.
    if not spans or not "".join(s.text for s in spans).strip():
        return None

    dx, dy = line["dir"]
    return RawLine(
        bbox=_bbox(line["bbox"]),
        direction=(round(dx, 3), round(dy, 3)),
        spans=spans,
    )


def extract_page(page: pymupdf.Page) -> RawPage:
    data = page.get_text("dict", flags=TEXT_FLAGS)

    blocks: list[RawBlock] = []
    for block in data["blocks"]:
        if block["type"] != _TEXT_BLOCK:
            continue

        lines = [line for line in map(_line, block["lines"]) if line is not None]
        if lines:
            blocks.append(RawBlock(
                block_no=block["number"],
                bbox=_bbox(block["bbox"]),
                lines=lines,
            ))

    return RawPage(
        page_number=page.number + 1,
        width=round(data["width"], _PRECISION),
        height=round(data["height"], _PRECISION),
        blocks=blocks,
    )


def extract_raw_document(source: PaperSource, document: pymupdf.Document) -> RawDocument:
    """Extract every page. Fails on the first page that cannot be extracted:
    a document with a silently missing page would give wrong answers later."""

    start = time.perf_counter()
    pages: list[RawPage] = []

    for page in document.pages():
        try:
            pages.append(extract_page(page))
        except Exception as exc:
            raise ExtractionError(
                f"Failed to extract page {page.number + 1}: {exc}",
                paper_id=source.paper_id,
                page_number=page.number + 1,
            ) from exc

    raw = RawDocument(paper_id=source.paper_id, page_count=document.page_count, pages=pages)

    lines = [line for page in pages for block in page.blocks for line in block.lines]
    spans = [span for line in lines for span in line.spans]

    empty_pages = [page.page_number for page in pages if not page.blocks]
    if empty_pages:
        logger.warning("{} | pages with no text: {}", source.paper_id, empty_pages)

    unmapped = sum(span.text.count(_UNMAPPED_CHAR) for span in spans)
    if unmapped:
        logger.warning("{} | {} characters could not be mapped to Unicode", source.paper_id, unmapped)

    logger.info(
        "Extracted {} | {} pages | {} blocks | {} lines | {} spans | {} chars | {:.0f} ms",
        source.paper_id,
        len(pages),
        sum(len(page.blocks) for page in pages),
        len(lines),
        len(spans),
        sum(len(span.text) for span in spans),
        (time.perf_counter() - start) * 1000,
    )

    return raw


def main() -> int:
    failures = 0

    for source in PAPERS:
        try:
            with open_pdf(source) as document:
                extract_raw_document(source, document)
        except IngestionError as exc:
            failures += 1
            logger.error("Cannot ingest: {}", exc)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
