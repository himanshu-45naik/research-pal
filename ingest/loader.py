"""Step 1: open a paper's PDF safely.

Every later step gets its document from open_pdf(), so a missing or
unreadable file surfaces here, once, with a clear error.

Check all papers from the project root:
    python -m ingest.loader
"""

import sys
from collections.abc import Iterator
from contextlib import contextmanager

import pymupdf
from loguru import logger

from ingest.errors import IngestionError, PdfNotFoundError, PdfOpenError
from ingest.sources import PAPERS
from schemas.ingestion import PaperSource


@contextmanager
def open_pdf(source: PaperSource) -> Iterator[pymupdf.Document]:
    """Open a paper's PDF. Closes the document on exit.

    Raises PdfNotFoundError or PdfOpenError.
    """

    path = source.pdf_path

    if not path.is_file():
        raise PdfNotFoundError(f"PDF not found at {path}", paper_id=source.paper_id)

    try:
        document = pymupdf.open(path)
    except RuntimeError as exc:
        # pymupdf.FileDataError (empty or corrupt file) subclasses RuntimeError.
        raise PdfOpenError(f"Cannot open {path.name}: {exc}", paper_id=source.paper_id) from exc

    try:
        logger.info(
            "Loaded {} | {} pages | {:.1f} MB",
            source.paper_id,
            document.page_count,
            path.stat().st_size / 1_000_000,
        )

        yield document

    finally:
        document.close()


def main() -> int:
    failures = 0

    for source in PAPERS:
        try:
            with open_pdf(source):
                pass
        except IngestionError as exc:
            failures += 1
            logger.error("Cannot ingest: {}", exc)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
