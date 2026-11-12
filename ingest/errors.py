
class IngestionError(Exception):
    """Base class for ingestion failures. Records which paper failed."""

    def __init__(self, message: str, *, paper_id: str | None = None):
        super().__init__(message)
        self.message = message
        self.paper_id = paper_id

    def __str__(self) -> str:
        return f"[{self.paper_id}] {self.message}" if self.paper_id else self.message


class PdfNotFoundError(IngestionError):
    """The PDF file is missing."""


class PdfOpenError(IngestionError):
    """The file exists but PyMuPDF cannot open it: empty, corrupt, or unreadable."""


class ExtractionError(IngestionError):
    """Text could not be extracted from a page."""

    def __init__(self, message: str, *, paper_id: str | None = None, page_number: int | None = None):
        super().__init__(message, paper_id=paper_id)
        self.page_number = page_number
