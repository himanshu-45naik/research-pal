
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


# arXiv id with an explicit version, e.g. "1706.03762v7".
ARXIV_ID_PATTERN = r"^\d{4}\.\d{4,5}v\d+$"


class PaperSource(BaseModel):
    """A paper the library may answer from, and where its PDF lives."""

    model_config = ConfigDict(frozen=True)

    paper_id: str = Field(pattern=ARXIV_ID_PATTERN)
    short_name: str = Field(pattern=r"^[a-z0-9_]+$")
    title: str = Field(min_length=1)
    pdf_path: Path

    # mypy does not understand decorators stacked on @property; Pydantic's
    # docs recommend this ignore for computed fields.
    @computed_field  # type: ignore[prop-decorator]
    @property
    def arxiv_id(self) -> str:
        return self.paper_id.rsplit("v", 1)[0]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def version(self) -> str:
        return "v" + self.paper_id.rsplit("v", 1)[1]


# x0, y0, x1, y1 in PDF points (1/72 inch), origin at the top-left of the page.
BBox = tuple[float, float, float, float]

# PyMuPDF font flag bits.
_FONT_SUPERSCRIPT = 1
_FONT_ITALIC = 2
_FONT_MONOSPACED = 8
_FONT_BOLD = 16


class RawSpan(BaseModel):
    """A run of text in one font, size and colour."""

    model_config = ConfigDict(frozen=True)

    text: str
    bbox: BBox
    font: str
    size: float = Field(gt=0)
    flags: int = Field(ge=0)      # PyMuPDF font flag bits
    color: int = Field(ge=0)      # sRGB packed as 0xRRGGBB

    @property
    def is_bold(self) -> bool:
        return bool(self.flags & _FONT_BOLD)

    @property
    def is_italic(self) -> bool:
        return bool(self.flags & _FONT_ITALIC)

    @property
    def is_superscript(self) -> bool:
        return bool(self.flags & _FONT_SUPERSCRIPT)

    @property
    def is_monospaced(self) -> bool:
        return bool(self.flags & _FONT_MONOSPACED)


class RawLine(BaseModel):
    """One line of text as laid out on the page."""

    model_config = ConfigDict(frozen=True)

    bbox: BBox
    # Writing direction as a unit vector. (1, 0) is normal left-to-right text;
    # anything else is rotated, like the arXiv stamp in the page-1 margin.
    direction: tuple[float, float]
    spans: list[RawSpan] = Field(min_length=1)

    @property
    def text(self) -> str:
        return "".join(span.text for span in self.spans)

    @property
    def is_horizontal(self) -> bool:
        return self.direction == (1.0, 0.0)


class RawBlock(BaseModel):
    """A group of lines PyMuPDF placed together, usually a paragraph."""

    model_config = ConfigDict(frozen=True)

    block_no: int = Field(ge=0)   # PyMuPDF's block number on the page
    bbox: BBox
    lines: list[RawLine] = Field(min_length=1)

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


class RawPage(BaseModel):
    """All text blocks on one page, in the PDF's content order."""

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(ge=1)   # 1-based
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    blocks: list[RawBlock]


class RawDocument(BaseModel):
    """Raw text of a whole paper, before any cleanup."""

    model_config = ConfigDict(frozen=True)

    paper_id: str = Field(pattern=ARXIV_ID_PATTERN)
    page_count: int = Field(ge=1)
    pages: list[RawPage]

    @model_validator(mode="after")
    def _check_pages(self) -> "RawDocument":
        numbers = [page.page_number for page in self.pages]
        if numbers != list(range(1, self.page_count + 1)):
            raise ValueError(f"Expected pages 1..{self.page_count} in order, got {numbers[:5]}...")
        return self
