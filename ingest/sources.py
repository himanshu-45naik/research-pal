
from pathlib import Path

from schemas.ingestion import PaperSource


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = PROJECT_ROOT / "research_papers"


PAPERS: tuple[PaperSource, ...] = (
    PaperSource(
        paper_id="1706.03762v7",
        short_name="attention",
        title="Attention Is All You Need",
        pdf_path=PAPERS_DIR / "1706.03762v7.pdf",
    ),
    PaperSource(
        paper_id="2410.00037v1",
        short_name="moshi",
        title="Moshi: a speech-text foundation model for real-time dialogue",
        pdf_path=PAPERS_DIR / "2410.00037v1.pdf",
    ),
    PaperSource(
        paper_id="2507.07955v2",
        short_name="hnet",
        title="Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        pdf_path=PAPERS_DIR / "2507.07955v2.pdf",
    ),
)
