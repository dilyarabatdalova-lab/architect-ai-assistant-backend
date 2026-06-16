from pathlib import Path
from typing import Dict, List
import re

from docx import Document
from pypdf import PdfReader


class DocumentService:
    """Loads prepared PDF, DOCX and TXT files from the developer document base."""

    def __init__(self, documents_dir: Path) -> None:
        self.documents_dir = documents_dir

    def load_documents(self) -> List[Dict[str, str]]:
        """Read all supported files and split their text into searchable chunks."""
        chunks: List[Dict[str, str]] = []

        for path in self.documents_dir.rglob("*"):
            if not path.is_file():
                continue

            text = self._read_file(path)

            if not text.strip():
                continue

            for index, chunk in enumerate(self._split_text(text)):
                chunks.append(
                    {
                        "id": f"{path.name}_{index}",
                        "source": path.name,
                        "text": chunk,
                    }
                )

        return chunks

    def _read_file(self, path: Path) -> str:
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return self._read_pdf(path)

        if suffix == ".docx":
            return self._read_docx(path)

        if suffix == ".txt":
            return path.read_text(encoding="utf-8", errors="ignore")

        return ""

    def _read_pdf(self, path: Path) -> str:
        reader = PdfReader(str(path))
        pages = []

        for page in reader.pages:
            pages.append(page.extract_text() or "")

        return "\n".join(pages)

    def _read_docx(self, path: Path) -> str:
        document = Document(str(path))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)

    def _split_text(self, text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
        """Split text with overlap so important rules are not cut off at boundaries."""
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        sections = re.split(r"(?=Раздел:)", text)
        section_chunks = []

        for section in sections:
            if not section.strip().startswith("Раздел:"):
                continue

            lines = [" ".join(line.split()) for line in section.split("\n")]
            section_chunks.append("\n".join(line for line in lines if line))

        if section_chunks:
            return section_chunks

        normalized = " ".join(text.split())

        if len(normalized) <= chunk_size:
            return [normalized]

        chunks = []
        start = 0

        while start < len(normalized):
            end = start + chunk_size
            chunks.append(normalized[start:end])
            start = max(end - overlap, start + 1)

        return chunks
