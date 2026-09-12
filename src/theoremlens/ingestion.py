"""Document parsing and parent-child chunking using Docling and RecursiveCharacterTextSplitter."""

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from theoremlens.config import config

logger = logging.getLogger(__name__)


@dataclass
class ChildChunk:
    child_id: str
    content: str
    parent_id: str
    parent_content: str
    source: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ParentChunk:
    parent_id: str
    content: str
    source: str
    children: List[ChildChunk] = field(default_factory=list)


class MathDocumentParser:
    """Parses academic mathematical PDFs into Markdown with preserved LaTeX and tables using Docling."""

    def __init__(self):
        self._converter: Optional[DocumentConverter] = None

    @property
    def converter(self) -> DocumentConverter:
        if self._converter is None:
            logger.info("Initializing Docling DocumentConverter (CPU ONNX mode)...")
            self._converter = DocumentConverter()
        return self._converter

    def parse_to_markdown(self, pdf_path: str | Path) -> Tuple[str, str]:
        """Converts a PDF file to markdown text. Returns (markdown_text, source_filename)."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        logger.info(f"Parsing PDF with Docling: {path.name}")
        conv_result = self.converter.convert(str(path))
        markdown_text = conv_result.document.export_to_markdown()
        logger.info(f"Successfully converted {path.name} to markdown ({len(markdown_text)} characters).")
        return markdown_text, path.name


class ParentChildChunker:
    """Implements the Parent-Child chunking pattern for math texts.

    - Parent chunk: 1,200 chars (overlap 150)
    - Child chunk: 250 chars (overlap 40)
    """

    def __init__(
        self,
        parent_chunk_size: int = config.parent_chunk_size,
        parent_chunk_overlap: int = config.parent_chunk_overlap,
        child_chunk_size: int = config.child_chunk_size,
        child_chunk_overlap: int = config.child_chunk_overlap,
        parent_separators: Optional[List[str]] = None,
    ):
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap
        self.parent_separators = parent_separators or config.parent_separators

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.parent_chunk_overlap,
            separators=self.parent_separators,
        )

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.child_chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def split_document(self, text: str, source: str) -> List[ParentChunk]:
        """Splits markdown text into parent chunks, and splits each parent chunk into child chunks."""
        parent_texts = self.parent_splitter.split_text(text)
        logger.info(f"Split document '{source}' into {len(parent_texts)} parent chunks.")

        parent_chunks: List[ParentChunk] = []

        for p_idx, p_text in enumerate(parent_texts):
            p_id = str(uuid.uuid4())
            parent = ParentChunk(parent_id=p_id, content=p_text, source=source)

            child_texts = self.child_splitter.split_text(p_text)
            for c_idx, c_text in enumerate(child_texts):
                c_id = f"{p_id}_c{c_idx}"
                child = ChildChunk(
                    child_id=c_id,
                    content=c_text,
                    parent_id=p_id,
                    parent_content=p_text,
                    source=source,
                    metadata={
                        "parent_id": p_id,
                        "parent_content": p_text,
                        "child_id": c_id,
                        "source": source,
                        "parent_index": p_idx,
                        "child_index": c_idx,
                    },
                )
                parent.children.append(child)

            parent_chunks.append(parent)

        total_children = sum(len(p.children) for p in parent_chunks)
        logger.info(f"Generated {total_children} child chunks from {len(parent_chunks)} parent chunks.")
        return parent_chunks


def parse_and_chunk_pdf(pdf_path: str | Path) -> Tuple[List[ParentChunk], str]:
    """Helper function to parse a PDF and return parent chunks with markdown text."""
    parser = MathDocumentParser()
    markdown_text, source_name = parser.parse_to_markdown(pdf_path)
    chunker = ParentChildChunker()
    parents = chunker.split_document(markdown_text, source=source_name)
    return parents, markdown_text
