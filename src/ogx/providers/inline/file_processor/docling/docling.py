# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import io
import os
import tempfile
import time
import uuid
from typing import Any, NamedTuple

from fastapi import UploadFile

# `docling-core` ships only the lightweight type definitions; the heavy `docling` package
# (DocumentConverter, HybridChunker) is imported lazily inside the methods that need it. This
# keeps unit-test imports cheap and avoids dragging in torch/HuggingFace at module-load time.
try:
    from docling_core.types.doc.document import PictureItem
except ImportError:  # pragma: no cover - fallback only exercised when docling-core is absent

    class PictureItem:  # type: ignore[no-redef]
        """Sentinel — isinstance checks always return False when the real type is unavailable."""


from ogx.log import get_logger
from ogx.providers.utils.vector_io.vector_utils import generate_chunk_id
from ogx_api.file_processors import ProcessFileRequest, ProcessFileResponse
from ogx_api.files import OpenAIFilePurpose, RetrieveFileContentRequest, RetrieveFileRequest, UploadFileRequest
from ogx_api.vector_io import (
    Chunk,
    ChunkMetadata,
    VectorStoreChunkingStrategy,
)

from .config import DoclingFileProcessorConfig

log = get_logger(name=__name__, category="providers::file_processors")

# Metadata key under which extracted image file_ids are stored on each chunk. Documented as part
# of the multimodal RAG pipeline contract — Phase 2 work consumes this key during retrieval and
# Phase 3 work folds the referenced images into the next chat completion as input_image parts.
IMAGE_FILE_IDS_METADATA_KEY = "image_file_ids"


class ExtractedPicture(NamedTuple):
    """A picture rendered from the source document and uploaded to the Files API.

    `pages` is the set of 1-indexed page numbers the picture appears on, taken from the
    docling provenance entries. Chunks whose text overlaps any of these pages will pick the
    picture up via `_collect_chunk_image_file_ids`.
    """

    picture_ref: str
    file_id: str
    pages: frozenset[int]


class DoclingFileProcessor:
    """Docling-based file processor with structure-aware chunking.

    Supports multiple file formats via docling's DocumentConverter (PDF, DOCX, PPTX, HTML, images, etc.).
    When `config.extract_images` is enabled and a Files API binding is available, each embedded
    picture in the source document is rasterised, uploaded to the Files API, and its returned
    file_id is appended to the owning chunk's metadata under `image_file_ids`.
    """

    def __init__(self, config: DoclingFileProcessorConfig, files_api: Any | None = None) -> None:
        self.config = config
        self.files_api = files_api

    async def process_file(
        self,
        request: ProcessFileRequest,
        file: UploadFile | None = None,
    ) -> ProcessFileResponse:
        """Process a file using docling and return chunks."""
        file_id = request.file_id
        chunking_strategy = request.chunking_strategy

        # Validate input
        if not file and not file_id:
            raise ValueError("Either file or file_id must be provided")
        if file and file_id:
            raise ValueError("Cannot provide both file and file_id")

        start_time = time.time()

        # Get file content
        if file:
            content = await file.read()
            filename = file.filename or f"{uuid.uuid4()}.bin"
        elif file_id:
            file_info = await self.files_api.openai_retrieve_file(RetrieveFileRequest(file_id=file_id))
            filename = file_info.filename

            content_response = await self.files_api.openai_retrieve_file_content(
                RetrieveFileContentRequest(file_id=file_id)
            )
            content = content_response.body

        # Preserve original file extension so DocumentConverter can detect the format
        suffix = os.path.splitext(filename)[1] or ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(content)
            tmp.flush()

            converter = self._build_converter()
            result = converter.convert(tmp.name)

        doc = result.document
        page_count = doc.num_pages()

        document_id = str(uuid.uuid4())

        document_metadata: dict[str, Any] = {"filename": filename}
        if file_id:
            document_metadata["file_id"] = file_id

        # Extract and upload images before chunking so the picture metadata is ready when we
        # walk chunks. Failures upload-side degrade gracefully: chunks are still produced with
        # no image_file_ids.
        pictures = await self._extract_and_upload_pictures(doc, filename=filename)

        chunks = self._create_chunks(doc, document_id, chunking_strategy, document_metadata, pictures)

        processing_time_ms = int((time.time() - start_time) * 1000)

        response_metadata: dict[str, Any] = {
            "processor": "docling",
            "processing_time_ms": processing_time_ms,
            "page_count": page_count,
            "extraction_method": "docling",
            "file_size_bytes": len(content),
            "extracted_image_count": len(pictures),
        }

        return ProcessFileResponse(chunks=chunks, metadata=response_metadata)

    def _build_converter(self) -> Any:
        """Construct a DocumentConverter, enabling per-picture image generation when configured.

        Picture images are populated only when the underlying pipeline rasterises them. For PDFs
        that means PdfPipelineOptions.generate_picture_images=True plus a non-trivial images_scale
        — without these, PictureItem.get_image(doc) returns None and we have nothing to upload.
        """
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        if not self.config.extract_images:
            return DocumentConverter()

        pdf_pipeline = PdfPipelineOptions()
        pdf_pipeline.images_scale = self.config.images_scale
        pdf_pipeline.generate_picture_images = True
        # The table-structure and OCR pipelines pull heavy ML models (TableFormer,
        # EasyOCR/Tesseract) whose transitive deps include OpenCV → libxcb.so.1, which is not
        # present in the slim OGX runtime image. Phase 1 only needs picture extraction;
        # disabling these keeps initialisation cheap and side-steps the missing X11 lib.
        pdf_pipeline.do_table_structure = False
        pdf_pipeline.do_ocr = False

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline),
            }
        )

    async def _extract_and_upload_pictures(self, doc: Any, filename: str) -> list[ExtractedPicture]:
        """Walk the document, rasterise each picture, upload via Files API.

        Returns the extracted pictures in document iteration order, each carrying its uploaded
        file_id and the set of page numbers it appears on (for chunk-locality matching downstream).
        """
        if not self.config.extract_images or self.files_api is None:
            return []

        pictures: list[ExtractedPicture] = []
        base_name, _ = os.path.splitext(filename)
        min_dim = self.config.min_image_dim_px

        # iterate_items walks the doc tree; with_groups=False yields content items, traverse_pictures
        # is irrelevant here since PictureItem nodes are visited at the top level by default.
        for item, _level in doc.iterate_items():
            if not isinstance(item, PictureItem):
                continue

            try:
                pil_image = item.get_image(doc)
            except Exception as e:
                log.warning("Failed to render picture", picture_ref=item.self_ref, error=str(e))
                continue

            if pil_image is None:
                continue

            if pil_image.width < min_dim or pil_image.height < min_dim:
                log.debug(
                    "Skipping picture below min_image_dim_px",
                    picture_ref=item.self_ref,
                    width=pil_image.width,
                    height=pil_image.height,
                    min_dim=min_dim,
                )
                continue

            buffer = io.BytesIO()
            try:
                pil_image.save(buffer, format="PNG")
            except Exception as e:
                log.warning("Failed to encode picture as PNG", picture_ref=item.self_ref, error=str(e))
                continue

            png_bytes = buffer.getvalue()

            try:
                file_obj = await self._upload_picture_bytes(png_bytes, base_name, item.self_ref)
            except Exception as e:
                log.warning("Failed to upload picture", picture_ref=item.self_ref, error=str(e))
                continue

            pages = frozenset(_pages_for_item(item))
            pictures.append(ExtractedPicture(picture_ref=item.self_ref, file_id=file_obj.id, pages=pages))

        log.info(
            "Extracted document pictures",
            filename=filename,
            extracted_count=len(pictures),
        )
        return pictures

    async def _upload_picture_bytes(self, png_bytes: bytes, base_name: str, picture_ref: str) -> Any:
        """Upload PNG bytes to the Files API as an OpenAIFilePurpose.ASSISTANTS file."""
        safe_ref = picture_ref.replace("#", "").replace("/", "_").strip("_")
        upload_filename = f"{base_name}_{safe_ref}.png"

        buffer = io.BytesIO(png_bytes)
        upload = UploadFile(file=buffer, filename=upload_filename)
        # The Starlette/FastAPI UploadFile reports content_type via its headers attribute; reading
        # it back through self.content_type works on supported versions, but we don't rely on it
        # — the Files API only needs the bytes.

        return await self.files_api.openai_upload_file(
            request=UploadFileRequest(purpose=OpenAIFilePurpose.ASSISTANTS),
            file=upload,
        )

    def _create_chunks(
        self,
        doc: Any,
        document_id: str,
        chunking_strategy: VectorStoreChunkingStrategy | None,
        document_metadata: dict[str, Any],
        pictures: list[ExtractedPicture],
    ) -> list[Chunk]:
        """Create chunks from a docling Document.

        Chunking semantics:
        - chunking_strategy is None -> return all text as a single chunk
        - chunking_strategy.type == "auto" -> HybridChunker with configured defaults
        - chunking_strategy.type == "static" -> HybridChunker with provided max_tokens
        """
        if not chunking_strategy:
            # No chunking — collect all text as a single chunk. With nothing to match pictures
            # against, every extracted picture is attached to the single chunk.
            text = doc.export_to_markdown()
            if not text or not text.strip():
                return []

            chunk_id = generate_chunk_id(document_id, text)
            metadata: dict[str, Any] = {
                "document_id": document_id,
                **document_metadata,
            }
            if pictures:
                metadata[IMAGE_FILE_IDS_METADATA_KEY] = [p.file_id for p in pictures]

            return [
                Chunk(
                    content=text,
                    chunk_id=chunk_id,
                    metadata=metadata,
                    chunk_metadata=ChunkMetadata(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        source=document_metadata.get("filename", ""),
                        content_token_count=len(text.split()),
                    ),
                )
            ]

        # Determine max_tokens based on strategy
        if chunking_strategy.type == "auto":
            max_tokens = self.config.default_chunk_size_tokens
        elif chunking_strategy.type == "static":
            max_tokens = chunking_strategy.static.max_chunk_size_tokens
        else:
            max_tokens = self.config.default_chunk_size_tokens

        from docling.chunking import HybridChunker
        from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

        # max_tokens is set on the tokenizer, not on HybridChunker directly
        default_chunker = HybridChunker()
        tokenizer = HuggingFaceTokenizer(
            tokenizer=default_chunker.tokenizer.tokenizer,  # type: ignore[attr-defined]
            max_tokens=max_tokens,
        )
        chunker = HybridChunker(tokenizer=tokenizer)
        doc_chunks = list(chunker.chunk(doc))

        if not doc_chunks:
            return []

        chunks: list[Chunk] = []
        for i, doc_chunk in enumerate(doc_chunks):
            text = doc_chunk.text
            if not text or not text.strip():
                continue

            headings = getattr(doc_chunk, "headings", None)
            chunk_window = f"{i}"

            chunk_id = generate_chunk_id(document_id, text, chunk_window)

            meta: dict[str, Any] = {
                "document_id": document_id,
                **document_metadata,
            }
            if headings:
                meta["headings"] = headings

            chunk_image_ids = self._collect_chunk_image_file_ids(doc_chunk, pictures)
            if chunk_image_ids:
                meta[IMAGE_FILE_IDS_METADATA_KEY] = chunk_image_ids

            chunks.append(
                Chunk(
                    content=text,
                    chunk_id=chunk_id,
                    metadata=meta,
                    chunk_metadata=ChunkMetadata(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        source=document_metadata.get("filename", ""),
                        content_token_count=len(text.split()),
                        chunk_window=chunk_window,
                    ),
                )
            )

        return chunks

    @staticmethod
    def _collect_chunk_image_file_ids(doc_chunk: Any, pictures: list[ExtractedPicture]) -> list[str]:
        """Return the file_ids of pictures whose pages overlap this chunk's text.

        HybridChunker emits chunks containing only TextItem refs in `meta.doc_items` — never
        PictureItem refs — so matching by `self_ref` produces no hits. Instead we compute the
        set of pages spanned by the chunk's text items (via their `prov[*].page_no`) and pick
        the pictures whose page set intersects. Pictures are returned in document order (the
        order they were extracted by `_extract_and_upload_pictures`); duplicates within a chunk
        are not possible because each picture appears once in the list.
        """
        if not pictures:
            return []

        meta = getattr(doc_chunk, "meta", None)
        doc_items = getattr(meta, "doc_items", None) if meta is not None else None
        if not doc_items:
            return []

        chunk_pages: set[int] = set()
        for source_item in doc_items:
            chunk_pages.update(_pages_for_item(source_item))
        if not chunk_pages:
            return []

        return [p.file_id for p in pictures if p.pages & chunk_pages]

    async def shutdown(self) -> None:
        pass


def _pages_for_item(item: Any) -> list[int]:
    """Extract page_no values from a docling item's prov entries. Tolerates missing fields."""
    prov = getattr(item, "prov", None)
    if not prov:
        return []
    pages: list[int] = []
    for entry in prov:
        page_no = getattr(entry, "page_no", None)
        if isinstance(page_no, int):
            pages.append(page_no)
    return pages
