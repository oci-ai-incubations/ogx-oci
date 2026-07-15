# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import base64
import io
import json
import os
import tempfile
import time
import uuid
from typing import Any, NamedTuple

from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

# `docling-core` ships only the lightweight type definitions; the heavy `docling` package
# (DocumentConverter, HybridChunker) is imported lazily inside the methods that need it. This
# keeps unit-test imports cheap and avoids dragging in torch/HuggingFace at module-load time.
try:
    from docling_core.types.doc.document import PictureItem
except ImportError:  # pragma: no cover - fallback only exercised when docling-core is absent

    class PictureItem:  # type: ignore[no-redef]
        """Sentinel — isinstance checks always return False when the real type is unavailable."""


from ogx.log import get_logger
from ogx.providers.inline.file_processor.zip_utils import validate_zip_content
from ogx.providers.utils.files.response import response_body_bytes
from ogx.providers.utils.vector_io.vector_utils import generate_chunk_id
from ogx_api.file_processors import (
    EXTRACTED_IMAGE_FILE_IDS_METADATA_KEY,
    ProcessFileRequest,
    ProcessFileResponse,
)
from ogx_api.files import (
    OpenAIFileUploadPurpose,
    RetrieveFileContentRequest,
    RetrieveFileRequest,
    UploadFileRequest,
)
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
#
# Stored as a JSON-encoded string, NOT a list: the OpenAI vector store search response schema
# (VectorStoreSearchResponse.attributes) constrains each value to str | float | bool, so a raw
# list[str] fails Pydantic validation and aborts the entire file_search call with empty results.
# Consumers must json.loads() to recover the list.
IMAGE_FILE_IDS_METADATA_KEY = "image_file_ids"

# EXTRACTED_IMAGE_FILE_IDS_METADATA_KEY now lives in ogx_api.file_processors so the producer
# (this provider) and consumer (openai_vector_store_mixin) share a single source of truth.
# Re-exported here for backward compatibility with anything that still imports it from docling.
__all__ = [
    "EXTRACTED_IMAGE_FILE_IDS_METADATA_KEY",
    "IMAGE_FILE_IDS_METADATA_KEY",
    "DoclingFileProcessor",
    "ExtractedPicture",
]

# Filename suffixes that docling routes through InputFormat.IMAGE. A standalone image of any of
# these types is treated by docling as the document itself rather than an embedded picture, so
# `iterate_items` yields no PictureItem and the regular extraction path returns []. We special-
# case this in process_file by synthesising a self-picture so the image-only fallback chunk path
# can still produce a retrievable entry.
_IMAGE_INPUT_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif"})


class ExtractedPicture(NamedTuple):
    """A picture rendered from the source document and uploaded to the Files API.

    `pages` is the set of 1-indexed page numbers the picture appears on, taken from the
    docling provenance entries. Chunks whose text overlaps any of these pages will pick the
    picture up via `_collect_chunk_image_file_ids`.

    `caption` is populated only when caption_images is enabled on the config; otherwise it is
    None. For image-only inputs (PNG/JPG uploaded directly), the caption is what makes the
    chunk's text content semantically meaningful — without it the chunker has nothing to embed.
    """

    picture_ref: str
    file_id: str
    pages: frozenset[int]
    caption: str | None = None


DOCLING_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # .pptx
    "text/html",
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/bmp",
    "image/tiff",
    "image/webp",
}


class DoclingFileProcessor:
    """Docling-based file processor with structure-aware chunking.

    Supports multiple file formats via docling's DocumentConverter (PDF, DOCX, PPTX, HTML, images, etc.).
    When `config.extract_images` is enabled and a Files API binding is available, each embedded
    picture in the source document is rasterised, uploaded to the Files API, and its returned
    file_id is appended to the owning chunk's metadata under `image_file_ids`.
    """

    def __init__(
        self,
        config: DoclingFileProcessorConfig,
        files_api: Any | None = None,
        inference_api: Any | None = None,
    ) -> None:
        self.config = config
        self.files_api = files_api
        self.inference_api = inference_api
        self._vlm_enabled = False

        # Fail fast on an invalid vlm_preset at construction rather than deferring to the first
        # process_file call. Only when a VLM pipeline would actually be built (vlm_model set and an
        # inference provider available) — this is the only branch that touches docling's heavy VLM
        # imports, so the common non-VLM path stays import-free at construction time.
        if self.config.vlm_model and self.inference_api:
            self._validate_vlm_preset()

    def supported_mime_types(self) -> set[str] | None:
        return DOCLING_MIME_TYPES

    def _validate_vlm_preset(self) -> None:
        from docling.datamodel.pipeline_options import VlmConvertOptions

        available_presets = list(VlmConvertOptions._presets.keys())
        if self.config.vlm_preset not in available_presets:
            raise ValueError(f"Invalid vlm_preset '{self.config.vlm_preset}'. Available presets: {available_presets}")

    def _build_vlm_converter(self) -> Any:
        # docling's VLM pipeline drags in the full docling/torch/HuggingFace stack; import it
        # lazily so deployments that never enable vlm_model pay no import cost at module load.
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
        from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions, VlmEngineType
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.pipeline.vlm_pipeline import VlmPipeline

        from ogx.providers.inline.file_processor.docling.vlm_engine import OgxInferenceVlmEngine

        assert self.config.vlm_model is not None

        self._validate_vlm_preset()

        vlm_options = VlmConvertOptions.from_preset(
            self.config.vlm_preset,
            engine_options=ApiVlmEngineOptions(engine_type=VlmEngineType.API_OPENAI),
        )

        vlm_pipeline_options = VlmPipelineOptions(
            vlm_options=vlm_options,
            enable_remote_services=True,
        )

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=vlm_pipeline_options,
                )
            }
        )

        converter.initialize_pipeline(InputFormat.PDF)

        event_loop = asyncio.get_event_loop()
        ogx_engine = OgxInferenceVlmEngine(
            inference_api=self.inference_api,
            model=self.config.vlm_model,
            event_loop=event_loop,
        )

        for pipeline in converter.initialized_pipelines.values():
            for stage in getattr(pipeline, "build_pipe", []):
                if hasattr(stage, "engine"):
                    stage.engine = ogx_engine
                    break

        self._vlm_enabled = True
        log.info(
            "VLM pipeline enabled",
            vlm_model=self.config.vlm_model,
            vlm_preset=self.config.vlm_preset,
        )

        return converter

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
            assert self.files_api is not None, "Failed to process file_id: files_api is not configured"
            file_info = await self.files_api.openai_retrieve_file(RetrieveFileRequest(file_id=file_id))
            filename = file_info.filename

            content_response = await self.files_api.openai_retrieve_file_content(
                RetrieveFileContentRequest(file_id=file_id)
            )
            content = await response_body_bytes(content_response)

        validate_zip_content(content, filename)

        # Preserve original file extension so DocumentConverter can detect the format
        suffix = os.path.splitext(filename)[1] or ".bin"

        # iPhone HDR / portrait / live-photo shots arrive as MPO (multi-frame JPEG). Docling's
        # PILImageBackend whitelists Pillow's `.format` string and rejects "MPO", so coerce to
        # plain JPEG using the first embedded frame before handing off.
        if _is_image_input_filename(filename):
            content = _coerce_mpo_to_jpeg(content)

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

        # Standalone image uploads (PNG/JPG) don't surface as PictureItem nodes — docling treats
        # the whole image as the document page, so iterate_items yields nothing. Synthesise a
        # self-picture from the source file so the image-only fallback chunk path can produce a
        # captioned (or filename-labelled) entry instead of failing with "No chunks were generated".
        if not pictures and file_id and _is_image_input_filename(filename):
            self_picture = await self._build_self_picture(content=content, file_id=file_id)
            if self_picture is not None:
                pictures = [self_picture]

        chunks = self._create_chunks(doc, document_id, chunking_strategy, document_metadata, pictures)

        processing_time_ms = int((time.time() - start_time) * 1000)

        extraction_method = "docling-vlm" if self._vlm_enabled else "docling"
        response_metadata: dict[str, Any] = {
            "processor": "docling",
            "processing_time_ms": processing_time_ms,
            "page_count": page_count,
            "extraction_method": extraction_method,
            "file_size_bytes": len(content),
            "extracted_image_count": len(pictures),
            # Full list of uploaded image file_ids — consumers (vector_store mixin) use this to
            # tie the lifetime of the extracted images to the parent file, so deletion of the
            # parent vector_store_file cascades cleanup of the children.
            EXTRACTED_IMAGE_FILE_IDS_METADATA_KEY: [p.file_id for p in pictures],
        }
        if self._vlm_enabled:
            response_metadata["vlm_model"] = self.config.vlm_model
            response_metadata["vlm_preset"] = self.config.vlm_preset

        return ProcessFileResponse(chunks=chunks, metadata=response_metadata)

    def _build_converter(self) -> Any:
        """Construct a DocumentConverter, enabling per-picture image generation when configured.

        Picture images are populated only when the underlying pipeline rasterises them. For PDFs
        that means PdfPipelineOptions.generate_picture_images=True plus a non-trivial images_scale
        — without these, PictureItem.get_image(doc) returns None and we have nothing to upload.
        """
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption

        # When a vision model is configured, route to the VLM pipeline instead of the
        # picture-extraction pipeline. The two are mutually exclusive: VLM transcribes the whole
        # document via the model, while the standard path rasterises embedded pictures.
        if self.config.vlm_model and self.inference_api:
            return self._build_vlm_converter()
        if self.config.vlm_model and not self.inference_api:
            log.warning(
                "vlm_model is configured but no inference provider is available, falling back to standard pipeline",
                vlm_model=self.config.vlm_model,
            )

        if not self.config.extract_images:
            return DocumentConverter()

        pdf_pipeline = PdfPipelineOptions()
        pdf_pipeline.images_scale = self.config.images_scale
        pdf_pipeline.generate_picture_images = True
        # The table-structure and OCR pipelines pull heavy ML models (TableFormer,
        # EasyOCR/Tesseract) whose transitive deps include OpenCV → libxcb.so.1, which is not
        # present in the slim OGX runtime image. Phase 1 only needs picture extraction;
        # disabling these keeps initialisation cheap and side-steps the missing X11 lib.
        # Mirrored on the image-format option below because standalone images route through
        # their own ImageFormatOption — which would otherwise reach for the default (table
        # structure ON) pipeline options and reproduce the crash.
        pdf_pipeline.do_table_structure = False
        pdf_pipeline.do_ocr = False

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline),
                InputFormat.IMAGE: ImageFormatOption(pipeline_options=pdf_pipeline),
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
            caption = await self._maybe_caption_picture(png_bytes, picture_ref=item.self_ref)
            pictures.append(
                ExtractedPicture(
                    picture_ref=item.self_ref,
                    file_id=file_obj.id,
                    pages=pages,
                    caption=caption,
                )
            )

        log.info(
            "Extracted document pictures",
            filename=filename,
            extracted_count=len(pictures),
        )
        return pictures

    async def _build_self_picture(self, content: bytes, file_id: str) -> ExtractedPicture | None:
        """Build an ExtractedPicture representing the input file when the input is itself an image.

        The source bytes are already stored in the Files API under `file_id`, so we don't
        re-upload — we just pin the existing id and (best-effort) caption the bytes. Pages are
        fixed at {1} because a standalone image is a single-page document.
        """
        caption = await self._maybe_caption_picture(content, picture_ref="self")
        return ExtractedPicture(
            picture_ref="self",
            file_id=file_id,
            pages=frozenset({1}),
            caption=caption,
        )

    async def _upload_picture_bytes(self, png_bytes: bytes, base_name: str, picture_ref: str) -> Any:
        """Upload PNG bytes to the Files API as an OpenAIFileUploadPurpose.ASSISTANTS file."""
        safe_ref = picture_ref.replace("#", "").replace("/", "_").strip("_")
        upload_filename = f"{base_name}_{safe_ref}.png"

        buffer = io.BytesIO(png_bytes)
        upload = UploadFile(file=buffer, filename=upload_filename)
        # The Starlette/FastAPI UploadFile reports content_type via its headers attribute; reading
        # it back through self.content_type works on supported versions, but we don't rely on it
        # — the Files API only needs the bytes.

        assert self.files_api is not None, "Failed to upload picture bytes: files_api is not configured"
        return await self.files_api.openai_upload_file(
            request=UploadFileRequest(purpose=OpenAIFileUploadPurpose.ASSISTANTS),
            file=upload,
        )

    async def _maybe_caption_picture(self, image_bytes: bytes, picture_ref: str) -> str | None:
        """Call the configured vision model to caption a picture.

        Returns the caption text on success, or None if captioning is disabled, the inference
        API isn't wired in, the configured model is unset, or the call fails. Per-picture
        skips and call-point events log at DEBUG so PDFs with hundreds of figures don't flood
        INFO; configuration mistakes (caption_model unset) and hard failures (caption call
        raised) stay at WARNING so they remain visible at default verbosity. The aggregate
        document-level summary ("Extracted document pictures") still logs at INFO so a
        deployment can see captioning is firing without scanning per-picture spam.
        Caption-generation is best-effort and must not block the rest of ingest from
        succeeding.

        The mime type of the data URL passed to the vision model is detected from the bytes'
        magic header (PNG / JPEG / WebP / GIF), falling back to PNG. Some OCI vision adapters
        reject data URLs whose declared mime doesn't match the body, so a hardcoded
        `image/png` tag was silently rejecting JPG uploads downstream.
        """
        if not self.config.caption_images:
            return None
        if self.inference_api is None:
            log.debug(
                "Caption skipped: caption_images=True but Inference API not bound",
                picture_ref=picture_ref,
            )
            return None
        model_id = self.config.caption_model
        if not model_id:
            log.warning(
                "Caption skipped: caption_images=True but caption_model is unset",
                picture_ref=picture_ref,
            )
            return None

        # Lazy import to keep module-load cheap when captioning isn't used.
        from ogx_api.inference.models import (
            OpenAIChatCompletionContentPartImageParam,
            OpenAIChatCompletionContentPartTextParam,
            OpenAIChatCompletionRequestWithExtraBody,
            OpenAIImageURL,
            OpenAIUserMessageParam,
        )

        mime_type = _detect_image_mime(image_bytes)
        data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
        log.debug(
            "Captioning picture",
            picture_ref=picture_ref,
            model=model_id,
            mime_type=mime_type,
            byte_count=len(image_bytes),
        )
        params = OpenAIChatCompletionRequestWithExtraBody(
            model=model_id,
            messages=[
                OpenAIUserMessageParam(
                    content=[
                        OpenAIChatCompletionContentPartTextParam(text=self.config.caption_prompt),
                        OpenAIChatCompletionContentPartImageParam(image_url=OpenAIImageURL(url=data_url)),
                    ]
                )
            ],
            # OpenAI deprecated `max_tokens` in favour of `max_completion_tokens` on the
            # Chat Completions API, but the OCI generative-ai-inference adapter (and most
            # other adapters we route through) still honours the legacy `max_tokens` name
            # for backwards compatibility. If/when OCI flips to require the new name, swap
            # this to `max_completion_tokens` and re-run the integration tests.
            max_tokens=self.config.caption_max_tokens,
            stream=False,
        )

        try:
            resp = await self.inference_api.openai_chat_completion(params)
        except Exception as e:
            log.warning(
                "Failed to caption picture",
                picture_ref=picture_ref,
                model=model_id,
                error=str(e),
            )
            return None

        caption = _extract_caption_text(resp)
        if caption is None:
            log.debug(
                "Caption response produced no text",
                picture_ref=picture_ref,
                model=model_id,
            )
        return caption

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
                # Mirror the chunker branch's image-only fallback: when the document has no
                # extractable text but we do have pictures (e.g. a standalone image input
                # synthesised into a self-picture), emit one chunk per picture so the file is
                # still retrievable instead of returning an empty list.
                if pictures:
                    return self._build_image_only_chunks(document_id, document_metadata, pictures)
                return []

            chunk_id = generate_chunk_id(document_id, text)
            metadata: dict[str, Any] = {
                "document_id": document_id,
                **document_metadata,
            }
            if pictures:
                metadata[IMAGE_FILE_IDS_METADATA_KEY] = json.dumps([p.file_id for p in pictures])

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
                meta[IMAGE_FILE_IDS_METADATA_KEY] = json.dumps(chunk_image_ids)

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

        # Image-only inputs (PNG/JPG, OCR-disabled PDFs of pure imagery) produce no text chunks
        # from HybridChunker. Fall back to emitting one chunk per extracted picture so the file
        # has a retrievable presence in the vector store.
        if not chunks and pictures:
            chunks.extend(self._build_image_only_chunks(document_id, document_metadata, pictures))

        return chunks

    @staticmethod
    def _build_image_only_chunks(
        document_id: str,
        document_metadata: dict[str, Any],
        pictures: list[ExtractedPicture],
    ) -> list[Chunk]:
        """Emit one chunk per picture for documents that produced no text chunks.

        Used as a fallback path so standalone-image uploads (PNG/JPG) still result in
        retrievable vector store entries. Caption text — when available — is what makes the
        chunk semantically searchable; without it the body falls back to the filename so the
        chunk is at least findable by document identity.
        """
        filename = document_metadata.get("filename", "")
        chunks: list[Chunk] = []
        for i, picture in enumerate(pictures):
            if picture.caption:
                text_body = picture.caption
            elif filename:
                text_body = f"Image: {filename}"
            else:
                text_body = "Image"
            chunk_window = f"image-{i}"
            chunk_id = generate_chunk_id(document_id, text_body, chunk_window)
            meta: dict[str, Any] = {
                "document_id": document_id,
                **document_metadata,
                IMAGE_FILE_IDS_METADATA_KEY: json.dumps([picture.file_id]),
            }
            chunks.append(
                Chunk(
                    content=text_body,
                    chunk_id=chunk_id,
                    metadata=meta,
                    chunk_metadata=ChunkMetadata(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        source=filename,
                        content_token_count=len(text_body.split()),
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


def _is_image_input_filename(filename: str) -> bool:
    """Return True when the filename suffix is one docling routes through InputFormat.IMAGE."""
    return os.path.splitext(filename)[1].lower() in _IMAGE_INPUT_SUFFIXES


def _coerce_mpo_to_jpeg(content: bytes) -> bytes:
    """Return JPEG bytes for an MPO blob, or the original bytes unchanged.

    MPO files carry a JPEG magic prefix but Pillow reports format=="MPO", which docling's
    PILImageBackend rejects with ConversionError("File format not allowed: ..."). Re-encoding
    the first embedded frame yields a single-frame JPEG that the backend accepts.
    """
    try:
        with Image.open(io.BytesIO(content)) as probe:
            if probe.format != "MPO":
                return content
            probe.seek(0)
            buf = io.BytesIO()
            probe.convert("RGB").save(buf, format="JPEG", quality=95)
            return buf.getvalue()
    except (UnidentifiedImageError, OSError):
        return content


def _detect_image_mime(image_bytes: bytes) -> str:
    """Sniff the image format from its magic header.

    Falls back to image/png so callers that pass already-rendered PNG bytes (the existing
    PictureItem path) keep working without change. The self-picture path passes raw upload
    bytes, which may be JPEG / WebP / GIF — vision adapters typically validate the data URL
    mime against the body, so a wrong tag silently fails the caption call.
    """
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"GIF87a") or image_bytes.startswith(b"GIF89a"):
        return "image/gif"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    if image_bytes[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    return "image/png"


def _extract_caption_text(resp: Any) -> str | None:
    """Pull caption text from an OpenAI-shaped chat completion response.

    `message.content` may be a plain string (classic Chat Completions) or a list of content
    parts each with `type` + `text` (the multimodal-aware shape some providers return). Return
    the concatenated text of all text-parts, stripped, or None when nothing usable is present.
    """
    choices = getattr(resp, "choices", None) or []
    if not choices:
        return None
    message = getattr(choices[0], "message", None)
    if message is None:
        return None
    content = getattr(message, "content", None)
    if isinstance(content, str):
        stripped = content.strip()
        return stripped or None
    if isinstance(content, list):
        texts: list[str] = []
        for part in content:
            part_type = getattr(part, "type", None) or (part.get("type") if isinstance(part, dict) else None)
            if part_type in ("text", "output_text"):
                text = getattr(part, "text", None) or (part.get("text") if isinstance(part, dict) else None)
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
        if texts:
            return " ".join(texts)
    return None


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
