# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import mimetypes
from typing import Any

from fastapi import HTTPException, UploadFile

from ogx.log import get_logger
from ogx.providers.inline.file_processor.docling.config import DoclingFileProcessorConfig
from ogx.providers.inline.file_processor.docling.docling import DoclingFileProcessor
from ogx.providers.inline.file_processor.markitdown.config import MarkItDownFileProcessorConfig
from ogx.providers.inline.file_processor.markitdown.markitdown_processor import MarkItDownFileProcessor
from ogx.providers.inline.file_processor.pypdf.config import PyPDFFileProcessorConfig
from ogx.providers.inline.file_processor.pypdf.pypdf import PYPDF_TEXT_LIKE_APPLICATION_MIME_TYPES, PyPDFFileProcessor
from ogx_api.file_processors import ProcessFileRequest, ProcessFileResponse
from ogx_api.files import RetrieveFileContentRequest, RetrieveFileRequest

from .config import AutoFileProcessorConfig

log = get_logger(name=__name__, category="providers::file_processors")

# MIME types routed to MarkItDown. Derived from markitdown's bundled converters:
# DocxConverter, PptxConverter, XlsxConverter, XlsConverter, HtmlConverter,
# EpubConverter, OutlookMsgConverter, IpynbConverter, RssConverter, ImageConverter,
# AudioConverter, ZipConverter. CSV, JSON, XML, and text/* are handled by PyPDF.
MARKITDOWN_MIME_TYPES = {
    # Office documents
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # .pptx
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
    "application/msword",  # .doc
    "application/vnd.ms-powerpoint",  # .ppt
    "application/vnd.ms-excel",  # .xls
    "application/rtf",  # .rtf
    # Structured formats
    "application/epub+zip",  # .epub
    "application/rss+xml",  # .rss
    # Archives
    "application/zip",  # .zip
    # Images
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/bmp",
    "image/tiff",
    "image/webp",
    # Audio
    "audio/mpeg",  # .mp3
    "audio/x-wav",  # .wav
}

# Derive the user-facing supported list directly from the routing allowlists so the message
# can't drift from reality. Previously this was a hand-written string that advertised JSON/XML
# even though the router didn't accept them — exactly the kind of doc-vs-code drift this
# derivation prevents.
SUPPORTED_DESCRIPTION = ", ".join(
    ["text/*", *sorted({"application/pdf", *PYPDF_TEXT_LIKE_APPLICATION_MIME_TYPES, *MARKITDOWN_MIME_TYPES})]
)


def _sniff_mime_from_head(head: bytes) -> str | None:
    """Recover a MIME type from the first bytes of a file when the filename has no usable
    extension (e.g. the upload arrived with a missing filename and the S3 provider stored
    it as "uploaded_file"). Conservative — only returns a type when the signature is
    unambiguous. Returning None falls through to the standard 422 rejection.
    """
    if head.startswith(b"%PDF-"):
        return "application/pdf"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith(b"GIF87a") or head.startswith(b"GIF89a"):
        return "image/gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    # ZIP-based Office (docx/pptx/xlsx) all start with PK\x03\x04 and embed "[Content_Types].xml"
    # within the first ~1 KiB. The exact OOXML flavour is determined later by the converter — we
    # just need to route to MarkItDown, which accepts the generic ZIP MIME for these formats.
    if head.startswith(b"PK\x03\x04") and b"[Content_Types].xml" in head[:2048]:
        return "application/zip"
    # Conservative JSON / XML sniffs — peek past leading whitespace.
    stripped = head.lstrip()
    if stripped.startswith(b"{") or stripped.startswith(b"["):
        return "application/json"
    if stripped.startswith(b"<?xml") or stripped[:1] == b"<":
        return "application/xml"
    return None


class AutoFileProcessor:
    """Composite file processor that dispatches to backends based on MIME type.

    Routes PDF and text files to PyPDF. Office documents, images, audio, and
    other rich formats are routed to MarkItDown. Unsupported formats are
    rejected with a 422 error listing the supported types.
    """

    def __init__(self, config: AutoFileProcessorConfig, files_api, inference_api=None) -> None:
        self.config = config
        self.files_api = files_api
        self.inference_api = inference_api

        pypdf_config = PyPDFFileProcessorConfig(
            default_chunk_size_tokens=config.default_chunk_size_tokens,
            default_chunk_overlap_tokens=config.default_chunk_overlap_tokens,
            extract_metadata=config.extract_metadata,
            clean_text=config.clean_text,
        )
        self.pypdf = PyPDFFileProcessor(pypdf_config, files_api)

        markitdown_config = MarkItDownFileProcessorConfig(
            default_chunk_size_tokens=config.default_chunk_size_tokens,
            default_chunk_overlap_tokens=config.default_chunk_overlap_tokens,
        )
        self.markitdown = MarkItDownFileProcessor(markitdown_config, files_api)

        # Lazily-instantiated docling backend — only constructed when prefer_docling_for_pdfs
        # is enabled, so deployments not opting in pay no startup cost for the heavy converter.
        # When constructed, the same instance handles both PDFs and image MIMEs (PNG/JPG), since
        # docling treats a standalone image as a 1-page document with a single PictureItem.
        self.docling: DoclingFileProcessor | None = None
        if config.prefer_docling_for_pdfs:
            # Only override caption_prompt when the user explicitly set one — leaving it None
            # preserves docling's built-in default rather than clobbering it with an empty value.
            docling_kwargs: dict[str, Any] = {
                "default_chunk_size_tokens": config.default_chunk_size_tokens,
                "default_chunk_overlap_tokens": config.default_chunk_overlap_tokens,
                "extract_images": config.default_extract_images,
                "images_scale": config.default_images_scale,
                "min_image_dim_px": config.default_min_image_dim_px,
                "caption_images": config.default_caption_images,
                "caption_model": config.default_caption_model,
                "caption_max_tokens": config.default_caption_max_tokens,
            }
            if config.default_caption_prompt is not None:
                docling_kwargs["caption_prompt"] = config.default_caption_prompt
            docling_config = DoclingFileProcessorConfig(**docling_kwargs)
            self.docling = DoclingFileProcessor(docling_config, files_api=files_api, inference_api=inference_api)

    async def process_file(
        self,
        request: ProcessFileRequest,
        file: UploadFile | None = None,
    ) -> ProcessFileResponse:
        filename = await self._resolve_filename(request, file)
        mime_type, _ = mimetypes.guess_type(filename)

        # Filename-based MIME lookup fails when the upload reached us without a usable name
        # (e.g. multipart Content-Disposition dropped the filename and the S3 provider fell
        # back to "uploaded_file"). Fall back to sniffing the first bytes so a valid file
        # doesn't get a misleading 422 just because its extension was lost upstream.
        if mime_type is None:
            head = await self._read_head(request, file, n=2048)
            sniffed = _sniff_mime_from_head(head)
            if sniffed is not None:
                log.warning(
                    "Filename had no recognizable extension; recovered MIME from byte signature",
                    filename=filename,
                    sniffed_mime=sniffed,
                )
                mime_type = sniffed

        mime_category = mime_type.split("/")[0] if (mime_type and "/" in mime_type) else None

        if self.docling is not None and (mime_type == "application/pdf" or mime_category == "image"):
            return await self.docling.process_file(request=request, file=file)

        if (
            mime_type == "application/pdf"
            or mime_category == "text"
            or mime_type in PYPDF_TEXT_LIKE_APPLICATION_MIME_TYPES
        ):
            return await self.pypdf.process_file(
                file=file,
                file_id=request.file_id,
                options=request.options,
                chunking_strategy=request.chunking_strategy,
            )

        if mime_type in MARKITDOWN_MIME_TYPES:
            return await self.markitdown.process_file(request=request, file=file)

        raise HTTPException(
            status_code=422,
            detail=(
                f"File type '{mime_type or 'unknown'}' (filename={filename!r}) is not supported. "
                f"Supported types: {SUPPORTED_DESCRIPTION}."
            ),
        )

    async def _read_head(
        self,
        request: ProcessFileRequest,
        file: UploadFile | None,
        n: int,
    ) -> bytes:
        """Return the first ``n`` bytes of the upload without consuming the stream that the
        downstream processor will re-read. For the file_id path we use the Files API to fetch
        the content; the downstream processor will fetch again, which is fine — this branch
        only fires when the filename was missing extension, an exception case."""
        if file is not None:
            head = await file.read(n)
            try:
                await file.seek(0)
            except Exception:
                # Some UploadFile backings raise on seek when the underlying stream is exhausted;
                # the downstream processor handles the case where the file is empty and we'd
                # rather degrade than block sniffing.
                pass
            return head
        if request.file_id is not None and self.files_api is not None:
            try:
                resp = await self.files_api.openai_retrieve_file_content(
                    RetrieveFileContentRequest(file_id=request.file_id)
                )
                return (resp.body or b"")[:n]
            except Exception as e:
                log.debug("Could not fetch content for sniff fallback", file_id=request.file_id, error=str(e))
        return b""

    async def _resolve_filename(self, request: ProcessFileRequest, file: UploadFile | None) -> str:
        if file is not None:
            name: str | None = file.filename
            if name is not None:
                return name
        if request.file_id is not None:
            file_info = await self.files_api.openai_retrieve_file(RetrieveFileRequest(file_id=request.file_id))
            resolved: str = file_info.filename
            return resolved
        return "unknown"

    async def shutdown(self) -> None:
        pass
