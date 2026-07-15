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
from ogx.providers.inline.file_processor.markitdown.markitdown_processor import (
    MARKITDOWN_MIME_TYPES,
    MarkItDownFileProcessor,
)
from ogx.providers.inline.file_processor.pypdf.config import PyPDFFileProcessorConfig
from ogx.providers.inline.file_processor.pypdf.pypdf import PYPDF_TEXT_LIKE_APPLICATION_MIME_TYPES, PyPDFFileProcessor
from ogx_api.file_processors import ProcessFileRequest, ProcessFileResponse
from ogx_api.files import RetrieveFileRequest

from .config import AutoFileProcessorConfig

log = get_logger(name=__name__, category="providers::file_processors")

# Built from the actual allowlists so the user-facing description can't drift
# from what the router will accept. "text/*" covers txt, csv, md, html, and most
# source-code extensions that mimetypes maps under text/*.
SUPPORTED_DESCRIPTION = ", ".join(
    ["text/*", *sorted({"application/pdf", *PYPDF_TEXT_LIKE_APPLICATION_MIME_TYPES, *MARKITDOWN_MIME_TYPES})]
)


class AutoFileProcessor:
    """Composite file processor that dispatches to backends based on MIME type.

    When ``priority`` is configured, interrogates sibling providers to build a
    dispatch map. Each provider declares which MIME types it supports via
    ``supported_mime_types()``. The first provider in the priority list that
    supports a given MIME type handles it.

    When no ``priority`` is configured, falls back to built-in PyPDF (PDF/text)
    and MarkItDown (office/media) backends. This legacy behavior is deprecated
    and will be removed in a future release.
    """

    def __init__(self, config: AutoFileProcessorConfig, files_api, inference_api=None) -> None:
        self.config = config
        self.files_api = files_api
        self.inference_api = inference_api

        self.dispatch_map: dict[str, Any] = {}
        self.category_map: dict[str, Any] = {}
        self._using_priority = False

        self._init_legacy_backends(config, files_api)

    def set_sibling_providers(self, siblings: dict[str, Any]) -> None:
        if not siblings:
            return

        provider_order = self.config.priority if self.config.priority else list(siblings.keys())
        self._build_dispatch_map(siblings, provider_order)
        self._using_priority = True

    def _init_legacy_backends(self, config: AutoFileProcessorConfig, files_api: Any) -> None:
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
            self.docling = DoclingFileProcessor(docling_config, files_api=files_api, inference_api=self.inference_api)

    def _build_dispatch_map(self, providers: dict[str, Any], provider_order: list[str]) -> None:
        for provider_id in provider_order:
            if provider_id not in providers:
                raise ValueError(
                    f"Failed to resolve priority entry '{provider_id}': "
                    f"no sibling provider with that ID is configured. "
                    f"Available providers: {', '.join(sorted(providers.keys()))}"
                )

            provider = providers[provider_id]

            if not hasattr(provider, "supported_mime_types"):
                log.warning(
                    "Provider does not implement supported_mime_types, skipping",
                    provider_id=provider_id,
                )
                continue

            mime_types = provider.supported_mime_types()

            if mime_types is None:
                log.warning(
                    "Provider returned None from supported_mime_types, skipping",
                    provider_id=provider_id,
                )
                continue

            for mime in mime_types:
                if mime.endswith("/*"):
                    category = mime.split("/")[0]
                    if category not in self.category_map:
                        self.category_map[category] = provider
                elif mime not in self.dispatch_map:
                    self.dispatch_map[mime] = provider

            log.info("Provider registered", provider_id=provider_id, mime_type_count=len(mime_types))

    async def process_file(
        self,
        request: ProcessFileRequest,
        file: UploadFile | None = None,
    ) -> ProcessFileResponse:
        filename = await self._resolve_filename(request, file)
        mime_type, _ = mimetypes.guess_type(filename)
        mime_category = mime_type.split("/")[0] if (mime_type and "/" in mime_type) else None

        if self._using_priority:
            return await self._dispatch_priority(request, file, mime_type, mime_category)
        return await self._dispatch_legacy(request, file, mime_type, mime_category)

    async def _dispatch_priority(
        self,
        request: ProcessFileRequest,
        file: UploadFile | None,
        mime_type: str | None,
        mime_category: str | None,
    ) -> ProcessFileResponse:
        if mime_type and mime_type in self.dispatch_map:
            result: ProcessFileResponse = await self.dispatch_map[mime_type].process_file(request=request, file=file)
            return result

        if mime_category and mime_category in self.category_map:
            result = await self.category_map[mime_category].process_file(request=request, file=file)
            return result

        raise HTTPException(
            status_code=422,
            detail=f"File type '{mime_type or 'unknown'}' is not supported by any configured provider.",
        )

    async def _dispatch_legacy(
        self,
        request: ProcessFileRequest,
        file: UploadFile | None,
        mime_type: str | None,
        mime_category: str | None,
    ) -> ProcessFileResponse:
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
            detail=f"File type '{mime_type or 'unknown'}' is not supported. Supported types: {SUPPORTED_DESCRIPTION}.",
        )

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
