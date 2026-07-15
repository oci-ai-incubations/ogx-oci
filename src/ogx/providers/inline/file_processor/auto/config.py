# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from ogx_api.vector_io import VectorStoreChunkingStrategyStaticConfig


class AutoFileProcessorConfig(BaseModel):
    """Configuration for the auto file processor.

    The auto file processor dispatches to the appropriate backend based on file
    MIME type. When configured with a ``priority`` list, it interrogates sibling
    file processor providers to build a MIME type dispatch map. The first provider
    in the priority list that supports a given MIME type wins.

    When no ``priority`` is configured, it falls back to built-in PyPDF (PDF and
    text files) and MarkItDown (office, media) backends. This legacy behavior is
    deprecated and will be removed in a future release.
    """

    default_chunk_size_tokens: int = Field(
        default=VectorStoreChunkingStrategyStaticConfig.model_fields["max_chunk_size_tokens"].default,
        ge=100,
        le=4096,
        description="Default chunk size in tokens when chunking_strategy type is 'auto'",
    )
    default_chunk_overlap_tokens: int = Field(
        default=VectorStoreChunkingStrategyStaticConfig.model_fields["chunk_overlap_tokens"].default,
        ge=0,
        le=2048,
        description="Default chunk overlap in tokens when chunking_strategy type is 'auto'",
    )

    extract_metadata: bool = Field(default=True, description="Whether to extract PDF metadata (title, author, etc.)")

    clean_text: bool = Field(
        default=True, description="Whether to clean extracted text (remove extra whitespace, normalize line breaks)"
    )

    prefer_docling_for_pdfs: bool = Field(
        default=False,
        description=(
            "When True, PDFs are processed by the inline docling backend instead of PyPDF. "
            "Docling is structure-aware and natively extracts embedded pictures (writing "
            "their file_ids onto chunk.metadata['image_file_ids']) — the foundation for "
            "multimodal RAG. Trade-offs vs PyPDF: significantly slower ingest, larger memory "
            "footprint, and structurally-different chunk shape. Other MIME types are unaffected."
        ),
    )
    default_extract_images: bool = Field(
        default=True,
        description=(
            "Only consulted when prefer_docling_for_pdfs is True. Whether the docling backend "
            "should extract and upload embedded pictures to the Files API. See "
            "DoclingFileProcessorConfig.extract_images for details."
        ),
    )
    default_images_scale: float = Field(
        default=2.0,
        ge=1.0,
        le=4.0,
        description=(
            "Only consulted when prefer_docling_for_pdfs is True. Render scale passed to "
            "docling's PDF pipeline for picture rasterisation."
        ),
    )
    default_min_image_dim_px: int = Field(
        default=64,
        ge=1,
        le=4096,
        description=(
            "Only consulted when prefer_docling_for_pdfs is True. Images smaller than this "
            "width or height (in pixels) are skipped during extraction."
        ),
    )
    default_caption_images: bool = Field(
        default=False,
        description=(
            "Only consulted when prefer_docling_for_pdfs is True. When enabled, every "
            "extracted picture is captioned by a vision-capable model (default_caption_model). "
            "The caption is what makes standalone-image uploads (PNG/JPG) retrievable via "
            "semantic search; without it those chunks carry only filename text. Each picture "
            "incurs one vision-model call at ingest time."
        ),
    )
    default_caption_model: str | None = Field(
        default=None,
        description=(
            "Only consulted when default_caption_images is True. Model identifier passed to "
            "the Inference API. Must be a vision-capable model; example: "
            "'oci/meta.llama-3.2-90b-vision-instruct'."
        ),
    )
    default_caption_prompt: str | None = Field(
        default=None,
        description=(
            "Override the caption-generation prompt sent to the vision model. Leave unset (None) "
            "to use docling's built-in default prompt — useful for tuning toward text-in-image "
            "OCR, scene description, or technical-diagram extraction without forking the "
            "downstream DoclingFileProcessorConfig default."
        ),
    )
    default_caption_max_tokens: int = Field(
        default=180,
        ge=16,
        le=2048,
        description=(
            "Upper bound on caption length in tokens. Captions are embedded as chunk text and "
            "longer captions dilute retrieval signal, so the default is intentionally small."
        ),
    )
    priority: list[str] = Field(
        default_factory=list,
        description=(
            "Ordered list of sibling provider IDs to dispatch to. Each provider "
            "declares the MIME types it supports; the first provider in the list "
            "that supports a given MIME type handles it."
        ),
    )

    @classmethod
    def sample_run_config(cls, **kwargs: Any) -> dict[str, Any]:
        return {}
