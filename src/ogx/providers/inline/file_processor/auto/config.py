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
    MIME type. It always includes PyPDF for PDF and text files. When a supported
    document-conversion backend is available, it routes office formats (DOCX,
    PPTX, XLSX, HTML) there instead of rejecting them.
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
    docling_extract_images: bool = Field(
        default=True,
        description=(
            "Only consulted when prefer_docling_for_pdfs is True. Whether the docling backend "
            "should extract and upload embedded pictures to the Files API. See "
            "DoclingFileProcessorConfig.extract_images for details."
        ),
    )
    docling_images_scale: float = Field(
        default=2.0,
        ge=1.0,
        le=4.0,
        description=(
            "Only consulted when prefer_docling_for_pdfs is True. Render scale passed to "
            "docling's PDF pipeline for picture rasterisation."
        ),
    )
    docling_min_image_dim_px: int = Field(
        default=64,
        ge=1,
        le=4096,
        description=(
            "Only consulted when prefer_docling_for_pdfs is True. Images smaller than this "
            "width or height (in pixels) are skipped during extraction."
        ),
    )

    @classmethod
    def sample_run_config(cls, **kwargs: Any) -> dict[str, Any]:
        return {}
