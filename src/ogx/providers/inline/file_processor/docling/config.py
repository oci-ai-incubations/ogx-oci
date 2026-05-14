# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from ogx_api.vector_io import VectorStoreChunkingStrategyStaticConfig


class DoclingFileProcessorConfig(BaseModel):
    """Configuration for Docling file processor."""

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
    extract_images: bool = Field(
        default=True,
        description=(
            "Whether to extract embedded images (figures/pictures) from the source document, "
            "upload each via the Files API, and attach the resulting file_ids to the owning "
            "chunk's metadata under 'image_file_ids'. Requires a Files API binding."
        ),
    )
    images_scale: float = Field(
        default=2.0,
        ge=1.0,
        le=4.0,
        description=(
            "Render scale passed to docling's PDF pipeline for picture/figure rasterisation. "
            "Higher values yield sharper extracted images at the cost of more processing time "
            "and larger uploaded file sizes."
        ),
    )
    min_image_dim_px: int = Field(
        default=64,
        ge=1,
        le=4096,
        description=(
            "Images with width or height below this threshold (in pixels) are skipped during "
            "extraction. Filters out spurious icons, bullet points, and decorative artefacts."
        ),
    )

    @classmethod
    def sample_run_config(cls, **kwargs: Any) -> dict[str, Any]:
        return {
            "default_chunk_size_tokens": 800,
            "default_chunk_overlap_tokens": 400,
            "extract_images": True,
            "images_scale": 2.0,
            "min_image_dim_px": 64,
        }
