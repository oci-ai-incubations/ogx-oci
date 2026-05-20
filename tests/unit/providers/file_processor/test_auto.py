# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, UploadFile

from ogx.providers.inline.file_processor.auto.auto import AutoFileProcessor
from ogx.providers.inline.file_processor.auto.config import AutoFileProcessorConfig
from ogx_api.file_processors import ProcessFileRequest


@pytest.fixture
def auto_processor():
    config = AutoFileProcessorConfig()
    files_api = MagicMock()
    return AutoFileProcessor(config, files_api)


@pytest.fixture
def auto_processor_with_files_api():
    config = AutoFileProcessorConfig()
    files_api = MagicMock()
    file_info = MagicMock()
    file_info.filename = "document.txt"
    files_api.openai_retrieve_file = AsyncMock(return_value=file_info)

    content_response = MagicMock()
    content_response.body = b"Hello from file storage."
    files_api.openai_retrieve_file_content = AsyncMock(return_value=content_response)

    return AutoFileProcessor(config, files_api)


async def test_routes_pdf_to_pypdf(auto_processor):
    pdf_bytes = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\nxref\n0 3\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \ntrailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n115\n%%EOF"
    file = UploadFile(filename="test.pdf", file=io.BytesIO(pdf_bytes))
    request = ProcessFileRequest()

    result = await auto_processor.process_file(request, file=file)
    assert result is not None


async def test_routes_text_to_pypdf(auto_processor):
    text_bytes = b"Hello, this is plain text."
    file = UploadFile(filename="readme.txt", file=io.BytesIO(text_bytes))
    request = ProcessFileRequest()

    result = await auto_processor.process_file(request, file=file)
    assert result is not None
    assert len(result.chunks) >= 1


async def test_routes_csv_to_pypdf(auto_processor):
    csv_bytes = b"name,age\nAlice,30\nBob,25"
    file = UploadFile(filename="data.csv", file=io.BytesIO(csv_bytes))
    request = ProcessFileRequest()

    result = await auto_processor.process_file(request, file=file)
    assert result is not None
    assert len(result.chunks) >= 1


async def test_routes_markdown_to_pypdf(auto_processor):
    md_bytes = b"# Hello\n\nThis is markdown."
    file = UploadFile(filename="README.md", file=io.BytesIO(md_bytes))
    request = ProcessFileRequest()

    result = await auto_processor.process_file(request, file=file)
    assert result is not None
    assert len(result.chunks) >= 1


async def test_routes_docx_to_markitdown(auto_processor):
    docx_bytes = b"PK\x03\x04fake_docx_content"
    file = UploadFile(filename="test.docx", file=io.BytesIO(docx_bytes))
    request = ProcessFileRequest()

    with pytest.raises(HTTPException) as exc_info:
        await auto_processor.process_file(request, file=file)

    assert exc_info.value.status_code == 422
    assert "Failed to process file" in exc_info.value.detail


async def test_routes_pptx_to_markitdown(auto_processor):
    pptx_bytes = b"PK\x03\x04fake_pptx_content"
    file = UploadFile(filename="presentation.pptx", file=io.BytesIO(pptx_bytes))
    request = ProcessFileRequest()

    with pytest.raises(HTTPException) as exc_info:
        await auto_processor.process_file(request, file=file)

    assert exc_info.value.status_code == 422
    assert "Failed to process file" in exc_info.value.detail


async def test_routes_xlsx_to_markitdown(auto_processor):
    xlsx_bytes = b"PK\x03\x04fake_xlsx_content"
    file = UploadFile(filename="data.xlsx", file=io.BytesIO(xlsx_bytes))
    request = ProcessFileRequest()

    result = await auto_processor.process_file(request, file=file)
    assert result is not None
    assert result.metadata["processor"] == "markitdown"


async def test_rejects_unsupported_format_with_422(auto_processor):
    file = UploadFile(filename="test.xyz", file=io.BytesIO(b"some data"))
    request = ProcessFileRequest()

    with pytest.raises(HTTPException) as exc_info:
        await auto_processor.process_file(request, file=file)

    assert exc_info.value.status_code == 422
    detail = exc_info.value.detail.lower()
    assert "not supported" in detail
    assert "pdf" in detail


async def test_routes_json_to_pypdf(auto_processor):
    """Regression: application/json was rejected even though the error message
    advertised json as supported (allowlist and description had drifted)."""
    json_bytes = b'{"name": "Alice", "age": 30}'
    file = UploadFile(filename="data.json", file=io.BytesIO(json_bytes))
    request = ProcessFileRequest()

    result = await auto_processor.process_file(request, file=file)
    assert result is not None
    assert len(result.chunks) >= 1


async def test_routes_xml_to_pypdf(auto_processor):
    xml_bytes = b"<?xml version='1.0'?><root><item>hello</item></root>"
    file = UploadFile(filename="data.xml", file=io.BytesIO(xml_bytes))
    request = ProcessFileRequest()

    result = await auto_processor.process_file(request, file=file)
    assert result is not None
    assert len(result.chunks) >= 1


def test_supported_description_lists_every_allowlisted_type():
    """The human-readable description must be derived from the allowlists so the
    two can't disagree — the bug this guards against was the description
    advertising json/xml as supported while the allowlist omitted them."""
    from ogx.providers.inline.file_processor.auto.auto import (
        MARKITDOWN_MIME_TYPES,
        SUPPORTED_DESCRIPTION,
    )
    from ogx.providers.inline.file_processor.pypdf.pypdf import (
        PYPDF_TEXT_LIKE_APPLICATION_MIME_TYPES,
    )

    expected = {"application/pdf", *PYPDF_TEXT_LIKE_APPLICATION_MIME_TYPES, *MARKITDOWN_MIME_TYPES}
    for mime in expected:
        assert mime in SUPPORTED_DESCRIPTION, f"{mime} missing from SUPPORTED_DESCRIPTION"


async def test_routes_file_id_using_resolved_filename(auto_processor_with_files_api):
    request = ProcessFileRequest(file_id="file-123456")

    result = await auto_processor_with_files_api.process_file(request)
    assert result is not None
    assert len(result.chunks) >= 1


async def test_docling_backend_not_constructed_when_flag_off():
    config = AutoFileProcessorConfig()
    files_api = MagicMock()
    proc = AutoFileProcessor(config, files_api)
    assert proc.docling is None


async def test_prefer_docling_for_pdfs_constructs_docling_backend():
    config = AutoFileProcessorConfig(
        prefer_docling_for_pdfs=True,
        default_extract_images=True,
        default_images_scale=2.5,
        default_min_image_dim_px=128,
    )
    files_api = MagicMock()
    proc = AutoFileProcessor(config, files_api)

    assert proc.docling is not None
    assert proc.docling.config.extract_images is True
    assert proc.docling.config.images_scale == 2.5
    assert proc.docling.config.min_image_dim_px == 128
    # Same files_api instance gets threaded through
    assert proc.docling.files_api is files_api


async def test_pdf_routes_to_docling_when_flag_on():
    config = AutoFileProcessorConfig(prefer_docling_for_pdfs=True)
    files_api = MagicMock()
    proc = AutoFileProcessor(config, files_api)

    sentinel_response = MagicMock()
    proc.docling.process_file = AsyncMock(return_value=sentinel_response)
    proc.pypdf.process_file = AsyncMock(return_value=MagicMock())  # should not be called

    file = UploadFile(filename="test.pdf", file=io.BytesIO(b"%PDF-1.4 minimal"))
    request = ProcessFileRequest()

    result = await proc.process_file(request, file=file)

    assert result is sentinel_response
    proc.docling.process_file.assert_awaited_once()
    proc.pypdf.process_file.assert_not_awaited()


async def test_text_files_still_route_to_pypdf_when_docling_flag_on():
    config = AutoFileProcessorConfig(prefer_docling_for_pdfs=True)
    files_api = MagicMock()
    proc = AutoFileProcessor(config, files_api)

    sentinel_response = MagicMock()
    proc.pypdf.process_file = AsyncMock(return_value=sentinel_response)
    proc.docling.process_file = AsyncMock(return_value=MagicMock())  # should not be called

    file = UploadFile(filename="notes.txt", file=io.BytesIO(b"plain text"))
    request = ProcessFileRequest()

    result = await proc.process_file(request, file=file)

    assert result is sentinel_response
    proc.pypdf.process_file.assert_awaited_once()
    proc.docling.process_file.assert_not_awaited()


async def test_image_routes_to_docling_when_flag_on():
    config = AutoFileProcessorConfig(prefer_docling_for_pdfs=True)
    proc = AutoFileProcessor(config, MagicMock())

    sentinel = MagicMock()
    proc.docling.process_file = AsyncMock(return_value=sentinel)
    proc.markitdown.process_file = AsyncMock(return_value=MagicMock())  # should not be called

    file = UploadFile(filename="photo.jpg", file=io.BytesIO(b"\xff\xd8\xff"))  # JPEG SOI
    result = await proc.process_file(ProcessFileRequest(), file=file)

    assert result is sentinel
    proc.docling.process_file.assert_awaited_once()
    proc.markitdown.process_file.assert_not_awaited()


async def test_image_still_goes_to_markitdown_when_flag_off():
    # Existing deployments that haven't opted in get the previous behaviour for images.
    proc = AutoFileProcessor(AutoFileProcessorConfig(), MagicMock())
    sentinel = MagicMock()
    proc.markitdown.process_file = AsyncMock(return_value=sentinel)

    file = UploadFile(filename="photo.png", file=io.BytesIO(b"\x89PNG\r\n\x1a\n"))
    result = await proc.process_file(ProcessFileRequest(), file=file)

    assert result is sentinel
    proc.markitdown.process_file.assert_awaited_once()


async def test_inference_api_threaded_into_docling_when_caption_enabled():
    config = AutoFileProcessorConfig(
        prefer_docling_for_pdfs=True,
        default_caption_images=True,
        default_caption_model="vl-model",
    )
    files_api = MagicMock()
    inference_api = MagicMock()

    proc = AutoFileProcessor(config, files_api, inference_api=inference_api)

    assert proc.docling is not None
    assert proc.docling.inference_api is inference_api
    assert proc.docling.config.caption_images is True
    assert proc.docling.config.caption_model == "vl-model"


async def test_caption_max_tokens_threaded_into_docling():
    """default_caption_max_tokens overrides docling's caption_max_tokens default."""
    config = AutoFileProcessorConfig(
        prefer_docling_for_pdfs=True,
        default_caption_images=True,
        default_caption_model="vl-model",
        default_caption_max_tokens=64,
    )
    proc = AutoFileProcessor(config, MagicMock(), inference_api=MagicMock())

    assert proc.docling is not None
    assert proc.docling.config.caption_max_tokens == 64


async def test_caption_prompt_falls_back_to_docling_default_when_unset():
    """An unset default_caption_prompt must preserve docling's built-in default rather than
    overwriting it with None — otherwise captions would be generated with no prompt."""
    from ogx.providers.inline.file_processor.docling.config import DoclingFileProcessorConfig

    config_unset = AutoFileProcessorConfig(prefer_docling_for_pdfs=True)
    proc_unset = AutoFileProcessor(config_unset, MagicMock())
    assert proc_unset.docling is not None
    docling_default = DoclingFileProcessorConfig.model_fields["caption_prompt"].default
    assert proc_unset.docling.config.caption_prompt == docling_default

    # When explicitly overridden, the user's prompt wins.
    config_set = AutoFileProcessorConfig(
        prefer_docling_for_pdfs=True,
        default_caption_prompt="Read every label on this part. Return one short line.",
    )
    proc_set = AutoFileProcessor(config_set, MagicMock())
    assert proc_set.docling is not None
    assert proc_set.docling.config.caption_prompt == "Read every label on this part. Return one short line."
