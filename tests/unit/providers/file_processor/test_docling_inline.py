# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for the inline docling file processor's image-extraction pipeline."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from ogx.providers.inline.file_processor.docling.config import DoclingFileProcessorConfig
from ogx.providers.inline.file_processor.docling.docling import (
    IMAGE_FILE_IDS_METADATA_KEY,
    DoclingFileProcessor,
)


def _make_pil_image(width: int = 200, height: int = 200, colour: tuple[int, int, int] = (255, 0, 0)) -> Image.Image:
    return Image.new("RGB", (width, height), colour)


def _fake_picture(self_ref: str, image: Image.Image | None) -> SimpleNamespace:
    """Return a SimpleNamespace that mimics a docling PictureItem for our isinstance shim."""
    return SimpleNamespace(self_ref=self_ref, _image=image, get_image=lambda doc: image)


def _fake_doc(items: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(iterate_items=lambda: [(it, 0) for it in items])


class _FakePictureBase:
    """Replacement for docling's PictureItem at isinstance() check sites."""


def _picture(self_ref: str, image: Image.Image | None) -> _FakePictureBase:
    fp = _FakePictureBase()
    fp.self_ref = self_ref  # type: ignore[attr-defined]
    fp.get_image = lambda doc: image  # type: ignore[attr-defined]
    return fp


def _file_obj(file_id: str) -> SimpleNamespace:
    return SimpleNamespace(id=file_id)


class TestCollectChunkImageFileIds:
    """Pure unit tests for the static helper that maps chunk doc_items → file_ids."""

    def test_returns_empty_when_no_picture_map(self) -> None:
        chunk = SimpleNamespace(meta=SimpleNamespace(doc_items=[SimpleNamespace(self_ref="#/pictures/0")]))
        assert DoclingFileProcessor._collect_chunk_image_file_ids(chunk, {}) == []

    def test_returns_empty_when_chunk_has_no_doc_items(self) -> None:
        chunk = SimpleNamespace(meta=None)
        assert DoclingFileProcessor._collect_chunk_image_file_ids(chunk, {"#/pictures/0": "file-1"}) == []

    def test_collects_in_order_of_appearance(self) -> None:
        chunk = SimpleNamespace(
            meta=SimpleNamespace(
                doc_items=[
                    SimpleNamespace(self_ref="#/texts/0"),
                    SimpleNamespace(self_ref="#/pictures/1"),
                    SimpleNamespace(self_ref="#/texts/1"),
                    SimpleNamespace(self_ref="#/pictures/0"),
                ]
            )
        )
        picture_map = {"#/pictures/0": "file-a", "#/pictures/1": "file-b"}
        assert DoclingFileProcessor._collect_chunk_image_file_ids(chunk, picture_map) == ["file-b", "file-a"]

    def test_dedupes_repeated_pictures(self) -> None:
        chunk = SimpleNamespace(
            meta=SimpleNamespace(
                doc_items=[
                    SimpleNamespace(self_ref="#/pictures/0"),
                    SimpleNamespace(self_ref="#/pictures/0"),
                    SimpleNamespace(self_ref="#/pictures/1"),
                ]
            )
        )
        picture_map = {"#/pictures/0": "file-a", "#/pictures/1": "file-b"}
        assert DoclingFileProcessor._collect_chunk_image_file_ids(chunk, picture_map) == ["file-a", "file-b"]

    def test_skips_doc_items_with_no_self_ref(self) -> None:
        chunk = SimpleNamespace(
            meta=SimpleNamespace(
                doc_items=[
                    SimpleNamespace(self_ref=None),
                    SimpleNamespace(),  # no self_ref attribute at all
                    SimpleNamespace(self_ref="#/pictures/0"),
                ]
            )
        )
        assert DoclingFileProcessor._collect_chunk_image_file_ids(chunk, {"#/pictures/0": "file-a"}) == ["file-a"]


class TestExtractAndUploadPictures:
    """Behavioural tests for the image extraction + upload phase."""

    @pytest.fixture
    def config(self) -> DoclingFileProcessorConfig:
        return DoclingFileProcessorConfig(
            extract_images=True,
            images_scale=2.0,
            min_image_dim_px=64,
        )

    @pytest.fixture
    def files_api(self) -> AsyncMock:
        api = AsyncMock()
        # Each call returns a unique-ish file id keyed off the upload count
        api._uploads = 0

        async def _upload(request, file):  # type: ignore[no-untyped-def]
            api._uploads += 1
            return _file_obj(f"file-{api._uploads:03d}")

        api.openai_upload_file.side_effect = _upload
        return api

    @pytest.mark.asyncio
    async def test_returns_empty_when_extract_images_disabled(self, files_api: AsyncMock) -> None:
        cfg = DoclingFileProcessorConfig(extract_images=False)
        processor = DoclingFileProcessor(cfg, files_api=files_api)
        doc = _fake_doc([_picture("#/pictures/0", _make_pil_image())])

        result = await processor._extract_and_upload_pictures(doc, filename="x.pdf")

        assert result == {}
        files_api.openai_upload_file.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_returns_empty_when_files_api_missing(self, config: DoclingFileProcessorConfig) -> None:
        processor = DoclingFileProcessor(config, files_api=None)
        doc = _fake_doc([_picture("#/pictures/0", _make_pil_image())])

        result = await processor._extract_and_upload_pictures(doc, filename="x.pdf")

        assert result == {}

    @pytest.mark.asyncio
    async def test_uploads_each_picture_and_returns_map(
        self, config: DoclingFileProcessorConfig, files_api: AsyncMock
    ) -> None:
        processor = DoclingFileProcessor(config, files_api=files_api)
        items = [
            _picture("#/pictures/0", _make_pil_image()),
            _picture("#/pictures/1", _make_pil_image()),
        ]
        doc = _fake_doc(items)

        with patch(
            "ogx.providers.inline.file_processor.docling.docling.PictureItem",
            _FakePictureBase,
        ):
            result = await processor._extract_and_upload_pictures(doc, filename="report.pdf")

        assert result == {"#/pictures/0": "file-001", "#/pictures/1": "file-002"}
        assert files_api.openai_upload_file.await_count == 2

    @pytest.mark.asyncio
    async def test_skips_pictures_below_min_dim(self, config: DoclingFileProcessorConfig, files_api: AsyncMock) -> None:
        processor = DoclingFileProcessor(config, files_api=files_api)
        items = [
            _picture("#/pictures/small", _make_pil_image(width=32, height=32)),
            _picture("#/pictures/big", _make_pil_image(width=200, height=200)),
        ]
        doc = _fake_doc(items)

        with patch(
            "ogx.providers.inline.file_processor.docling.docling.PictureItem",
            _FakePictureBase,
        ):
            result = await processor._extract_and_upload_pictures(doc, filename="report.pdf")

        assert result == {"#/pictures/big": "file-001"}
        assert files_api.openai_upload_file.await_count == 1

    @pytest.mark.asyncio
    async def test_continues_when_individual_upload_fails(self, config: DoclingFileProcessorConfig) -> None:
        files_api = AsyncMock()
        files_api._uploads = 0

        async def _upload(request, file):  # type: ignore[no-untyped-def]
            files_api._uploads += 1
            if files_api._uploads == 1:
                raise RuntimeError("S3 unavailable")
            return _file_obj(f"file-{files_api._uploads:03d}")

        files_api.openai_upload_file.side_effect = _upload
        processor = DoclingFileProcessor(config, files_api=files_api)
        items = [
            _picture("#/pictures/0", _make_pil_image()),
            _picture("#/pictures/1", _make_pil_image()),
        ]
        doc = _fake_doc(items)

        with patch(
            "ogx.providers.inline.file_processor.docling.docling.PictureItem",
            _FakePictureBase,
        ):
            result = await processor._extract_and_upload_pictures(doc, filename="report.pdf")

        # First upload fails, second succeeds. Only the surviving picture appears in the map.
        assert result == {"#/pictures/1": "file-002"}

    @pytest.mark.asyncio
    async def test_skips_pictures_with_no_renderable_image(
        self, config: DoclingFileProcessorConfig, files_api: AsyncMock
    ) -> None:
        processor = DoclingFileProcessor(config, files_api=files_api)
        items = [
            _picture("#/pictures/0", None),
            _picture("#/pictures/1", _make_pil_image()),
        ]
        doc = _fake_doc(items)

        with patch(
            "ogx.providers.inline.file_processor.docling.docling.PictureItem",
            _FakePictureBase,
        ):
            result = await processor._extract_and_upload_pictures(doc, filename="report.pdf")

        assert result == {"#/pictures/1": "file-001"}
        assert files_api.openai_upload_file.await_count == 1

    @pytest.mark.asyncio
    async def test_uploaded_filename_contains_picture_ref(
        self, config: DoclingFileProcessorConfig, files_api: AsyncMock
    ) -> None:
        processor = DoclingFileProcessor(config, files_api=files_api)
        doc = _fake_doc([_picture("#/pictures/3", _make_pil_image())])

        with patch(
            "ogx.providers.inline.file_processor.docling.docling.PictureItem",
            _FakePictureBase,
        ):
            await processor._extract_and_upload_pictures(doc, filename="annual_report.pdf")

        call = files_api.openai_upload_file.await_args
        assert call is not None
        upload_file = call.kwargs.get("file") or call.args[1]
        assert upload_file.filename == "annual_report_pictures_3.png"


class TestCreateChunksNoStrategy:
    """The no-chunking-strategy path attaches every extracted file_id to the single chunk."""

    def test_picture_ids_land_on_single_chunk(self) -> None:
        cfg = DoclingFileProcessorConfig(extract_images=True)
        processor = DoclingFileProcessor(cfg, files_api=AsyncMock())
        doc = SimpleNamespace(export_to_markdown=lambda: "Hello world.")

        chunks = processor._create_chunks(
            doc=doc,
            document_id="doc-1",
            chunking_strategy=None,
            document_metadata={"filename": "x.pdf"},
            picture_file_ids={"#/pictures/0": "file-a", "#/pictures/1": "file-b"},
        )

        assert len(chunks) == 1
        assert chunks[0].metadata[IMAGE_FILE_IDS_METADATA_KEY] == ["file-a", "file-b"]
        assert chunks[0].metadata["filename"] == "x.pdf"
        assert chunks[0].metadata["document_id"] == "doc-1"

    def test_no_image_key_when_picture_map_empty(self) -> None:
        cfg = DoclingFileProcessorConfig(extract_images=True)
        processor = DoclingFileProcessor(cfg, files_api=AsyncMock())
        doc = SimpleNamespace(export_to_markdown=lambda: "Hello world.")

        chunks = processor._create_chunks(
            doc=doc,
            document_id="doc-1",
            chunking_strategy=None,
            document_metadata={"filename": "x.pdf"},
            picture_file_ids={},
        )

        assert len(chunks) == 1
        assert IMAGE_FILE_IDS_METADATA_KEY not in chunks[0].metadata

    def test_returns_empty_for_blank_doc(self) -> None:
        cfg = DoclingFileProcessorConfig(extract_images=True)
        processor = DoclingFileProcessor(cfg, files_api=AsyncMock())
        doc = SimpleNamespace(export_to_markdown=lambda: "   ")

        chunks = processor._create_chunks(
            doc=doc,
            document_id="doc-1",
            chunking_strategy=None,
            document_metadata={"filename": "x.pdf"},
            picture_file_ids={"#/pictures/0": "file-a"},
        )

        assert chunks == []


# NOTE: _build_converter wiring is intentionally not unit-tested here. It does lazy imports of
# the heavy `docling` package and constructs DocumentConverter / PdfFormatOption with
# images_scale and generate_picture_images=True. Mocking that import chain without installing
# docling proper (which pulls torch / huggingface) would obscure rather than verify behaviour;
# the wiring is exercised end-to-end by the integration test suite under tests/integration.
