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
    ExtractedPicture,
)


def _make_pil_image(width: int = 200, height: int = 200, colour: tuple[int, int, int] = (255, 0, 0)) -> Image.Image:
    return Image.new("RGB", (width, height), colour)


def _fake_doc(items: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(iterate_items=lambda: [(it, 0) for it in items])


class _FakePictureBase:
    """Replacement for docling's PictureItem at isinstance() check sites."""


def _picture(self_ref: str, image: Image.Image | None, pages: tuple[int, ...] = (1,)) -> _FakePictureBase:
    fp = _FakePictureBase()
    fp.self_ref = self_ref  # type: ignore[attr-defined]
    fp.get_image = lambda doc: image  # type: ignore[attr-defined]
    fp.prov = [SimpleNamespace(page_no=p) for p in pages]  # type: ignore[attr-defined]
    return fp


def _file_obj(file_id: str) -> SimpleNamespace:
    return SimpleNamespace(id=file_id)


def _chunk_on_pages(*pages: int) -> SimpleNamespace:
    """Build a fake chunk whose text doc_items together span the given pages."""
    doc_items = [
        SimpleNamespace(self_ref=f"#/texts/{i}", prov=[SimpleNamespace(page_no=p)]) for i, p in enumerate(pages)
    ]
    return SimpleNamespace(meta=SimpleNamespace(doc_items=doc_items))


class TestCollectChunkImageFileIds:
    """Page-based matching: chunks pick up pictures whose page set overlaps the chunk's text."""

    def test_returns_empty_when_no_pictures(self) -> None:
        chunk = _chunk_on_pages(3)
        assert DoclingFileProcessor._collect_chunk_image_file_ids(chunk, []) == []

    def test_returns_empty_when_chunk_has_no_doc_items(self) -> None:
        chunk = SimpleNamespace(meta=None)
        pictures = [ExtractedPicture("#/pictures/0", "file-1", frozenset({1}))]
        assert DoclingFileProcessor._collect_chunk_image_file_ids(chunk, pictures) == []

    def test_returns_empty_when_chunk_pages_are_unknown(self) -> None:
        # doc_items present but every prov is empty — no pages can be inferred
        chunk = SimpleNamespace(meta=SimpleNamespace(doc_items=[SimpleNamespace(self_ref="#/texts/0", prov=[])]))
        pictures = [ExtractedPicture("#/pictures/0", "file-1", frozenset({1}))]
        assert DoclingFileProcessor._collect_chunk_image_file_ids(chunk, pictures) == []

    def test_matches_picture_on_overlapping_page(self) -> None:
        chunk = _chunk_on_pages(3)
        pictures = [
            ExtractedPicture("#/pictures/0", "file-a", frozenset({1})),
            ExtractedPicture("#/pictures/1", "file-b", frozenset({3})),
            ExtractedPicture("#/pictures/2", "file-c", frozenset({5})),
        ]
        assert DoclingFileProcessor._collect_chunk_image_file_ids(chunk, pictures) == ["file-b"]

    def test_preserves_document_iteration_order(self) -> None:
        # Pictures appear in iteration order in the input list; chunk pages happen to include both
        chunk = _chunk_on_pages(2, 3)
        pictures = [
            ExtractedPicture("#/pictures/0", "file-a", frozenset({3})),
            ExtractedPicture("#/pictures/1", "file-b", frozenset({2})),
        ]
        assert DoclingFileProcessor._collect_chunk_image_file_ids(chunk, pictures) == ["file-a", "file-b"]

    def test_chunk_spanning_multiple_pages_picks_all_matching(self) -> None:
        chunk = _chunk_on_pages(8, 9)
        pictures = [
            ExtractedPicture("#/pictures/0", "file-a", frozenset({8})),
            ExtractedPicture("#/pictures/1", "file-b", frozenset({9})),
            ExtractedPicture("#/pictures/2", "file-c", frozenset({9})),
            ExtractedPicture("#/pictures/3", "file-d", frozenset({10})),
        ]
        assert DoclingFileProcessor._collect_chunk_image_file_ids(chunk, pictures) == ["file-a", "file-b", "file-c"]

    def test_picture_spanning_multiple_pages_matches_any_overlap(self) -> None:
        chunk = _chunk_on_pages(5)
        pictures = [ExtractedPicture("#/pictures/0", "file-a", frozenset({4, 5}))]
        assert DoclingFileProcessor._collect_chunk_image_file_ids(chunk, pictures) == ["file-a"]


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

    async def test_returns_empty_when_extract_images_disabled(self, files_api: AsyncMock) -> None:
        cfg = DoclingFileProcessorConfig(extract_images=False)
        processor = DoclingFileProcessor(cfg, files_api=files_api)
        doc = _fake_doc([_picture("#/pictures/0", _make_pil_image())])

        result = await processor._extract_and_upload_pictures(doc, filename="x.pdf")

        assert result == []
        files_api.openai_upload_file.assert_not_awaited()

    async def test_returns_empty_when_files_api_missing(self, config: DoclingFileProcessorConfig) -> None:
        processor = DoclingFileProcessor(config, files_api=None)
        doc = _fake_doc([_picture("#/pictures/0", _make_pil_image())])

        result = await processor._extract_and_upload_pictures(doc, filename="x.pdf")

        assert result == []

    async def test_uploads_each_picture_and_returns_list(
        self, config: DoclingFileProcessorConfig, files_api: AsyncMock
    ) -> None:
        processor = DoclingFileProcessor(config, files_api=files_api)
        items = [
            _picture("#/pictures/0", _make_pil_image(), pages=(1,)),
            _picture("#/pictures/1", _make_pil_image(), pages=(2, 3)),
        ]
        doc = _fake_doc(items)

        with patch(
            "ogx.providers.inline.file_processor.docling.docling.PictureItem",
            _FakePictureBase,
        ):
            result = await processor._extract_and_upload_pictures(doc, filename="report.pdf")

        assert [(p.picture_ref, p.file_id, p.pages) for p in result] == [
            ("#/pictures/0", "file-001", frozenset({1})),
            ("#/pictures/1", "file-002", frozenset({2, 3})),
        ]
        assert files_api.openai_upload_file.await_count == 2

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

        assert [p.picture_ref for p in result] == ["#/pictures/big"]
        assert files_api.openai_upload_file.await_count == 1

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

        # First upload fails, second succeeds. Only the surviving picture is returned.
        assert [p.picture_ref for p in result] == ["#/pictures/1"]
        assert result[0].file_id == "file-002"

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

        assert [p.picture_ref for p in result] == ["#/pictures/1"]
        assert files_api.openai_upload_file.await_count == 1

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
            pictures=[
                ExtractedPicture("#/pictures/0", "file-a", frozenset({1})),
                ExtractedPicture("#/pictures/1", "file-b", frozenset({2})),
            ],
        )

        assert len(chunks) == 1
        assert chunks[0].metadata[IMAGE_FILE_IDS_METADATA_KEY] == ["file-a", "file-b"]
        assert chunks[0].metadata["filename"] == "x.pdf"
        assert chunks[0].metadata["document_id"] == "doc-1"

    def test_no_image_key_when_no_pictures(self) -> None:
        cfg = DoclingFileProcessorConfig(extract_images=True)
        processor = DoclingFileProcessor(cfg, files_api=AsyncMock())
        doc = SimpleNamespace(export_to_markdown=lambda: "Hello world.")

        chunks = processor._create_chunks(
            doc=doc,
            document_id="doc-1",
            chunking_strategy=None,
            document_metadata={"filename": "x.pdf"},
            pictures=[],
        )

        assert len(chunks) == 1
        assert IMAGE_FILE_IDS_METADATA_KEY not in chunks[0].metadata

    def test_returns_empty_for_blank_doc_with_no_pictures(self) -> None:
        cfg = DoclingFileProcessorConfig(extract_images=True)
        processor = DoclingFileProcessor(cfg, files_api=AsyncMock())
        doc = SimpleNamespace(export_to_markdown=lambda: "   ")

        chunks = processor._create_chunks(
            doc=doc,
            document_id="doc-1",
            chunking_strategy=None,
            document_metadata={"filename": "x.pdf"},
            pictures=[],
        )

        assert chunks == []

    def test_blank_doc_with_pictures_emits_image_only_fallback_chunks(self) -> None:
        """Mirror the chunker branch: blank doc + pictures should fall back to per-picture chunks."""
        cfg = DoclingFileProcessorConfig(extract_images=True)
        processor = DoclingFileProcessor(cfg, files_api=AsyncMock())
        doc = SimpleNamespace(export_to_markdown=lambda: "   ")

        chunks = processor._create_chunks(
            doc=doc,
            document_id="doc-1",
            chunking_strategy=None,
            document_metadata={"filename": "11.jpg"},
            pictures=[ExtractedPicture("self", "file-src", frozenset({1}), caption=None)],
        )

        assert len(chunks) == 1
        assert chunks[0].content == "Image: 11.jpg"
        assert chunks[0].metadata[IMAGE_FILE_IDS_METADATA_KEY] == ["file-src"]


# NOTE: _build_converter wiring is intentionally not unit-tested here. It does lazy imports of
# the heavy `docling` package and constructs DocumentConverter / PdfFormatOption with
# images_scale and generate_picture_images=True. Mocking that import chain without installing
# docling proper (which pulls torch / huggingface) would obscure rather than verify behaviour;
# the wiring is exercised end-to-end by the integration test suite under tests/integration.


class TestMaybeCaptionPicture:
    """Vision-model captioning is opt-in and degrades gracefully on any error."""

    async def test_returns_none_when_caption_images_disabled(self) -> None:
        processor = DoclingFileProcessor(DoclingFileProcessorConfig(caption_images=False), inference_api=AsyncMock())
        assert await processor._maybe_caption_picture(b"\x89PNG", picture_ref="#/pictures/0") is None

    async def test_returns_none_when_inference_api_missing(self) -> None:
        cfg = DoclingFileProcessorConfig(caption_images=True, caption_model="vl-model")
        processor = DoclingFileProcessor(cfg, inference_api=None)
        assert await processor._maybe_caption_picture(b"\x89PNG", picture_ref="#/pictures/0") is None

    async def test_returns_none_when_model_unset(self) -> None:
        cfg = DoclingFileProcessorConfig(caption_images=True)  # caption_model is None
        processor = DoclingFileProcessor(cfg, inference_api=AsyncMock())
        assert await processor._maybe_caption_picture(b"\x89PNG", picture_ref="#/pictures/0") is None

    async def test_returns_caption_from_chat_completion(self) -> None:
        cfg = DoclingFileProcessorConfig(caption_images=True, caption_model="vl-model")
        inference = AsyncMock()
        inference.openai_chat_completion = AsyncMock(
            return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="A red race car."))])
        )
        processor = DoclingFileProcessor(cfg, inference_api=inference)

        result = await processor._maybe_caption_picture(b"\x89PNG\r\n\x1a\n", picture_ref="#/pictures/0")

        assert result == "A red race car."
        sent = inference.openai_chat_completion.await_args.args[0]
        assert sent.model == "vl-model"
        # Caption call carries both the prompt text and an image_url part
        content = sent.messages[0].content
        types = [getattr(p, "type", None) for p in content]
        assert "text" in types and "image_url" in types
        # data URL contains base64 of the bytes
        image_part = next(p for p in content if getattr(p, "type", None) == "image_url")
        assert image_part.image_url.url.startswith("data:image/png;base64,")

    async def test_tolerates_inference_failure(self) -> None:
        cfg = DoclingFileProcessorConfig(caption_images=True, caption_model="vl-model")
        inference = AsyncMock()
        inference.openai_chat_completion = AsyncMock(side_effect=RuntimeError("vision endpoint 503"))
        processor = DoclingFileProcessor(cfg, inference_api=inference)
        assert await processor._maybe_caption_picture(b"\x89PNG", picture_ref="#/pictures/0") is None

    async def test_returns_none_on_empty_choices(self) -> None:
        cfg = DoclingFileProcessorConfig(caption_images=True, caption_model="vl-model")
        inference = AsyncMock()
        inference.openai_chat_completion = AsyncMock(return_value=SimpleNamespace(choices=[]))
        processor = DoclingFileProcessor(cfg, inference_api=inference)
        assert await processor._maybe_caption_picture(b"\x89PNG", picture_ref="#/pictures/0") is None


class TestImageOnlyFallbackChunks:
    """When HybridChunker produces 0 text chunks, fall back to one chunk per picture."""

    def test_emits_one_chunk_per_picture_with_caption(self) -> None:
        chunks = DoclingFileProcessor._build_image_only_chunks(
            document_id="doc-1",
            document_metadata={"filename": "1.jpg"},
            pictures=[
                ExtractedPicture("#/pictures/0", "file-a", frozenset({1}), caption="A turbocharger."),
                ExtractedPicture("#/pictures/1", "file-b", frozenset({1}), caption=None),
            ],
        )

        assert len(chunks) == 2
        # First chunk uses the caption as its body
        assert chunks[0].content == "A turbocharger."
        assert chunks[0].metadata[IMAGE_FILE_IDS_METADATA_KEY] == ["file-a"]
        # Second chunk has no caption — falls back to a filename-based label
        assert chunks[1].content == "Image: 1.jpg"
        assert chunks[1].metadata[IMAGE_FILE_IDS_METADATA_KEY] == ["file-b"]
        # Each chunk records the picture index in chunk_window so callers can correlate
        assert chunks[0].chunk_metadata.chunk_window == "image-0"
        assert chunks[1].chunk_metadata.chunk_window == "image-1"

    def test_falls_back_to_generic_image_label_when_no_filename(self) -> None:
        chunks = DoclingFileProcessor._build_image_only_chunks(
            document_id="doc-1",
            document_metadata={},
            pictures=[ExtractedPicture("#/pictures/0", "file-a", frozenset({1}), caption=None)],
        )
        assert chunks[0].content == "Image"

    def test_empty_picture_list_yields_no_chunks(self) -> None:
        assert (
            DoclingFileProcessor._build_image_only_chunks(
                document_id="doc-1",
                document_metadata={"filename": "1.jpg"},
                pictures=[],
            )
            == []
        )


class TestIsImageInputFilename:
    """Filename-suffix gate for the self-picture synthesis path."""

    @pytest.mark.parametrize(
        "name",
        ["1.jpg", "photo.JPEG", "diagram.png", "chart.WebP", "anim.gif", "scan.tiff", "scan.tif"],
    )
    def test_recognises_image_suffixes(self, name: str) -> None:
        from ogx.providers.inline.file_processor.docling.docling import _is_image_input_filename

        assert _is_image_input_filename(name) is True

    @pytest.mark.parametrize("name", ["report.pdf", "notes.txt", "deck.pptx", "no-extension", ""])
    def test_rejects_non_image_suffixes(self, name: str) -> None:
        from ogx.providers.inline.file_processor.docling.docling import _is_image_input_filename

        assert _is_image_input_filename(name) is False


class TestBuildSelfPicture:
    """Synthesise an ExtractedPicture for standalone-image inputs.

    The bytes are already stored in the Files API under the source file_id, so the synthesis
    must NOT re-upload — it should reuse the id and only attempt a (best-effort) caption.
    """

    async def test_reuses_source_file_id_and_no_caption_when_disabled(self) -> None:
        cfg = DoclingFileProcessorConfig(caption_images=False)
        processor = DoclingFileProcessor(cfg, files_api=AsyncMock(), inference_api=AsyncMock())

        picture = await processor._build_self_picture(content=b"\xff\xd8\xff", file_id="file-src")

        assert picture is not None
        assert picture.file_id == "file-src"
        assert picture.picture_ref == "self"
        assert picture.pages == frozenset({1})
        assert picture.caption is None
        # No upload happened: files_api was never called
        processor.files_api.openai_upload_file.assert_not_called()

    async def test_captions_via_inference_when_enabled(self) -> None:
        cfg = DoclingFileProcessorConfig(caption_images=True, caption_model="vl-model")
        inference = AsyncMock()
        inference.openai_chat_completion = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="A red turbocharger."))]
            )
        )
        processor = DoclingFileProcessor(cfg, files_api=AsyncMock(), inference_api=inference)

        picture = await processor._build_self_picture(content=b"\xff\xd8\xff", file_id="file-src")

        assert picture is not None
        assert picture.caption == "A red turbocharger."
        assert picture.file_id == "file-src"


class TestProcessFileImageSelfPictureFallback:
    """End-to-end: process_file on a standalone image must synthesise a self-picture and
    produce one fallback chunk instead of returning [] and raising "No chunks were generated"."""

    async def test_standalone_image_with_no_picture_items_emits_one_chunk(self) -> None:
        from ogx_api.file_processors import ProcessFileRequest

        cfg = DoclingFileProcessorConfig(caption_images=False)

        files_api = AsyncMock()
        files_api.openai_retrieve_file = AsyncMock(return_value=SimpleNamespace(filename="11.jpg"))
        files_api.openai_retrieve_file_content = AsyncMock(return_value=SimpleNamespace(body=b"\xff\xd8\xff\xe0jpeg"))

        processor = DoclingFileProcessor(cfg, files_api=files_api, inference_api=None)

        # Docling sees an image with no embedded PictureItems — iterate_items returns nothing.
        empty_doc = SimpleNamespace(
            iterate_items=lambda: [],
            num_pages=lambda: 1,
            export_to_markdown=lambda: "",
        )
        fake_result = SimpleNamespace(document=empty_doc)
        fake_converter = SimpleNamespace(convert=lambda _path: fake_result)

        with patch.object(processor, "_build_converter", return_value=fake_converter):
            response = await processor.process_file(
                ProcessFileRequest(file_id="file-src", chunking_strategy=None),
                file=None,
            )

        assert len(response.chunks) == 1
        # Caption disabled and no caption available — fallback body is the filename label
        assert response.chunks[0].content == "Image: 11.jpg"
        assert response.chunks[0].metadata[IMAGE_FILE_IDS_METADATA_KEY] == ["file-src"]
