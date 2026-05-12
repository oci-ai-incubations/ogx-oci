# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for admin model visibility — tombstone-on-unregister, resurrect-on-register,
admin listing, and survival of tombstones across provider refresh."""

from unittest.mock import AsyncMock

import pytest

from ogx.core.datatypes import RegistryEntrySource
from ogx.core.routing_tables.models import ModelsRoutingTable
from ogx_api import (
    Api,
    Model,
    ModelNotFoundError,
    ModelType,
)


class InferenceImpl:
    """Minimal inference-provider stand-in for routing-table tests."""

    def __init__(self, models: list[Model] | None = None):
        self.api = Api.inference
        self._models = models or []

    @property
    def __provider_spec__(self):
        spec = AsyncMock()
        spec.api = self.api
        return spec

    async def register_model(self, model: Model):
        return model

    async def unregister_model(self, model_id: str):
        return model_id

    async def should_refresh_models(self):
        return False

    async def list_models(self):
        return self._models

    async def shutdown(self):
        pass


def _provider_model(identifier: str, *, provider_id: str = "oci") -> Model:
    return Model(
        identifier=identifier,
        provider_resource_id=identifier.split("/", 1)[-1],
        provider_id=provider_id,
        metadata={},
        model_type=ModelType.llm,
    )


async def _seed_listed_from_provider(table: ModelsRoutingTable, provider_id: str, models: list[Model]) -> None:
    """Populate the registry with provider-listed models, the path refresh would take."""
    await table.update_registered_models(provider_id, models)


async def _get_raw(table: ModelsRoutingTable, identifier: str):
    """Fetch directly from dist_registry, bypassing routing-table filters."""
    return await table.dist_registry.get("model", identifier)


async def test_unregister_listed_from_provider_tombstones(cached_disk_dist_registry):
    table = ModelsRoutingTable({"oci": InferenceImpl()}, cached_disk_dist_registry, {})
    await table.initialize()
    await _seed_listed_from_provider(table, "oci", [_provider_model("oci/cohere.broken")])

    await table.unregister_model(model_id="oci/cohere.broken")

    # The raw row stays put, but its source flips to admin_removed
    raw = await _get_raw(table, "oci/cohere.broken")
    assert raw is not None
    assert raw.source == RegistryEntrySource.admin_removed

    # User-facing listing no longer sees it
    listed = await table.list_models()
    assert all(m.identifier != "oci/cohere.broken" for m in listed.data)

    openai_listed = await table.openai_list_models()
    assert all(m.id != "oci/cohere.broken" for m in openai_listed.data)


async def test_unregister_via_register_api_hard_deletes(cached_disk_dist_registry):
    table = ModelsRoutingTable({"oci": InferenceImpl()}, cached_disk_dist_registry, {})
    await table.initialize()
    await table.register_model(model_id="custom", provider_id="oci")
    assert await _get_raw(table, "oci/custom") is not None

    await table.unregister_model(model_id="oci/custom")

    # Hard-deleted from the registry — no row remains
    assert await _get_raw(table, "oci/custom") is None


async def test_tombstone_survives_update_registered_models(cached_disk_dist_registry):
    """A hidden model must not reappear on the next provider refresh."""
    table = ModelsRoutingTable({"oci": InferenceImpl()}, cached_disk_dist_registry, {})
    await table.initialize()
    provider_models = [_provider_model("oci/cohere.broken"), _provider_model("oci/cohere.good")]
    await _seed_listed_from_provider(table, "oci", provider_models)

    await table.unregister_model(model_id="oci/cohere.broken")

    # Simulate the next refresh — provider still lists both models
    await table.update_registered_models("oci", provider_models)

    raw_hidden = await _get_raw(table, "oci/cohere.broken")
    assert raw_hidden is not None
    assert raw_hidden.source == RegistryEntrySource.admin_removed

    listed_ids = {m.identifier for m in (await table.list_models()).data}
    assert "oci/cohere.broken" not in listed_ids
    assert "oci/cohere.good" in listed_ids


async def test_register_resurrects_tombstone(cached_disk_dist_registry):
    table = ModelsRoutingTable({"oci": InferenceImpl()}, cached_disk_dist_registry, {})
    await table.initialize()
    await _seed_listed_from_provider(table, "oci", [_provider_model("oci/cohere.broken")])
    await table.unregister_model(model_id="oci/cohere.broken")

    # Re-register by full identifier
    await table.register_model(model_id="oci/cohere.broken", provider_id="oci")

    raw = await _get_raw(table, "oci/cohere.broken")
    assert raw is not None
    assert raw.source == RegistryEntrySource.via_register_api

    listed_ids = {m.identifier for m in (await table.list_models()).data}
    assert "oci/cohere.broken" in listed_ids


async def test_register_resurrects_tombstone_via_bare_id(cached_disk_dist_registry):
    """Resurrection works when the caller passes the bare model_id instead of the qualified one."""
    table = ModelsRoutingTable({"oci": InferenceImpl()}, cached_disk_dist_registry, {})
    await table.initialize()
    await _seed_listed_from_provider(table, "oci", [_provider_model("oci/cohere.broken")])
    await table.unregister_model(model_id="oci/cohere.broken")

    await table.register_model(model_id="cohere.broken", provider_id="oci")

    raw = await _get_raw(table, "oci/cohere.broken")
    assert raw is not None
    assert raw.source == RegistryEntrySource.via_register_api


async def test_list_all_models_returns_both_visible_and_hidden(cached_disk_dist_registry):
    table = ModelsRoutingTable({"oci": InferenceImpl()}, cached_disk_dist_registry, {})
    await table.initialize()
    await _seed_listed_from_provider(table, "oci", [_provider_model("oci/live"), _provider_model("oci/hidden")])
    await table.unregister_model(model_id="oci/hidden")

    admin = await table.list_all_models()
    by_id = {m.identifier: m for m in admin.data}

    assert "oci/live" in by_id and by_id["oci/live"].hidden is False
    assert "oci/hidden" in by_id and by_id["oci/hidden"].hidden is True


async def test_lookup_model_404s_for_tombstoned(cached_disk_dist_registry):
    table = ModelsRoutingTable({"oci": InferenceImpl()}, cached_disk_dist_registry, {})
    await table.initialize()
    await _seed_listed_from_provider(table, "oci", [_provider_model("oci/cohere.broken")])
    await table.unregister_model(model_id="oci/cohere.broken")

    with pytest.raises(ModelNotFoundError):
        await table.get_model("oci/cohere.broken")
    assert not await table.has_model("oci/cohere.broken")


async def test_unregister_unknown_raises_not_found(cached_disk_dist_registry):
    table = ModelsRoutingTable({"oci": InferenceImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    with pytest.raises(ModelNotFoundError):
        await table.unregister_model(model_id="oci/does-not-exist")


async def test_unregister_already_hidden_raises_not_found(cached_disk_dist_registry):
    """Hiding twice should look like a not-found — tombstones are invisible to lookup."""
    table = ModelsRoutingTable({"oci": InferenceImpl()}, cached_disk_dist_registry, {})
    await table.initialize()
    await _seed_listed_from_provider(table, "oci", [_provider_model("oci/cohere.broken")])
    await table.unregister_model(model_id="oci/cohere.broken")

    with pytest.raises(ModelNotFoundError):
        await table.unregister_model(model_id="oci/cohere.broken")
