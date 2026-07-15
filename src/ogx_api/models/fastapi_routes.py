# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the Models API.

This module defines the FastAPI router for the Models API using standard
FastAPI route decorators. Supports OpenAI, Anthropic, and Google SDK
response formats via header-based SDK detection.
"""

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Query, Response, status
from fastapi.responses import JSONResponse

from ogx_api.messages.models import ANTHROPIC_VERSION
from ogx_api.router_utils import create_path_dependency, standard_responses
from ogx_api.version import OGX_API_V1

from .api import Models
from .models import (
    AdminListModelsResponse,
    AnthropicModelInfo,
    GetModelRequest,
    GoogleModelInfo,
    Model,
    OpenAIListModelsResponse,
    RegisterModelRequest,
    UnregisterModelRequest,
)

# Path parameter dependencies for single-field models
get_model_request = create_path_dependency(GetModelRequest)
unregister_model_request = create_path_dependency(UnregisterModelRequest)


def _is_google_sdk_request(
    x_goog_api_key: str | None,
    x_goog_user_project: str | None,
    x_goog_api_client: str | None,
) -> bool:
    return any((x_goog_api_key, x_goog_user_project, x_goog_api_client))


def _normalize_google_model_id(model_id: str) -> str:
    if model_id.startswith("/models/"):
        return model_id.removeprefix("/")
    if model_id.startswith("models/"):
        return model_id.removeprefix("models/")
    return model_id


async def _resolve_google_model_request(impl: Models, model_request: GetModelRequest) -> GetModelRequest:
    normalized_model_id = _normalize_google_model_id(model_request.model_id)

    all_models = await impl.list_models()
    matching_identifiers = sorted(
        {
            model.identifier
            for model in all_models.data
            if model.identifier == normalized_model_id or model.provider_resource_id == normalized_model_id
        }
    )
    if len(matching_identifiers) > 1:
        raise ValueError(
            "Failed to get model: Google model ID is ambiguous across providers. Use 'models/{provider_id}/{model_id}'."
        )
    if len(matching_identifiers) == 1:
        normalized_model_id = matching_identifiers[0]

    return GetModelRequest(model_id=normalized_model_id)


def create_router(impl: Models) -> APIRouter:
    """Create a FastAPI router for the Models API.

    Args:
        impl: The Models implementation instance

    Returns:
        APIRouter configured for the Models API
    """
    router = APIRouter(
        prefix=f"/{OGX_API_V1}",
        tags=["Models"],
        responses=standard_responses,
    )

    @router.get(
        "/models",
        response_model=OpenAIListModelsResponse,
        summary="List models.",
        description="List models. Returns OpenAI, Anthropic, or Google response format based on SDK detection headers.",
        responses={
            200: {"description": "A list of model objects."},
        },
    )
    async def list_models(
        before_id: Annotated[
            str | None, Query(description="Return models before this model ID (Anthropic SDK format only).")
        ] = None,
        after_id: Annotated[
            str | None, Query(description="Return models after this model ID (Anthropic SDK format only).")
        ] = None,
        limit: Annotated[
            int | None,
            Query(ge=1, le=1000, description="Maximum number of models to return (Anthropic SDK format only)."),
        ] = None,
        anthropic_version: Annotated[str | None, Header(alias="anthropic-version")] = None,
        x_goog_api_key: Annotated[str | None, Header(alias="x-goog-api-key")] = None,
        x_goog_user_project: Annotated[str | None, Header(alias="x-goog-user-project")] = None,
        x_goog_api_client: Annotated[str | None, Header(alias="x-goog-api-client")] = None,
    ) -> OpenAIListModelsResponse | Response:
        if anthropic_version:
            anthropic_result = await impl.anthropic_list_models(before_id=before_id, after_id=after_id, limit=limit)
            return JSONResponse(
                content=anthropic_result.model_dump(exclude_none=True),
                headers={"anthropic-version": ANTHROPIC_VERSION},
            )
        elif _is_google_sdk_request(x_goog_api_key, x_goog_user_project, x_goog_api_client):
            google_result = await impl.google_list_models()
            return JSONResponse(content=google_result.model_dump(exclude_none=True))

        return await impl.openai_list_models()

    @router.get(
        "/admin/models",
        response_model=AdminListModelsResponse,
        summary="Admin: list every model including hidden ones.",
        description=(
            "Admin-only listing of all models registered in OGX, including those that have been "
            "administratively hidden from the user-facing model list. Each entry carries a `hidden` "
            "flag. Visibility into a row requires delete permission on the model."
        ),
        responses={
            200: {"description": "A list of model objects with their visibility state."},
        },
    )
    async def list_all_models() -> AdminListModelsResponse:
        return await impl.list_all_models()

    @router.get(
        "/models/{model_id:path}",
        response_model=Model,
        summary="Get a model by its identifier.",
        description="Get a model by its identifier. Returns OpenAI, Anthropic, or Google response format based on SDK detection headers.",
        responses={
            200: {"description": "The model object."},
        },
    )
    async def get_model(
        model_request: Annotated[GetModelRequest, Depends(get_model_request)],
        anthropic_version: Annotated[str | None, Header(alias="anthropic-version")] = None,
        x_goog_api_key: Annotated[str | None, Header(alias="x-goog-api-key")] = None,
        x_goog_user_project: Annotated[str | None, Header(alias="x-goog-user-project")] = None,
        x_goog_api_client: Annotated[str | None, Header(alias="x-goog-api-client")] = None,
    ) -> Model | Response:
        normalized_model_request = model_request
        if _is_google_sdk_request(x_goog_api_key, x_goog_user_project, x_goog_api_client):
            normalized_model_request = await _resolve_google_model_request(impl, model_request)

        model = await impl.get_model(normalized_model_request)

        if anthropic_version:
            anthropic_model = AnthropicModelInfo(
                id=model.identifier,
                display_name=model.identifier,
                created_at=datetime.fromtimestamp(model.created, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
            return JSONResponse(
                content=anthropic_model.model_dump(exclude_none=True),
                headers={"anthropic-version": ANTHROPIC_VERSION},
            )
        elif _is_google_sdk_request(x_goog_api_key, x_goog_user_project, x_goog_api_client):
            google_model = GoogleModelInfo(
                name=f"models/{model.identifier}",
                display_name=model.identifier,
            )
            return JSONResponse(content=google_model.model_dump(exclude_none=True))

        return model

    @router.post(
        "/models",
        response_model=Model,
        summary="Register a model, or restore one that has been administratively hidden.",
        description=(
            "Register a new model in OGX, or restore a previously hidden one. Posting an identifier "
            "that matches a hidden (tombstoned) entry flips it back to visible."
        ),
        responses={
            200: {"description": "The registered (or restored) model object."},
        },
    )
    async def register_model(request: RegisterModelRequest) -> Model:
        return await impl.register_model(request)

    @router.delete(
        "/models/{model_id:path}",
        status_code=status.HTTP_204_NO_CONTENT,
        summary="Hide a model from the user-facing list.",
        description=(
            "Hide a model from the user-facing list and from inference requests. For models that "
            "originated from a provider's own listing, the entry is tombstoned in the registry so "
            "the hide survives provider refreshes; for user-registered models the entry is hard-deleted."
        ),
        responses={
            204: {"description": "Model successfully hidden or removed."},
        },
    )
    async def unregister_model(
        request: Annotated[UnregisterModelRequest, Depends(unregister_model_request)],
    ) -> None:
        await impl.unregister_model(request)

    return router
