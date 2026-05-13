# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the Models API.

This module defines the FastAPI router for the Models API using standard
FastAPI route decorators.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, status

from ogx_api.router_utils import create_path_dependency, standard_responses
from ogx_api.version import OGX_API_V1

from .api import Models
from .models import (
    AdminListModelsResponse,
    GetModelRequest,
    Model,
    OpenAIListModelsResponse,
    RegisterModelRequest,
    UnregisterModelRequest,
)

# Path parameter dependencies for single-field models
get_model_request = create_path_dependency(GetModelRequest)
unregister_model_request = create_path_dependency(UnregisterModelRequest)


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
        summary="List models using the OpenAI API.",
        description="List models using the OpenAI API.",
        responses={
            200: {"description": "A list of OpenAI model objects."},
        },
    )
    async def openai_list_models() -> OpenAIListModelsResponse:
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
        description="Get a model by its identifier.",
        responses={
            200: {"description": "The model object."},
        },
    )
    async def get_model(
        request: Annotated[GetModelRequest, Depends(get_model_request)],
    ) -> Model:
        return await impl.get_model(request)

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
