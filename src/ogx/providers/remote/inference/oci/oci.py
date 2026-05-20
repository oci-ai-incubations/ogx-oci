# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from collections.abc import Iterable
from typing import Any

import httpx
import oci
from oci.generative_ai.generative_ai_client import GenerativeAiClient
from oci.generative_ai.models import EndpointCollection, ModelCollection
from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from ogx.log import get_logger
from ogx.providers.remote.inference.oci.auth import OciInstancePrincipalAuth, OciUserPrincipalAuth
from ogx.providers.remote.inference.oci.config import OCIConfig
from ogx.providers.utils.inference.openai_mixin import OpenAIMixin
from ogx_api import (
    Model,
    ModelType,
    OpenAIEmbeddingData,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
    OpenAIEmbeddingUsage,
    validate_embeddings_input_is_text,
)

logger = get_logger(name=__name__, category="inference::oci")

OCI_AUTH_TYPE_INSTANCE_PRINCIPAL = "instance_principal"
OCI_AUTH_TYPE_CONFIG_FILE = "config_file"
VALID_OCI_AUTH_TYPES = [OCI_AUTH_TYPE_INSTANCE_PRINCIPAL, OCI_AUTH_TYPE_CONFIG_FILE]
DEFAULT_OCI_REGION = "us-ashburn-1"

MODEL_CAPABILITIES = ["TEXT_GENERATION", "TEXT_SUMMARIZATION", "TEXT_EMBEDDINGS", "CHAT"]

# DAC endpoints have OCIDs of this resource type; foundation models do not.
DAC_ENDPOINT_OCID_PREFIX = "ocid1.generativeaiendpoint."


class OCIInferenceAdapter(OpenAIMixin):
    """Inference adapter for Oracle Cloud Infrastructure Generative AI.

    Surfaces two kinds of OCI models:
      1. On-demand foundation models from `GenerativeAiClient.list_models`.
      2. Dedicated AI Cluster (DAC) endpoints from `GenerativeAiClient.list_endpoints`.

    On-demand requests are routed through the shared inference URL
    (`/20231130/actions/v1`). DAC requests are routed through a per-endpoint
    URL (`/<endpoint_ocid>/v1`); we build a one-shot AsyncOpenAI client for
    each DAC call so a single mixin can serve both paths.
    """

    config: OCIConfig

    embedding_models: list[str] = []

    async def initialize(self) -> None:
        """Initialize and validate OCI configuration."""
        # display_name -> endpoint OCID, populated by list_provider_model_ids.
        # Instance attribute, not a class default — mutable defaults would leak
        # across adapter instances.
        self._dac_endpoints: dict[str, str] = {}

        if self.config.oci_auth_type not in VALID_OCI_AUTH_TYPES:
            raise ValueError(
                f"Invalid OCI authentication type: {self.config.oci_auth_type}."
                f"Valid types are one of: {VALID_OCI_AUTH_TYPES}"
            )

        if not self.config.oci_compartment_id:
            raise ValueError("OCI_COMPARTMENT_OCID is a required parameter. Either set in env variable or config.")

    def get_base_url(self) -> str:
        region = self.config.oci_region or DEFAULT_OCI_REGION
        return f"https://inference.generativeai.{region}.oci.oraclecloud.com/20231130/actions/v1"

    def get_api_key(self) -> str | None:
        # OCI doesn't use API keys, it uses request signing
        return "<NOTUSED>"

    def get_extra_client_params(self) -> dict[str, Any]:
        auth = self._get_auth()
        compartment_id = self.config.oci_compartment_id or ""

        return {
            "http_client": DefaultAsyncHttpxClient(
                auth=auth,
                headers={
                    "CompartmentId": compartment_id,
                },
                verify=self.shared_ssl_context,
            ),
        }

    def _get_oci_signer(self) -> oci.signer.AbstractBaseSigner | None:
        if self.config.oci_auth_type == OCI_AUTH_TYPE_INSTANCE_PRINCIPAL:
            return oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
        return None

    def _get_oci_config(self) -> dict:
        if self.config.oci_auth_type == OCI_AUTH_TYPE_INSTANCE_PRINCIPAL:
            config = {"region": self.config.oci_region}
        elif self.config.oci_auth_type == OCI_AUTH_TYPE_CONFIG_FILE:
            config = oci.config.from_file(self.config.oci_config_file_path, self.config.oci_config_profile)
            if not config.get("region"):
                raise ValueError(
                    "Region not specified in config. Please specify in config or with OCI_REGION env variable."
                )

        return config

    def _get_auth(self) -> httpx.Auth:
        if self.config.oci_auth_type == OCI_AUTH_TYPE_INSTANCE_PRINCIPAL:
            return OciInstancePrincipalAuth()
        elif self.config.oci_auth_type == OCI_AUTH_TYPE_CONFIG_FILE:
            return OciUserPrincipalAuth(
                config_file=self.config.oci_config_file_path, profile_name=self.config.oci_config_profile
            )
        else:
            raise ValueError(f"Invalid OCI authentication type: {self.config.oci_auth_type}")

    def _generative_ai_client(self) -> GenerativeAiClient:
        oci_config = self._get_oci_config()
        oci_signer = self._get_oci_signer()
        if oci_signer is None:
            return GenerativeAiClient(config=oci_config)
        return GenerativeAiClient(config=oci_config, signer=oci_signer)

    async def list_provider_model_ids(self) -> Iterable[str]:
        """List on-demand foundation models AND DAC endpoints."""
        compartment_id = self.config.oci_compartment_id or ""
        client = self._generative_ai_client()

        seen_models: set[tuple[str, ModelType]] = set()
        model_ids: list[str] = []

        # 1. On-demand foundation models (existing behavior)
        models: ModelCollection = client.list_models(
            compartment_id=compartment_id,
            lifecycle_state="ACTIVE",
        ).data
        for model in models.items:
            if model.time_deprecated or model.time_on_demand_retired:
                continue
            if "UNKNOWN_ENUM_VALUE" in model.capabilities or "FINE_TUNE" in model.capabilities:
                continue
            key = (model.display_name, ModelType.llm)
            if key in seen_models:
                continue
            seen_models.add(key)
            model_ids.append(model.display_name)
            if "TEXT_EMBEDDINGS" in model.capabilities:
                self.embedding_models.append(model.display_name)

        # 2. DAC (Dedicated AI Cluster) endpoints — NEW.
        # These are user-provisioned endpoints; they don't show up in list_models.
        # We expose them via their display_name and remember the OCID for routing.
        try:
            endpoints: EndpointCollection = client.list_endpoints(
                compartment_id=compartment_id,
                lifecycle_state="ACTIVE",
            ).data
        except Exception as exc:
            # Don't fail the whole inventory if the caller lacks endpoint list permission.
            logger.warning("OCI list_endpoints failed; DAC endpoints will not be surfaced: %s", exc)
            return model_ids

        for endpoint in endpoints.items:
            if endpoint.lifecycle_state != "ACTIVE":
                continue
            identifier = endpoint.display_name or endpoint.id
            key = (identifier, ModelType.llm)
            if key in seen_models:
                # Shadow check: in case a foundation model has the same display name.
                identifier = endpoint.id
                key = (identifier, ModelType.llm)
                if key in seen_models:
                    continue
            seen_models.add(key)
            self._dac_endpoints[identifier] = endpoint.id
            model_ids.append(identifier)

        return model_ids

    def construct_model_from_identifier(self, identifier: str) -> Model:
        """Construct a Model instance corresponding to the given identifier."""
        if identifier in self.embedding_models:
            return Model(
                provider_id=self.__provider_id__,  # type: ignore[attr-defined]
                provider_resource_id=identifier,
                identifier=identifier,
                model_type=ModelType.embedding,
            )
        return Model(
            provider_id=self.__provider_id__,  # type: ignore[attr-defined]
            provider_resource_id=identifier,
            identifier=identifier,
            model_type=ModelType.llm,
        )

    # ------------------------------------------------------------------
    # DAC routing
    # ------------------------------------------------------------------

    def _resolve_dac_endpoint_id(self, model: str) -> str | None:
        """If `model` refers to a DAC endpoint, return its OCID, else None."""
        if model in self._dac_endpoints:
            return self._dac_endpoints[model]
        if model.startswith(DAC_ENDPOINT_OCID_PREFIX):
            return model
        return None

    def _build_dac_client(self, endpoint_id: str) -> AsyncOpenAI:
        """Build a one-shot AsyncOpenAI client that targets a single DAC endpoint."""
        region = self.config.oci_region or DEFAULT_OCI_REGION
        compartment_id = self.config.oci_compartment_id or ""
        return AsyncOpenAI(
            base_url=f"https://inference.generativeai.{region}.oci.oraclecloud.com/{endpoint_id}/v1",
            api_key="<NOTUSED>",
            http_client=DefaultAsyncHttpxClient(
                auth=self._get_auth(),
                headers={"CompartmentId": compartment_id},
                verify=self.shared_ssl_context,
            ),
        )

    async def openai_chat_completion(self, params):  # type: ignore[override]
        """Route DAC requests via per-endpoint URL; everything else via the mixin."""
        endpoint_id = self._resolve_dac_endpoint_id(params.model)
        if endpoint_id is None:
            return await super().openai_chat_completion(params)

        # DAC URL embeds the endpoint OCID. The `model` field in the request body
        # is informational for DAC (the path-encoded endpoint dispatches to the
        # right model). Pass the served-model display name if known, else the OCID.
        request_params = params.model_dump(exclude_none=True)
        async with self._build_dac_client(endpoint_id) as client:
            return await client.chat.completions.create(**request_params)

    async def openai_completion(self, params):  # type: ignore[override]
        """DAC routing for legacy completion endpoint."""
        endpoint_id = self._resolve_dac_endpoint_id(params.model)
        if endpoint_id is None:
            return await super().openai_completion(params)
        request_params = params.model_dump(exclude_none=True)
        async with self._build_dac_client(endpoint_id) as client:
            return await client.completions.create(**request_params)

    # ------------------------------------------------------------------
    # Embeddings (unchanged below)
    # ------------------------------------------------------------------

    async def openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        if "cohere" in params.model:
            return await self.cohere_embeddings(params)
        else:
            return await self.get_openai_embeddings(params)

    async def get_openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        if not self.supports_tokenized_embeddings_input:
            validate_embeddings_input_is_text(params)

        provider_model_id = await self._get_provider_model_id(params.model)
        self._validate_model_allowed(provider_model_id)

        request_params: dict[str, Any] = {
            "model": provider_model_id,
            "input": params.input,
        }
        if params.encoding_format is not None:
            request_params["encoding_format"] = params.encoding_format
        if params.dimensions is not None:
            request_params["dimensions"] = params.dimensions
        if params.user is not None:
            request_params["user"] = params.user
        if params.model_extra:
            request_params["extra_body"] = params.model_extra

        response = await self.client.embeddings.create(**request_params)

        data = []
        for i, embedding_data in enumerate(response.data):
            data.append(
                OpenAIEmbeddingData(
                    embedding=embedding_data.embedding,
                    index=i,
                )
            )

        usage = OpenAIEmbeddingUsage(
            prompt_tokens=response.usage.prompt_tokens,
            total_tokens=response.usage.total_tokens,
        )

        return OpenAIEmbeddingsResponse(
            data=data,
            model=params.model,
            usage=usage,
        )

    @property
    def cohere_client(self):
        oci_config = self._get_oci_config()
        oci_signer = self._get_oci_signer()
        if oci_signer is None:
            return oci.generative_ai_inference.GenerativeAiInferenceClient(
                config=oci_config, retry_strategy=oci.retry.NoneRetryStrategy(), timeout=(10, 240)
            )
        else:
            return oci.generative_ai_inference.GenerativeAiInferenceClient(
                config=oci_config, signer=oci_signer, retry_strategy=oci.retry.NoneRetryStrategy(), timeout=(10, 240)
            )

    def _validate_cohere_dimensions(self, dimensions: int, provider_model_id: str) -> None:
        if dimensions not in [256, 512, 1024, 1536]:
            raise ValueError(
                f"Model '{provider_model_id}' only accepts dimension in [256 512 1024 1536]"
                f"Request dimensions: {dimensions}"
            )

    async def cohere_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        if not self.supports_tokenized_embeddings_input:
            validate_embeddings_input_is_text(params)

        provider_model_id = await self._get_provider_model_id(params.model)
        self._validate_model_allowed(provider_model_id)

        self._validate_cohere_dimensions(params.dimensions, provider_model_id)

        embed_text_response = self.cohere_client.embed_text(
            oci.generative_ai_inference.models.EmbedTextDetails(
                compartment_id=self.config.oci_compartment_id,
                serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(model_id=provider_model_id),
                inputs=params.input if isinstance(params.input, list) else [params.input],
                embedding_types=[params.encoding_format],
            )
        )
        logger.debug(embed_text_response.data._embeddings_by_type[params.encoding_format][0:10])
        result = embed_text_response.data
        data = []
        for i, embedding_data in enumerate(result._embeddings_by_type[params.encoding_format]):
            data.append(
                OpenAIEmbeddingData(
                    embedding=embedding_data,
                    index=i,
                )
            )

        if hasattr(result, "usage") and result.usage:
            usage = OpenAIEmbeddingUsage(
                prompt_tokens=result.usage.prompt_tokens,
                total_tokens=result.usage.total_tokens,
            )
        else:
            usage = OpenAIEmbeddingUsage(
                prompt_tokens=0,
                total_tokens=0,
            )

        return OpenAIEmbeddingsResponse(
            data=data,
            model=params.model,
            usage=usage,
        )
