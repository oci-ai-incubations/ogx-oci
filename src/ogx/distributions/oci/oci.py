# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path

from ogx.core.datatypes import BuildProvider, Provider
from ogx.distributions.template import DistributionTemplate, RunConfigSettings
from ogx.providers.inline.batches.reference.config import ReferenceBatchesImplConfig
from ogx.providers.inline.file_processor.auto.config import AutoFileProcessorConfig
from ogx.providers.inline.file_processor.docling.config import DoclingFileProcessorConfig
from ogx.providers.inline.files.localfs.config import LocalfsFilesImplConfig
from ogx.providers.inline.safety.code_scanner.config import CodeScannerConfig
from ogx.providers.inline.safety.prompt_guard.config import PromptGuardConfig
from ogx.providers.inline.vector_io.faiss.config import FaissVectorIOConfig
from ogx.providers.inline.vector_io.sqlite_vec.config import SQLiteVectorIOConfig
from ogx.providers.remote.files.s3.config import S3FilesImplConfig
from ogx.providers.remote.inference.oci.config import OCIConfig
from ogx.providers.remote.vector_io.oci.config import OCI26aiVectorIOConfig


def get_distribution_template(name: str = "oci") -> DistributionTemplate:
    """Build the OCI Generative AI distribution template.

    Args:
        name: the distribution name.

    Returns:
        A DistributionTemplate configured for OCI inference.
    """
    providers = {
        "inference": [BuildProvider(provider_type="remote::oci")],
        "vector_io": [
            BuildProvider(provider_type="inline::faiss"),
            BuildProvider(provider_type="inline::sqlite-vec"),
            BuildProvider(provider_type="remote::chromadb"),
            BuildProvider(provider_type="remote::pgvector"),
            BuildProvider(provider_type="remote::oci"),
        ],
        "safety": [
            BuildProvider(provider_type="inline::llama-guard"),
            BuildProvider(provider_type="inline::code-scanner"),
            BuildProvider(provider_type="inline::prompt-guard"),
        ],
        "responses": [BuildProvider(provider_type="inline::builtin")],
        "tool_runtime": [
            BuildProvider(provider_type="remote::brave-search"),
            BuildProvider(provider_type="remote::tavily-search"),
            BuildProvider(provider_type="inline::file-search"),
            BuildProvider(provider_type="remote::model-context-protocol"),
        ],
        "files": [
            BuildProvider(provider_type="inline::localfs"),
            BuildProvider(provider_type="remote::s3"),
        ],
        "file_processors": [
            BuildProvider(provider_type="inline::auto"),
            BuildProvider(provider_type="inline::docling"),
        ],
        "batches": [BuildProvider(provider_type="inline::reference")],
    }

    inference_provider = Provider(
        provider_id="oci",
        provider_type="remote::oci",
        config=OCIConfig.sample_run_config(),
    )

    vector_io_provider = Provider(
        provider_id="faiss",
        provider_type="inline::faiss",
        config=FaissVectorIOConfig.sample_run_config(f"~/.ogx/distributions/{name}"),
    )

    oci_vector_io_provider = Provider(
        provider_id="oci",
        provider_type="remote::oci",
        config=OCI26aiVectorIOConfig.sample_run_config(f"~/.ogx/distributions/{name}"),
    )

    files_provider = Provider(
        provider_id="builtin-files",
        provider_type="inline::localfs",
        config=LocalfsFilesImplConfig.sample_run_config(f"~/.ogx/distributions/{name}"),
    )

    s3_files_provider = Provider(
        provider_id="s3-files",
        provider_type="remote::s3",
        config=S3FilesImplConfig.sample_run_config(f"~/.ogx/distributions/{name}"),
    )

    sqlite_vec_provider = Provider(
        provider_id="sqlite-vec",
        provider_type="inline::sqlite-vec",
        config=SQLiteVectorIOConfig.sample_run_config(f"~/.ogx/distributions/{name}"),
    )

    code_scanner_provider = Provider(
        provider_id="code-scanner",
        provider_type="inline::code-scanner",
        config=CodeScannerConfig.sample_run_config(f"~/.ogx/distributions/{name}"),
    )

    prompt_guard_provider = Provider(
        provider_id="prompt-guard",
        provider_type="inline::prompt-guard",
        config=PromptGuardConfig.sample_run_config(f"~/.ogx/distributions/{name}"),
    )

    auto_file_processor_provider = Provider(
        provider_id="auto",
        provider_type="inline::auto",
        config=AutoFileProcessorConfig.sample_run_config(),
    )

    docling_provider = Provider(
        provider_id="docling",
        provider_type="inline::docling",
        config=DoclingFileProcessorConfig.sample_run_config(),
    )

    batches_provider = Provider(
        provider_id="reference",
        provider_type="inline::reference",
        config=ReferenceBatchesImplConfig.sample_run_config(f"~/.ogx/distributions/{name}"),
    )
    return DistributionTemplate(
        name=name,
        distro_type="remote_hosted",
        description="Use Oracle Cloud Infrastructure (OCI) Generative AI for running LLM inference with scalable cloud services",
        container_image=None,
        template_path=Path(__file__).parent / "doc_template.md",
        providers=providers,
        run_configs={
            "config.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": [inference_provider],
                    "vector_io": [vector_io_provider, sqlite_vec_provider, oci_vector_io_provider],
                    "safety": [code_scanner_provider, prompt_guard_provider],
                    "files": [files_provider, s3_files_provider],
                    "file_processors": [auto_file_processor_provider, docling_provider],
                    "batches": [batches_provider],
                },
                auth_config={
                    "provider_config": {
                        "type": "custom",
                        "endpoint": "${env.AUTH_VALIDATE_ENDPOINT:=http://localhost:8080/auth/validate}",
                    },
                },
            ),
        },
        run_config_env_vars={
            "OCI_AUTH_TYPE": (
                "instance_principal",
                "OCI authentication type (instance_principal or config_file)",
            ),
            "OCI_REGION": (
                "",
                "OCI region (e.g., us-ashburn-1, us-chicago-1, us-phoenix-1, eu-frankfurt-1)",
            ),
            "OCI_COMPARTMENT_OCID": (
                "",
                "OCI compartment ID for the Generative AI service",
            ),
            "OCI_CONFIG_FILE_PATH": (
                "~/.oci/config",
                "OCI config file path (required if OCI_AUTH_TYPE is config_file)",
            ),
            "OCI_CLI_PROFILE": (
                "DEFAULT",
                "OCI CLI profile name to use from config file",
            ),
            "AUTH_VALIDATE_ENDPOINT": (
                "http://localhost:8080/auth/validate",
                "URL of the auth-service token validation endpoint (POSTs {api_key, request} and expects {principal, attributes})",
            ),
        },
    )
