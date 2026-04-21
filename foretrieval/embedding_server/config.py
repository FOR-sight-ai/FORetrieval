"""Configuration model for the remote embedding server."""

import os
from typing import Optional

from pydantic import BaseModel, field_validator, model_validator

# Model name substrings that vLLM supports for the /pooling endpoint.
# Only ColQwen3 / ColQwen3.5 architecture is supported in vLLM >=0.19.0.
_VLLM_COMPATIBLE_PATTERNS = ("colqwen3",)


class EmbeddingServerConfig(BaseModel):
    """Configuration for a remote vLLM embedding server.

    When set on ColPaliModel, the model weights are NOT loaded locally.
    Only the processor (tokenizer + image preprocessor) is loaded for
    heatmap sidecar computation and query tokenisation.

    Attributes:
        url: Full base URL of the vLLM server, e.g. "http://gpu-server:8000".
        model_name: HuggingFace model ID served by the vLLM instance,
            e.g. "athrael-soju/colqwen3.5-4.5B-v3".
            Must contain "colqwen3" — vLLM >=0.19.0 only supports the
            ColQwen3/ColQwen3.5 architecture for the /pooling endpoint.
            ColPali, ColQwen2, and ColQwen2.5 are not supported by vLLM.
        auto_deploy: When True, FORetrieval will SSH to ssh_host and
            deploy a Docker container if the server is not already running.
            Requires ssh_host to be set.
        ssh_host: Hostname or IP of the GPU server (SSH target).
            Required when auto_deploy=True.
        ssh_user: SSH username. Defaults to current OS user.
        ssh_key_path: Path to SSH private key file. If None, uses the
            SSH agent or default keys (~/.ssh/id_rsa etc.).
        n_gpus: Number of GPUs to use for tensor parallelism on the server.
            -1 (default) means all available GPUs detected via nvidia-smi.
        port: Port to expose the vLLM server on. Default 8000.
        hf_token: HuggingFace token passed as HF_TOKEN env var to the
            Docker container for downloading gated models.
        api_key: Optional bearer token for server authentication.
            When set, requests include "Authorization: Bearer <api_key>".
            Deploy the vLLM server with --api-key <api_key> to enable.
        verify_ssl: Whether to verify SSL certificates. Default True.
            Set to False for servers with self-signed certificates.
        batch_size: Initial number of images per /pooling request.
            Automatically halved on CUDA OOM, down to a minimum of 1.
        request_timeout: HTTP request timeout in seconds. Default 120.
    """

    url: str
    model_name: str
    auto_deploy: bool = False
    ssh_host: Optional[str] = None
    ssh_user: Optional[str] = None  # None → resolved to $USER at deploy time
    ssh_key_path: Optional[str] = None
    n_gpus: int = -1
    port: int = 8000
    hf_token: Optional[str] = None
    api_key: Optional[str] = None
    verify_ssl: bool = True
    batch_size: int = 4
    request_timeout: int = 120

    @field_validator("url")
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @field_validator("model_name")
    @classmethod
    def validate_vllm_compatible_model(cls, v: str) -> str:
        lower = v.lower()
        if not any(pat in lower for pat in _VLLM_COMPATIBLE_PATTERNS):
            raise ValueError(
                f"model_name '{v}' does not appear to be compatible with the vLLM "
                f"/pooling endpoint. vLLM >=0.19.0 supports only ColQwen3/ColQwen3.5 "
                f"models (model name must contain 'colqwen3'). "
                f"ColPali, ColQwen2, and ColQwen2.5 are not supported by vLLM. "
                f"Use a ColQwen3.5 model such as 'athrael-soju/colqwen3.5-4.5B-v3', "
                f"or run the model locally without an embedding server."
            )
        return v

    @field_validator("n_gpus")
    @classmethod
    def validate_n_gpus(cls, v: int) -> int:
        if v < -1 or v == 0:
            raise ValueError("n_gpus must be -1 (all GPUs) or a positive integer")
        return v

    @model_validator(mode="after")
    def auto_deploy_requires_ssh_host(self) -> "EmbeddingServerConfig":
        if self.auto_deploy and not self.ssh_host:
            raise ValueError("ssh_host is required when auto_deploy=True")
        return self

    @classmethod
    def from_dict(cls, d: dict) -> "EmbeddingServerConfig":
        """Convenience constructor from a plain config dict (e.g. from JSON)."""
        return cls(**d)
