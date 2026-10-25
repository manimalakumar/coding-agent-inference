"""Read-only provider validation before the measured workflow starts."""

from urllib.parse import quote, urlparse

import httpx

from agent.schemas import ModelConfig
from sandbox.docker import InfrastructureError


class ConfigurationError(InfrastructureError):
    pass


def parameter_plan(model: ModelConfig, metadata: dict) -> dict:
    endpoints = metadata.get("data", {}).get("endpoints", [])
    mandatory = {"tools", "tool_choice", "temperature", "max_tokens"}
    usable = [
        e for e in endpoints if mandatory <= set(e.get("supported_parameters", []))
    ]
    if not usable:
        raise ConfigurationError("No endpoint advertises all required agent parameters")
    parameters = {}
    omitted = []
    if model.reasoning is not None:
        reasoning_endpoints = [
            e for e in usable if "reasoning" in e.get("supported_parameters", [])
        ]
        if reasoning_endpoints:
            parameters["reasoning"] = model.reasoning
            usable = reasoning_endpoints
        elif model.reasoning.get("enabled") is True:
            raise ConfigurationError("Requested reasoning is unsupported by this model")
        else:
            omitted.append("reasoning")
    provider = dict(model.provider)
    provider["require_parameters"] = True
    eligible = sorted({e["provider_name"] for e in usable if e.get("provider_name")})
    slugs = sorted({e["tag"] for e in usable if e.get("tag")})
    if provider.get("only"):
        slugs = [
            slug
            for slug in slugs
            if any(
                slug == allowed or slug.startswith(allowed + "/")
                for allowed in provider["only"]
            )
        ]
        if not slugs:
            raise ConfigurationError(
                "Provider policy excludes all compatible endpoints"
            )
    if not eligible:
        raise ConfigurationError("Provider policy excludes all compatible endpoints")
    if slugs:
        provider["only"] = slugs
    return {
        "parameters": parameters,
        "provider": provider,
        "omitted_unsupported_parameters": omitted,
        "eligible_providers": eligible,
        "eligible_provider_slugs": slugs,
        "model": model.model,
        "validation_method": "read_only_endpoint_metadata_no_inference",
    }


def validate_model(model: ModelConfig) -> dict:
    import os
    if os.environ.get("PUBLIC_NO_INFERENCE") == "1":
        raise RuntimeError("Inference is disabled during public preparation controls")
    endpoint = urlparse(model.endpoint)
    if endpoint.scheme != "https" or endpoint.hostname != "openrouter.ai":
        raise ConfigurationError(
            "Schema v1 provider preflight supports OpenRouter HTTPS only"
        )
    url = (
        "https://openrouter.ai/api/v1/models/"
        + quote(model.model, safe="/")
        + "/endpoints"
    )
    try:
        with httpx.Client(timeout=20, trust_env=False) as client:
            response = client.get(url)
            response.raise_for_status()
            metadata = response.json()
        return {**parameter_plan(model, metadata), "endpoint_metadata": metadata}
    except (httpx.HTTPError, ValueError) as exc:
        raise ConfigurationError(
            "Provider metadata validation failed: " + type(exc).__name__
        ) from exc
