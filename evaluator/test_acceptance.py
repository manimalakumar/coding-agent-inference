"""Trusted acceptance suite. Never mount into an evaluated agent's container."""

import httpx
import pytest
from langchain_openai import ChatOpenAI


def model(**kwargs):
    return ChatOpenAI(api_key="sk-offline-placeholder", model="gpt-4.1-mini", **kwargs)


@pytest.mark.parametrize("alias", ["request_timeout", "timeout"])
def test_custom_timeout_preserved(alias):
    timeout = httpx.Timeout(60.0, connect=5.0, read=43.0, write=17.0, pool=3.0)
    instance = model(**{alias: timeout})
    assert instance.request_timeout.as_dict() == timeout.as_dict()
    for client in [instance.root_client, instance.root_async_client]:
        assert client.timeout.as_dict() == timeout.as_dict()
        assert client._client.timeout.as_dict() == timeout.as_dict()


def test_default_client_reuse():
    a, b = model(), model()
    assert a.root_client._client is b.root_client._client
    assert a.root_async_client._client is b.root_async_client._client


@pytest.mark.parametrize("timeout", [None, 12.0, 45.0])
def test_equivalent_numeric_and_none_configs(timeout):
    a, b = model(request_timeout=timeout), model(request_timeout=timeout)
    assert a.root_client._client is b.root_client._client
    assert a.root_async_client._client is b.root_async_client._client


def test_different_base_urls_and_timeouts():
    instances = [
        model(base_url=url, request_timeout=timeout)
        for url in ["https://a.invalid/v1", "https://b.invalid/v1"]
        for timeout in [10.0, 30.0]
    ]
    for attr in ["root_client", "root_async_client"]:
        assert len({id(getattr(m, attr)._client) for m in instances}) == 4


def test_different_custom_timeout_values():
    a = model(request_timeout=httpx.Timeout(60.0, connect=5.0))
    b = model(request_timeout=httpx.Timeout(30.0, connect=2.0))
    for attr in ["root_client", "root_async_client"]:
        assert getattr(a, attr)._client.timeout.connect == 5.0
        assert getattr(b, attr)._client.timeout.connect == 2.0
        assert getattr(a, attr)._client.timeout.read == 60.0
        assert getattr(b, attr)._client.timeout.read == 30.0


def test_explicit_clients_preserved():
    sync = httpx.Client(timeout=11.0)
    asynchronous = httpx.AsyncClient(timeout=12.0)
    instance = model(http_client=sync, http_async_client=asynchronous)
    assert instance.root_client._client is sync
    assert instance.root_async_client._client is asynchronous
