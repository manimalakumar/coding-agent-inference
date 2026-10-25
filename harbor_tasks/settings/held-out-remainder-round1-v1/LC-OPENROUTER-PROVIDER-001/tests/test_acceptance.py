import pytest
from langchain_openrouter import ChatOpenRouter
from langchain_openrouter.chat_models import _create_stream_generation_info

@pytest.mark.parametrize("provider", ["Backend-A", "Backend-B", None, ""])
def test_nonstream_provider_is_actual_and_optional(provider):
    model=ChatOpenRouter(model="offline-model", api_key="offline-placeholder", openrouter_provider={"only":["Requested-Backend"]})
    response={"id":"offline", "model":"offline-model", "provider":provider, "choices":[{"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}]}
    msg=model._create_chat_result(response).generations[0].message
    assert msg.content=="hello" and msg.response_metadata["model_provider"]=="openrouter"
    if provider:assert msg.response_metadata["provider"]==provider
    else:assert "provider" not in msg.response_metadata

@pytest.mark.parametrize("provider", ["Backend-A", "Backend-B", None, ""])
def test_stream_generation_provider_is_actual_and_optional(provider):
    info=_create_stream_generation_info({"id":"offline","model":"offline-model","provider":provider},{"finish_reason":"stop"},"offline-model")
    assert info["finish_reason"]=="stop"
    if provider:assert info["provider"]==provider
    else:assert "provider" not in info

def test_absent_provider_does_not_create_metadata():
    model=ChatOpenRouter(model="offline-model",api_key="offline-placeholder")
    msg=model._create_chat_result({"choices":[{"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}]}).generations[0].message
    assert "provider" not in msg.response_metadata
