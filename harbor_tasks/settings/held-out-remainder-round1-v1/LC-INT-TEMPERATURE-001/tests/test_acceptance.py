from typing import Any
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.language_models.fake import FakeListLLM

class Chat(FakeListChatModel):
    temperature: Any = None

class Completion(FakeListLLM):
    temperature: Any = None

@pytest.mark.parametrize("kind", [Chat, Completion])
@pytest.mark.parametrize("value", [0, 1, -1, 0.0, 0.25])
def test_numeric_keyword_temperature(kind,value):
    model=kind(responses=["unused"], temperature=0.7)
    result=model._get_ls_params(temperature=value,max_tokens=2048)
    assert result["ls_temperature"]==value
    assert result["ls_max_tokens"]==2048

@pytest.mark.parametrize("kind", [Chat, Completion])
@pytest.mark.parametrize("value", [0, 1, 0.25])
def test_numeric_configured_temperature(kind,value):
    assert kind(responses=["unused"],temperature=value)._get_ls_params()["ls_temperature"]==value

@pytest.mark.parametrize("kind", [Chat, Completion])
def test_invalid_keyword_falls_back_without_losing_numeric_attribute(kind):
    assert kind(responses=["unused"],temperature=1)._get_ls_params(temperature="invalid")["ls_temperature"]==1
    assert "ls_temperature" not in kind(responses=["unused"],temperature=None)._get_ls_params(temperature="invalid")
