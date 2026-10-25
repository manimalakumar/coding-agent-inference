"""Original public behavioral checks, not the historical private acceptance suite."""

import pytest
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.language_models.fake_chat_models import FakeListChatModel


@pytest.mark.parametrize("temperature", [0, 1, -1, 0.0, 0.5])
@pytest.mark.parametrize("model_type", [FakeListChatModel, FakeListLLM])
def test_supported_numeric_temperature_metadata(model_type, temperature):
    model = model_type(responses=["offline fixture"])
    assert (
        model._get_ls_params(temperature=temperature)["ls_temperature"] == temperature
    )
