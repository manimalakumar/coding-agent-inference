from langchain_core.language_models.fake import FakeListLLM
from langchain_core.language_models.fake_chat_models import FakeListChatModel

for model in [FakeListChatModel(responses=["ok"]), FakeListLLM(responses=["ok"])]:
    assert model._get_ls_params(temperature=0).get("ls_temperature") == 0, (
        "INTEGER_TEMPERATURE_DROPPED"
    )
print("Integer temperature retained for both model families")
