"""Offline feature probe using a custom counter."""
from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_core.messages import HumanMessage
from langchain.agents.middleware.context_editing import ContextEditingMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
calls = []
def counter(messages):
    calls.append(list(messages))
    return 0
messages = [HumanMessage(content="offline feature probe")]
request = ModelRequest(model=FakeChatModel(), messages=messages, tools=[], state={"messages":messages}, runtime=None, model_settings={})
middleware = ContextEditingMiddleware(token_counter=counter)
middleware.wrap_model_call(request, lambda req: ModelResponse(result=[]))
assert calls == [messages]
print("Custom token-counter initialization and execution succeeded")
