"""Trusted behavioral checks; never included in the evaluated-agent image."""
import asyncio
from functools import partial
from unittest.mock import Mock

import pytest
from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain.agents.middleware.context_editing import ClearToolUsesEdit, ContextEditingMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse


def request(messages=None):
    if messages is None:
        messages = [AIMessage(content='', tool_calls=[{'id':'a','name':'lookup','args':{'q':'x'}}]), ToolMessage(content='中文資料'*40,tool_call_id='a')]
    return ModelRequest(model=FakeChatModel(), messages=messages, system_message=SystemMessage(content='system context'), tools=[], state={'messages':messages}, runtime=None, model_settings={})


def execute(middleware, req, asynchronous):
    seen=[]
    result=ModelResponse(result=[AIMessage(content='done')])
    def handler(r):
        seen.append(r)
        return result
    async def ahandler(r):
        return handler(r)
    actual=asyncio.run(middleware.awrap_model_call(req, ahandler)) if asynchronous else middleware.wrap_model_call(req, handler)
    assert actual is result
    assert len(seen)==1
    return seen[0]


@pytest.mark.parametrize('asynchronous',[False,True])
@pytest.mark.parametrize('method',['approximate','model'])
def test_custom_counter_controls_edits(asynchronous, method, monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('default counter must not be used')
    monkeypatch.setattr(FakeChatModel,'get_num_tokens_from_messages',forbidden)
    monkeypatch.setattr('langchain.agents.middleware.context_editing.count_tokens_approximately',forbidden)
    calls=[]
    def counter(messages):
        calls.append([m.model_copy(deep=True) for m in messages])
        return 200 if messages[-1].content!='[cleared]' else 10
    middleware=ContextEditingMiddleware(token_counter=counter,token_count_method=method,edits=[ClearToolUsesEdit(trigger=100,keep=0,clear_at_least=100)])
    req=request(); original=[m.model_copy(deep=True) for m in req.messages]
    result=execute(middleware,req,asynchronous)
    assert result.messages[-1].content=='[cleared]'
    assert req.messages==original
    assert len(calls)>=2 and calls[0]==original


@pytest.mark.parametrize('asynchronous',[False,True])
def test_custom_counter_can_prevent_edit(asynchronous):
    middleware=ContextEditingMiddleware(token_counter=lambda messages:0,edits=[ClearToolUsesEdit(trigger=1,keep=0)])
    req=request();result=execute(middleware,req,asynchronous)
    assert result.messages==req.messages


@pytest.mark.parametrize('asynchronous',[False,True])
def test_falsy_callable_and_edit_order(asynchronous):
    class Counter:
        def __bool__(self):return False
        def __call__(self,messages):return len(messages)
    counter=Counter();seen=[]
    class Edit:
        def __init__(self,n):self.n=n
        def apply(self,messages,*,count_tokens):
            assert count_tokens is counter
            seen.append((self.n,count_tokens(messages)))
            messages.append(HumanMessage(content=str(self.n)))
    result=execute(ContextEditingMiddleware(token_counter=counter,edits=[Edit(1),Edit(2)]),request(),asynchronous)
    assert seen==[(1,2),(2,3)] and len(result.messages)==4


@pytest.mark.parametrize('asynchronous',[False,True])
def test_partial_counter(asynchronous):
    counter=partial(count_tokens_approximately,chars_per_token=2.0)
    req=request();result=execute(ContextEditingMiddleware(token_counter=counter,edits=[ClearToolUsesEdit(trigger=20,keep=0)]),req,asynchronous)
    assert result.messages[-1].content=='[cleared]'


@pytest.mark.parametrize('asynchronous',[False,True])
@pytest.mark.parametrize('method',['approximate','model'])
def test_none_preserves_existing_counting(asynchronous,method,monkeypatch):
    seen=[]
    def approximate(messages):seen.append(list(messages));return 0
    def model(self,messages,tools=None):seen.append((list(messages),tools));return 0
    monkeypatch.setattr('langchain.agents.middleware.context_editing.count_tokens_approximately',approximate)
    monkeypatch.setattr(FakeChatModel,'get_num_tokens_from_messages',model)
    req=request();execute(ContextEditingMiddleware(token_counter=None,token_count_method=method),req,asynchronous)
    assert seen==([req.messages] if method=='approximate' else [([req.system_message]+req.messages,req.tools)])


@pytest.mark.parametrize('asynchronous',[False,True])
def test_empty_messages_bypass_counter(asynchronous):
    counter=Mock(side_effect=AssertionError('empty conversation'))
    execute(ContextEditingMiddleware(token_counter=counter),request([]),asynchronous)
    counter.assert_not_called()


@pytest.mark.parametrize('asynchronous',[False,True])
def test_counter_exception_propagates(asynchronous):
    def counter(messages):raise ValueError('counter signal')
    with pytest.raises(ValueError,match='counter signal'):
        execute(ContextEditingMiddleware(token_counter=counter),request(),asynchronous)
