"""Offline contracts for parameter descriptions and unchanged tool behavior."""

import copy
import sys
import types
from uuid import uuid4

import pytest
from langchain_core.tools import tool
from langchain_core.tools.base import _infer_arg_descriptions


def fixture_function(*, postponed=True, source=None):
    name = "annotation_fixture_" + uuid4().hex
    module = types.ModuleType(name)
    sys.modules[name] = module
    code = "from __future__ import annotations\n" if postponed else ""
    code += "from typing import Annotated\nfrom typing_extensions import Annotated as ExtensionAnnotated\nAlias = list[str]\n"
    code += (
        source
        or '''def fragments(parts: Annotated[Alias, "Fixture fragment description"]) -> list[str]:
    """Return fragments without performing I/O."""
    return parts
'''
    )
    exec(
        compile(code, "<annotation-fixture>", "exec", dont_inherit=True),
        module.__dict__,
    )
    return module.fragments


@pytest.mark.parametrize("postponed", [False, True])
def test_public_tool_description_and_invocation(postponed):
    fn = fixture_function(postponed=postponed)
    original = copy.deepcopy(fn.__annotations__)
    wrapped = tool(fn)
    assert wrapped.args["parts"] == {
        "description": "Fixture fragment description",
        "items": {"type": "string"},
        "title": "Parts",
        "type": "array",
    }
    assert wrapped.invoke({"parts": ["one", "two"]}) == ["one", "two"]
    assert fn.__annotations__ == original


@pytest.mark.parametrize(
    "annotation",
    [
        'Annotated[str, "Plain text description"]',
        'ExtensionAnnotated[str, "Plain text description"]',
        'Annotated[str, 7, "Plain text description"]',
    ],
)
@pytest.mark.parametrize("postponed", [False, True])
def test_description_metadata(annotation, postponed):
    fn = fixture_function(
        postponed=postponed,
        source=f'''def fragments(query: {annotation}) -> str:
    """Return query."""
    return query
''',
    )
    description, args = _infer_arg_descriptions(fn)
    assert description == "Return query."
    assert args == {"query": "Plain text description"}
    assert tool(fn).args["query"]["description"] == "Plain text description"


def test_multiple_parameters_preserve_types():
    fn = fixture_function(
        source='''def fragments(query: Annotated[str, "Query text"], limit: Annotated[int, "Result limit"] = 3) -> str:
    """Return query."""
    return query * limit
'''
    )
    wrapped = tool(fn)
    assert wrapped.args["query"]["description"] == "Query text"
    assert wrapped.args["limit"]["description"] == "Result limit"
    assert wrapped.args["query"]["type"] == "string"
    assert wrapped.args["limit"]["type"] == "integer"
    assert wrapped.invoke({"query": "x"}) == "xxx"


def test_parsed_docstring_takes_precedence():
    fn = fixture_function(
        source='''def fragments(query: Annotated[str, "Annotation description"]) -> str:
    """Return query.

    Args:
        query: Explicit docstring description.
    """
    return query
'''
    )
    _, args = _infer_arg_descriptions(
        fn, parse_docstring=True, error_on_invalid_docstring=True
    )
    assert args == {"query": "Explicit docstring description."}


def test_unannotated_metadata_is_not_invented():
    fn = fixture_function(
        source='''def fragments(query: str) -> str:
    """Return query."""
    return query
'''
    )
    assert _infer_arg_descriptions(fn)[1] == {}
    wrapped = tool(fn)
    assert wrapped.args["query"]["type"] == "string"
    assert "description" not in wrapped.args["query"]


@pytest.mark.asyncio
async def test_async_tool_description_and_invocation():
    fn = fixture_function(
        source='''async def fragments(query: Annotated[str, "Async query description"]) -> str:
    """Return query asynchronously."""
    return query
'''
    )
    wrapped = tool(fn)
    assert wrapped.args["query"]["description"] == "Async query description"
    assert await wrapped.ainvoke({"query": "offline"}) == "offline"
