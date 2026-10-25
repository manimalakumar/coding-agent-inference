import pytest
from langchain_core.utils._merge import merge_lists, merge_dicts
from langchain_core.messages import AIMessageChunk

@pytest.mark.parametrize("ids,expected", [(("a","b"),2),(("a","a"),1),(("a",None),1),((None,"b"),1),((None,None),1)])
def test_identity_controls_merging(ids,expected):
    left={"index":0,"id":ids[0],"args":"A"}
    right={"index":0,"id":ids[1],"args":"B"}
    result=merge_lists([left],[right])
    assert len(result)==expected
    if expected==1:assert result[0]["args"]=="AB"
    else:assert [v["args"] for v in result]==["A","B"]

def test_many_parallel_calls_and_message_chunks():
    calls=[{"name":n,"id":n,"index":0,"args":"{}"} for n in ["a","b","c"]]
    assert [v["id"] for v in merge_lists(*[[c] for c in calls])]==["a","b","c"]
    combined=AIMessageChunk(content="",tool_call_chunks=[calls[0]])+AIMessageChunk(content="",tool_call_chunks=[calls[1]])
    assert [c["id"] for c in combined.tool_calls]==["a","b"]

def test_generic_fragment_concatenation_is_preserved():
    assert merge_dicts({"name":"read_","args":"{"},{"name":"file","args":"}"})=={"name":"read_file","args":"{}"}
    assert merge_lists([{"index":0,"args":"a"}],[{"index":0,"args":"b"}])==[{"index":0,"args":"ab"}]
