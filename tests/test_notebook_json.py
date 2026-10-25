"""The readable view decodes JSON strings, never interprets them as HTML."""

import copy

from experiments.notebook_json import readable_json


def test_strings_have_real_line_breaks_and_evidence_is_unchanged():
    request = {"messages": [{"role": "user", "content": "Problem\n\nRun tests"}]}
    before = copy.deepcopy(request)
    output = readable_json(request)
    assert "Problem\n\nRun tests" in output
    assert "Problem\\n" not in output
    assert "<details open" in output
    assert "white-space:pre-wrap" in output
    assert request == before


def test_labels_and_values_are_escaped_and_missing_request_is_explicit():
    output = readable_json({"<script>": "<img src='https://example.com'>"})
    assert "<script>" not in output and "<img" not in output
    assert "&lt;script&gt;" in output and "&lt;img" in output
    assert "No request recorded" in readable_json(None)
