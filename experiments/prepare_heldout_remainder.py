"""Trusted setup for round 1 of six additional held-out tasks.

No evaluated agent imports this module or receives reference material.
"""
import ast
import json
import shutil
import subprocess
import time
from pathlib import Path

from agent.schemas import Task
from sandbox.prepare import export_snapshot

ROOT=Path(__file__).resolve().parents[1]
LABEL="held-out-remainder-round1-v1"
TRUST=ROOT / ".local/resources"/LABEL
REFERENCE=ROOT / ".local/upstream/langchain"


def write(path,text):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)


def git(*args):
    return subprocess.check_output(["git","-C",str(REFERENCE),*args],text=True,stderr=subprocess.DEVNULL)


TASKS={
 "LC-FIREWORKS-RERANK-001": {
  "issue":39340,"pr":39732,"family":"feature","package":"libs/partners/fireworks","sources":["libs/partners/fireworks/langchain_fireworks"],"tests":["libs/partners/fireworks/tests/unit_tests"],"existing":[["libs/partners/fireworks","tests/unit_tests/test_embeddings.py"],["libs/partners/fireworks","tests/unit_tests/test_llms.py"],["libs/partners/fireworks","tests/unit_tests/test_chat_models.py"]],"expected":"FIREWORKS_RERANK_MISSING",
  "requirements":'''Add FireworksRerank, a BaseDocumentCompressor exported from langchain_fireworks. Required API: FireworksRerank(model: str, top_n: int | None = 3, fireworks_api_key: SecretStr | None = environment FIREWORKS_API_KEY, base_url: str = "https://api.fireworks.ai/inference/v1", user_agent: str = "langchain:partner", client=None, async_client=None). Support rerank/arerank(documents, query, *, rank_fields=None, model=None, top_n=-1, task=None), returning index/relevance_score dictionaries, and compress_documents/acompress_documents(documents, query, callbacks=None). Use OpenAI-compatible injected sync/async clients with post("/rerank", cast_to=dict[str, Any], body=payload). API payload: model, query, string documents, return_documents=False, optional top_n and task. Omitted/non-positive top_n uses configured default; None requests all results. Strings stay strings, Document uses page_content, mappings serialize as JSON (optionally selected rank_fields). Empty input returns [] without API access. Compression returns new Document copies in response order with original content/metadata preserved and relevance_score added, without mutating inputs. Require credentials unless both clients are injected; preserve supplied clients and construct only missing ones. Forbid unknown configuration fields. Preserve all existing public exports and unrelated integrations. No new dependencies. Use mocks only; do not contact Fireworks.''',
  "reproduce":'''import langchain_fireworks as module
assert hasattr(module, "FireworksRerank"), "FIREWORKS_RERANK_MISSING"
from langchain_core.documents import Document
class Client:
    def post(self, path, **kwargs):
        return {"data":[{"index":1,"relevance_score":0.9}]}
model=module.FireworksRerank(model="offline-reranker", client=Client(), async_client=Client())
docs=[Document(page_content="a"), Document(page_content="b",metadata={"source":"b"})]
result=model.compress_documents(docs,"query")
assert result[0].page_content=="b" and result[0].metadata["relevance_score"]==0.9
print("Offline rerank API and compression succeeded")
''',
 },
 "LC-OPENROUTER-PROVIDER-001": {
  "issue":39298,"pr":39301,"family":"feature","package":"libs/partners/openrouter","sources":["libs/partners/openrouter/langchain_openrouter"],"tests":["libs/partners/openrouter/tests/unit_tests"],"existing":[["libs/partners/openrouter","tests/unit_tests/test_chat_models.py"]],"expected":"SERVING_PROVIDER_MISSING",
  "requirements":'''Preserve the actual serving backend from the top-level OpenRouter response provider field under response_metadata["provider"]. Cover standard completion conversion and synchronous/asynchronous streaming. Keep model_provider="openrouter" as the integration identity, distinct from the actual backend. Do not echo requested routing settings as serving provenance. Include provider only when supplied/truthy; absent, None or empty provider must leave existing behavior unchanged. Preserve content, IDs, model, finish reasons, fingerprint, usage, cost and other metadata. No new API calls, network requests, dependencies or unrelated public API changes. Use fabricated response/chunk dictionaries and mocks offline.''',
  "reproduce":'''from langchain_openrouter import ChatOpenRouter
model=ChatOpenRouter(model="offline-model", api_key="offline-placeholder")
response={"id":"offline", "model":"offline-model", "provider":"Backend-A", "choices":[{"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}]}
message=model._create_chat_result(response).generations[0].message
assert message.response_metadata.get("provider")=="Backend-A", "SERVING_PROVIDER_MISSING"
print("Serving provider preserved")
''',
 },
 "LC-HF-ACCELERATOR-001": {
  "issue":38855,"pr":39640,"family":"feature","package":"libs/partners/huggingface","sources":["libs/partners/huggingface/langchain_huggingface/llms"],"tests":["libs/partners/huggingface/tests/unit_tests"],"existing":[["libs/partners/huggingface","tests/unit_tests/test_huggingface_pipeline.py"]],"expected":"ACCELERATOR_DISCOVERY_MISSING",
  "requirements":'''HuggingFacePipeline.from_model_id must support non-CUDA accelerators. Prefer torch.accelerator.device_count() when that API is available, falling back to torch.cuda.device_count() for older PyTorch. Apply device-agnostic discovery consistently to device validation and available-accelerator warnings. Permit valid non-CUDA device indices even when CUDA count is zero; reject out-of-range and below--1 indices; preserve CPU (-1), device_map, quantized-model and existing backend behavior. Update warnings to accelerator-neutral language. No GPU, model download or real model inference is needed: mock transformers loading/pipeline and hardware APIs. No new dependencies or unrelated API changes.''',
  "reproduce":'''from unittest.mock import MagicMock, patch
# Materialize Transformers lazy exports before mocking them.
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
accelerator=MagicMock();accelerator.device_count.return_value=2
with patch("transformers.AutoTokenizer.from_pretrained",return_value=MagicMock(pad_token_id=0)), patch("transformers.AutoModelForCausalLM.from_pretrained",return_value=MagicMock(is_loaded_in_4bit=False,is_loaded_in_8bit=False)), patch("transformers.pipeline",return_value=MagicMock(task="text-generation")), patch("torch.accelerator",accelerator), patch("torch.cuda.device_count",return_value=0):
    try:
        HuggingFacePipeline.from_model_id(model_id="offline-mock-model",task="text-generation",device=0)
    except ValueError as exc:
        raise AssertionError("ACCELERATOR_DISCOVERY_MISSING") from exc
assert accelerator.device_count.called
print("Non-CUDA discovery succeeded with mocked hardware")
''',
 },
 "LC-JSON-INPUT-001": {
  "issue":39192,"pr":39238,"family":"bug_fix","package":"libs/text-splitters","sources":["libs/text-splitters/langchain_text_splitters"],"tests":["libs/text-splitters/tests/unit_tests"],"existing":[["libs/text-splitters","tests/unit_tests/test_text_splitters.py -k split_json"]],"expected":"INVALID_JSON_INPUT_SILENTLY_DROPPED","type_command":["ty","check"],
  "requirements":'''RecursiveJsonSplitter.split_json must reject unsupported non-dict top-level values with TypeError rather than silently discarding them. Top-level lists require convert_lists=True; the error for a bare list should name its type and explain that option. With conversion enabled, lists must continue to produce the existing converted dictionary chunks. Reject other scalars (str/int/bool etc.) with their type in the message; do not suggest list conversion when it cannot help. Preserve None -> [] with or without conversion, {} -> [], valid dictionary splitting, nested empty values and chunk-boundary behavior. Entry points delegating to split_json must inherit validation. Keep existing signatures and dependencies. Use the prepared package-native strict type checker, ty check (all rules error), for changed production files.''',
  "reproduce":'''from langchain_text_splitters import RecursiveJsonSplitter
splitter=RecursiveJsonSplitter(max_chunk_size=50)
try:
    splitter.split_json([{"a":1},{"b":2}])
except TypeError:
    pass
else:
    raise AssertionError("INVALID_JSON_INPUT_SILENTLY_DROPPED")
assert splitter.split_json([{"a":1},{"b":2}],convert_lists=True)==[{"0":{"a":1},"1":{"b":2}}]
print("Unsupported input rejected without losing convertible lists")
''',
 },
 "LC-INT-TEMPERATURE-001": {
  "issue":35300,"pr":35302,"family":"bug_fix","package":"libs/core","sources":["libs/core/langchain_core/language_models"],"tests":["libs/core/tests/unit_tests"],"existing":[["libs/core","tests/unit_tests/language_models/chat_models/test_base.py"],["libs/core","tests/unit_tests/language_models/llms/test_base.py"]],"expected":"INTEGER_TEMPERATURE_DROPPED",
  "requirements":'''BaseChatModel._get_ls_params and BaseLLM._get_ls_params must preserve numeric integer temperature values (including 0 and 1) in LangSmith ls_temperature metadata, as they do floats. Handle both explicit keyword arguments and configured instance attributes, preserving numeric values and explicit numeric-argument precedence. Preserve existing handling of invalid/non-numeric values, fallback attributes, max_tokens, model/provider metadata and stop sequences. Do not change invocation behavior or make tracing/network requests. Add offline regression coverage for both model families. No new dependencies or unrelated API changes.''',
  "reproduce":'''from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.language_models.fake import FakeListLLM
for model in [FakeListChatModel(responses=["ok"]),FakeListLLM(responses=["ok"])]:
    assert model._get_ls_params(temperature=0).get("ls_temperature")==0, "INTEGER_TEMPERATURE_DROPPED"
print("Integer temperature retained for both model families")
''',
 },
 "LC-PARALLEL-TOOL-MERGE-001": {
  "issue":34807,"pr":35281,"family":"bug_fix","package":"libs/core","sources":["libs/core/langchain_core/utils"],"tests":["libs/core/tests/unit_tests"],"existing":[["libs/core","tests/unit_tests/test_messages.py"],["libs/core","tests/unit_tests/utils/test_utils.py"]],"expected":"DISTINCT_TOOL_CALLS_MERGED",
  "requirements":'''Keep distinct parallel tool-call chunks separate when they share an index but carry different non-None IDs. Preserve name, ID, argument fragments and ordering for each call, including more than two calls and later continuation chunks. Legitimate streaming fragments must still merge when IDs match or an ID is absent/None. Preserve normal string/argument-fragment concatenation, legacy no-ID merging, special string-index behavior and generic dictionary/list merge semantics. The task concerns incorrect aggregation of distinct tool calls, not suppressing all name or argument concatenation globally. Cover utility-level and message-chunk behavior offline. No new dependencies or unrelated APIs.''',
  "reproduce":'''from langchain_core.utils._merge import merge_lists
left={"index":0,"name":"read_file","id":"call-A","args":"{}"}
right={"index":0,"name":"search_text","id":"call-B","args":"{}"}
result=merge_lists([left],[right])
assert len(result)==2 and [v["id"] for v in result]==["call-A","call-B"], "DISTINCT_TOOL_CALLS_MERGED"
print("Distinct parallel tool calls remain separate")
''',
 },
}


def segment(text,node):
    first=min([node.lineno,*[d.lineno for d in getattr(node,"decorator_list",[])]])
    return "\n".join(text.splitlines()[first-1:node.end_lineno])


def export_all():
    for tid,spec in TASKS.items():
        state=TRUST/tid
        if (state/"export.json").exists():continue
        started=time.time();pr=json.loads((TRUST/f"reference-pr-{spec['pr']}.json").read_text())["pull_request"]
        commit=git("rev-parse",pr["merge_commit_sha"]+"^1").strip()
        task=Task(task_id=tid,commit=commit,package=spec["package"],instructions=spec["requirements"],source_paths=spec["sources"],test_paths=spec["tests"],development_checks=[],reproduction_command="python /task/reproduce.py",reproduction_script="reproduce.py",expected_failure=spec["expected"])
        state.mkdir(parents=True,exist_ok=True);provenance=export_snapshot(REFERENCE,task,state/"baseline")
        write(state/"provenance.json",json.dumps(provenance,indent=2));write(state/"reproduce.py",spec["reproduce"])
        source={};testfiles={}
        for name in git("diff","--name-only",commit,pr["merge_commit_sha"]).splitlines():
            if any(name.startswith(p+"/") for p in spec["sources"]):source[name]=git("show",pr["merge_commit_sha"]+":"+name)
            if "/unit_tests/" in name and name.endswith(".py"):testfiles[name]=git("show",pr["merge_commit_sha"]+":"+name)
        write(state/"reference-source.json",json.dumps(source,indent=2));write(state/"reference-tests.json",json.dumps(testfiles,indent=2))
        context=state/"dependency-build";shutil.copytree(state/"baseline",context/"source")
        recipe=f'''FROM ghcr.io/astral-sh/uv:0.11.29 AS uv
FROM python:3.14.6-slim-bookworm
COPY --from=uv /uv /usr/local/bin/uv
COPY source /workspace
WORKDIR /workspace/{spec['package']}
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_DOWNLOADS=never
RUN uv sync --frozen --group test --group lint --group typing --group test_integration
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN mkdir -p /opt/tiktoken && /opt/venv/bin/python -c 'import importlib.util; exec("import tiktoken; tiktoken.get_encoding(\\\"o200k_base\\\"); tiktoken.get_encoding(\\\"cl100k_base\\\")") if importlib.util.find_spec("tiktoken") else None'
RUN uv pip freeze --python /opt/venv/bin/python > /opt/requirements.lock && /opt/venv/bin/python -c 'import importlib.metadata as m,json,sys; print(json.dumps({{"python":sys.version,"packages":sorted((d.metadata["Name"],d.version) for d in m.distributions())}},indent=2))' > /opt/environment.json
ENV PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin HOME=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/workspace/{spec['package']}:/workspace/libs/core:/workspace/libs/standard-tests:/workspace/libs/partners/openai
ENV PYTEST_ADDOPTS="-p no:cacheprovider --no-cov"
USER 65532:65532
'''
        write(context/"Dockerfile",recipe)
        record={"task_id":tid,"spec":spec,"commit":commit,"baseline_checksum":provenance["baseline_checksum"],"trusted_reference_commit":pr["merge_commit_sha"],"setup_started_at":started,"model_requests":0,"status":"exported_not_validated"}
        write(state/"export.json",json.dumps(record,indent=2));print(tid,commit,flush=True)


if __name__=="__main__":export_all()
