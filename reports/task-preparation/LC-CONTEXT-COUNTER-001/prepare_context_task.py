import json,shutil,subprocess,time
from pathlib import Path
from agent.schemas import Task
from sandbox.prepare import export_snapshot
R=Path.cwd(); ID='LC-CONTEXT-COUNTER-001'; S=R / ".local/resources"/ID/'feature-v1';S.mkdir(parents=True,exist_ok=True)
T=R/'tasks'/ID; package='libs/langchain_v1'; source=package+'/langchain/agents/middleware'; tests=package+'/tests/unit_tests';commit='2019bf5ebe50324c548f67c2666a804343f9b772'
old=(R/'harbor_tasks/settings/comparison-4x2x3-v1/LC-CHROMA-NONE-001/instruction.md').read_text()
intro='''Task ID: LC-CONTEXT-COUNTER-001
Title: Support a custom token counter in ContextEditingMiddleware
Task type: Feature request

Use only the supplied unchanged source snapshot.

## Feature request and required API

Add this optional keyword-only constructor parameter, using the existing TokenCounter type:

ContextEditingMiddleware(..., token_counter: TokenCounter | None = None)

The middleware currently selects approximate counting or model counting internally. Approximate counting can under-count CJK-heavy conversations, and compatible providers may not implement model token counting. Allow callers to supply a synchronous custom message counter, including functools.partial(count_tokens_approximately, chars_per_token=2.0).

When supplied, token_counter takes precedence over token_count_method for context-edit triggers and subsequent counting performed by edits. Support synchronous and asynchronous model-call wrappers consistently. Preserve the configured counter and honor callable objects regardless of their truth value. When omitted or None, retain the existing approximate and model counting behavior, including handling of system messages and tools in model mode. Preserve empty-conversation behavior, edit ordering, original request immutability, normal handler results and exception propagation. Document the new parameter and precedence. No new dependencies or unrelated API changes.

## Reproduction and environment

Your first sandbox command must be exactly: `python /task/reproduce.py`.
This offline feature probe demonstrates the missing constructor API. Add new regression tests and observe failure before implementation. Preserve them in the final workspace. Use prepared Python 3.14.6 and exact frozen dependencies. langchain and langchain_core must import from this workspace. Do not install or upgrade packages. Use fake models, deterministic counters and fabricated messages; no live inference or network request is required.

An import, dependency or collection failure is an environment blocker, not evidence against the feature request.

'''
text=intro+old[old.index('## Allowed changes'):]
text=text.replace('libs/partners/chroma/langchain_chroma/',source+'/').replace('libs/partners/chroma/tests/unit_tests/',tests+'/').replace('handling of absent Chroma document content','this custom token-counter feature').replace('unrelated packages or public APIs','unrelated packages or public APIs beyond the requested optional parameter').replace('/workspace/libs/partners/chroma','/workspace/'+package).replace('python -m pytest tests/unit_tests -q','python -m pytest tests/unit_tests/agents/middleware/implementations/test_context_editing.py -q')
repro='''"""Offline probe for the requested constructor API."""
from langchain.agents.middleware.context_editing import ContextEditingMiddleware
from langchain_core.messages import HumanMessage

calls = []
def count(messages):
    calls.append(list(messages))
    return 100

middleware = ContextEditingMiddleware(token_counter=count)
assert middleware is not None
print("Custom token-counter initialization succeeded")
'''
(T/'reproduce.py').write_text(repro)
task=Task(task_id=ID,commit=commit,package=package,instructions=text,source_paths=[source],test_paths=[tests],development_checks=[{'name':'existing_tests','command':'python -m pytest tests/unit_tests/agents/middleware/implementations/test_context_editing.py -q --disable-socket --allow-unix-socket'}],reproduction_command='python /task/reproduce.py',reproduction_script='reproduce.py',expected_failure="unexpected keyword argument 'token_counter'",runtime_versions={'langchain':'1.3.15','langchain-core':'1.5.6'},workspace_imports=['langchain','langchain_core'])
(T/'task.json').write_text(task.model_dump_json(indent=2));(T/'instruction.md').write_text(text)
iso=(R/'tasks/LC-CHROMA-NONE-001/isolation.py').read_text().replace('libs/partners/chroma',package).replace('langchain_chroma/vectorstores.py','langchain/agents/middleware/context_editing.py').replace('langchain_chroma/','langchain/agents/middleware/').replace('test_vectorstores.py','agents/middleware/implementations/test_context_editing.py')
(T/'isolation.py').write_text(iso)
if not (S/'provenance.json').exists():
 p=export_snapshot(R / ".local/upstream/langchain",task,S/'baseline');(S/'provenance.json').write_text(json.dumps(p,indent=2))
C=S/'dependency-build';C.mkdir(exist_ok=True)
if not (C/'source').exists():shutil.copytree(S/'baseline',C/'source')
(C/'Dockerfile').write_text('''FROM ghcr.io/astral-sh/uv:0.11.29 AS uv
FROM python:3.14.6-slim-bookworm
COPY --from=uv /uv /usr/local/bin/uv
COPY source /workspace
WORKDIR /workspace/libs/langchain_v1
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_DOWNLOADS=never
RUN uv sync --frozen --group test --group lint --group typing --group test_integration
RUN uv pip freeze --python /opt/venv/bin/python > /opt/requirements.lock && /opt/venv/bin/python -c 'import importlib.metadata as m,json,sys; print(json.dumps({"python":sys.version,"packages":sorted((d.metadata["Name"],d.version) for d in m.distributions())},indent=2))' > /opt/environment.json
ENV PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin HOME=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/workspace/libs/langchain_v1:/workspace/libs/core:/workspace/libs/standard-tests:/workspace/libs/partners/openai
ENV PYTEST_ADDOPTS="-p no:cacheprovider --no-cov"
USER 65532:65532
''')
(S/'preparation-state.json').write_text(json.dumps({'task_id':ID,'starting_commit':commit,'setup_started_at':time.time(),'setup_model_calls':0},indent=2))
print(S)
