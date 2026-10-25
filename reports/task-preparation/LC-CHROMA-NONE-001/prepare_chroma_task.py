import json, shutil, subprocess, hashlib, time, copy, secrets
from pathlib import Path
import toml
from sandbox.prepare import export_snapshot
from sandbox.docker import digest, manifest
from agent.schemas import Task
R=Path.cwd(); ID='LC-CHROMA-NONE-001'; PREFIX='harbor-chroma-none-workspace-v1'
S=R / ".local/resources"/ID/'workspace-v1'; S.mkdir(parents=True,exist_ok=True)
T=R/'tasks'/ID; T.mkdir(exist_ok=True)
H=R/'harbor_tasks/settings'/PREFIX/ID
OLD=R/'harbor_tasks/settings/harbor-workspace-deliverable-v2/LC-ANNOTATED-TOOLS-001'
COMMIT='1ebcbf1d11578cb55db26013c745dc1e5722966e'
package='libs/partners/chroma'; source=package+'/langchain_chroma'; tests=package+'/tests/unit_tests'
oldtext=(OLD/'instruction.md').read_text()
intro='''Task ID: LC-CHROMA-NONE-001
Title: Handle missing document content in Chroma retrieval
Difficulty: Level 2

You are working in a supplied, unchanged historical LangChain source snapshot. Use only this workspace.

## Problem

Chroma-backed retrieval can raise a Pydantic validation error when a stored record has no document content (None). Public retriever invocation and other document-returning retrieval operations should return only valid documents rather than fail on such records.

Skip records whose document content is None. Preserve valid content, including the empty string, result order, metadata (with the existing empty-metadata default), document IDs, associated similarity scores and embedding vectors. Ensure consistent handling for scored search, vector-returning search and ID-based retrieval. Existing normal retrieval behavior must continue to work. Do not mutate the backend result payload.

## Reproduction and environment

Your first sandbox command must be exactly:

`python /task/reproduce.py`

This read-only script exercises retrieval with fabricated backend results offline and demonstrates the validation error. Add a regression test and observe it fail before changing production code. Preserve the regression when fixing the implementation.

Use the prepared Python 3.13.16 environment, Pydantic 2.12.5 and frozen dependencies. Imports must resolve to supplied workspace source. Do not install or upgrade packages. Use fake embeddings and mocked backend results. No model download, live service, network access or GPU is required.

A missing dependency, import failure or test-collection failure is an environment blocker, not evidence that the issue is invalid.

'''
text=intro+oldtext[oldtext.index('## Allowed changes'):]
text=text.replace('libs/core/langchain_core/tools/', source+'/').replace('libs/core/tests/unit_tests/',tests+'/').replace('tool argument descriptions','handling of absent Chroma document content').replace('/workspace/libs/core','/workspace/'+package).replace('tests/unit_tests/test_tools.py tests/unit_tests/utils/test_function_calling.py','tests/unit_tests').replace('new `test_*.py` files at `tests/unit_tests/`','new `test_*.py` files at `tests/unit_tests/`')
repro='''"""Offline reproduction using the public Chroma retriever."""
from unittest.mock import MagicMock
from langchain_chroma import Chroma
from langchain_core.embeddings import FakeEmbeddings

store = Chroma(embedding_function=FakeEmbeddings(size=3), client=MagicMock())
store._chroma_collection.query.return_value = {
    "documents": [["retained content", None]],
    "metadatas": [[{"kind": "valid"}, None]],
    "ids": [["valid-id", "missing-id"]],
    "distances": [[0.1, 0.2]],
}
result = store.as_retriever().invoke("offline query")
assert len(result) == 1 and result[0].page_content == "retained content"
assert result[0].id == "valid-id" and result[0].metadata == {"kind": "valid"}
print("Missing-content retrieval succeeded")
'''
(T/'reproduce.py').write_text(repro)
task=Task(task_id=ID, commit=COMMIT, package=package, instructions=text,source_paths=[source],test_paths=[tests],development_checks=[{'name':'existing_tests','command':'python -m pytest tests/unit_tests -q --disable-socket --allow-unix-socket'}],reproduction_command='python /task/reproduce.py',reproduction_script='reproduce.py',expected_failure='ValidationError',runtime_versions={'pydantic':'2.12.5','langchain-chroma':'0.2.4'},workspace_imports=['langchain_chroma','langchain_core'])
(T/'task.json').write_text(task.model_dump_json(indent=2))
isolation=(R/'tasks/LC-ANNOTATED-TOOLS-001/isolation.py').read_text().replace('/workspace/libs/core','/workspace/'+package).replace('langchain_core/tools/base.py','langchain_chroma/vectorstores.py').replace('langchain_core/tools/','langchain_chroma/').replace('test_tools.py','test_vectorstores.py').replace('tests/unit_tests/utils','tests/unit_tests/__init__.py')
(T/'isolation.py').write_text(isolation)
if not (S/'provenance.json').exists():
 if (S/'baseline').exists():
  for p in (S/'baseline').rglob('*'): p.chmod(0o755 if p.is_dir() else 0o644)
  (S/'baseline').chmod(0o755); shutil.rmtree(S/'baseline')
 p=export_snapshot(R / ".local/upstream/langchain",task,S/'baseline'); (S/'provenance.json').write_text(json.dumps(p,indent=2))
B=S/'baseline'; pyproject=toml.loads((B/package/'pyproject.toml').read_text()); core=toml.loads((B/'libs/core/pyproject.toml').read_text())['project']['version']
C=S/'dependency-build'; C.mkdir(exist_ok=True)
# Reuse the prepared Python 3.13 environment, pin all existing runtime versions.
base_setup=json.loads((R / ".local/resources/LC-ANNOTATED-TOOLS-001/c4989fbedda7c396/setup.json").read_text())
base=base_setup['image_id']
(C/'Dockerfile').write_text(f'''FROM {base}
USER root
COPY source /workspace
RUN /opt/venv/bin/python -m ensurepip && /opt/venv/bin/python -m pip freeze | sed '/^langchain-/d' > /opt/preserved-constraints.txt
RUN /opt/venv/bin/python -m pip install --constraint /opt/preserved-constraints.txt 'chromadb==1.0.9' 'onnxruntime==1.22.0' 'pdm-backend==2.4.4'
RUN /opt/venv/bin/python -m pip install --no-deps --no-build-isolation -e /workspace/libs/core -e /workspace/libs/standard-tests -e /workspace/libs/partners/openai -e /workspace/libs/partners/chroma
RUN /opt/venv/bin/python -m pip freeze > /opt/requirements.lock && /opt/venv/bin/python -c 'import importlib.metadata as m,json,sys; print(json.dumps({{"python":sys.version,"packages":sorted((d.metadata["Name"],d.version) for d in m.distributions())}},indent=2))' > /opt/environment.json
ENV PYTHONPATH=/workspace/libs/partners/chroma:/workspace/libs/core:/workspace/libs/standard-tests:/workspace/libs/partners/openai ANONYMIZED_TELEMETRY=False
ENV HOME=/tmp PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS="-p no:cacheprovider --no-cov"
WORKDIR /workspace/libs/partners/chroma
USER 65532:65532
''')
if not (C/'source').exists(): shutil.copytree(B,C/'source')
(S/'preparation-state.json').write_text(json.dumps({'task_id':ID,'core_version':core,'base_image':base,'starting_commit':COMMIT,'baseline_checksum':digest(manifest(B)),'setup_started_at':time.time(),'setup_model_calls':0},indent=2))
print(S)
