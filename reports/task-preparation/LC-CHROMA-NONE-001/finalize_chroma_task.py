exec(open('/tmp/prepare_chroma_task.py').read().split("if not (S/'provenance.json').exists():")[0])
import shlex
from harbor_integration.prepare_resources import build
H.mkdir(parents=True,exist_ok=True)
for d in ['tests','environment']: (H/d).mkdir(exist_ok=True)
for n in ['docker-compose.yaml']:
 shutil.copyfile(OLD/'environment'/n,H/'environment'/n)
for n in ['docker-compose.yaml','test.sh','workspace_policy.py','completion-policy.json']:
 shutil.copyfile(OLD/'tests'/n,H/'tests'/n)
compose=(H/'environment/docker-compose.yaml').read_text().replace('libs/core/langchain_core/tools',source).replace('libs/core/tests/unit_tests',tests)
(H/'environment/docker-compose.yaml').write_text(compose)
(H/'instruction.md').write_text(text)
accept='''"""Independent offline acceptance checks; never provided to the coding agent."""
from copy import deepcopy
from unittest.mock import MagicMock
import numpy as np
import pytest
from langchain_chroma import Chroma
from langchain_chroma.vectorstores import _results_to_docs_and_scores, _results_to_docs_and_vectors
from langchain_core.embeddings import FakeEmbeddings

@pytest.mark.parametrize("documents", [[None, "kept", ""], [None, None, None], ["first", "second", "third"]])
@pytest.mark.parametrize("kind", ["scores", "vectors", "ids", "retriever"])
def test_retrieval(documents, kind):
    metadata=[{} if kind == "ids" else None, {"n": 2}, {"n": 3}]
    ids=["a", "b", "c"]
    vectors=[[1., 2.], [3., 4.], [5., 6.]]
    scores=[.1, .2, .3]
    data={"documents":[documents], "metadatas":[metadata], "ids":[ids], "distances":[scores], "embeddings":[vectors]}
    original=deepcopy(data)
    store=Chroma(client=MagicMock(), embedding_function=FakeEmbeddings(size=2))
    if kind == "scores":
        output=_results_to_docs_and_scores(data)
    elif kind == "vectors":
        output=_results_to_docs_and_vectors(data)
    elif kind == "ids":
        store._chroma_collection.get.return_value={"documents":documents,"metadatas":metadata,"ids":ids}
        output=[(d, None) for d in store.get_by_ids(ids)]
    else:
        store._chroma_collection.query.return_value=data
        output=[(d, None) for d in store.as_retriever().invoke("offline")]
    expected=[i for i,d in enumerate(documents) if d is not None]
    assert len(output)==len(expected)
    for (document,associated),i in zip(output,expected):
        assert document.page_content==documents[i]
        assert document.metadata==(metadata[i] or {})
        assert document.id==ids[i]
        if kind=="scores": assert associated==scores[i]
        if kind=="vectors": np.testing.assert_array_equal(associated,vectors[i])
    assert data==original
'''
(H/'tests/test_acceptance.py').write_text(accept)
verify=(OLD/'tests/verify.py').read_text().replace('libs/core/langchain_core/tools',source).replace('libs/core/tests/unit_tests',tests).replace('PACKAGE = "libs/core"',f'PACKAGE = "{package}"').replace('"/workspace/libs/core:/workspace/libs/standard-tests:/workspace/libs/partners/openai"','"/workspace/libs/partners/chroma:/workspace/libs/core:/workspace/libs/standard-tests:/workspace/libs/partners/openai"').replace('["libs/core", "libs/standard-tests", "libs/partners/openai"]','["libs/partners/chroma", "libs/core", "libs/standard-tests", "libs/partners/openai"]').replace('"tests/unit_tests/test_tools.py",\n                "tests/unit_tests/utils/test_function_calling.py",','"tests/unit_tests",')
# Exclude newly added tests from the separately recorded existing-test suite.
verify=verify.replace('"tests/unit_tests",\n                "-q",','*[str(WORK / TESTS / p) for p in manifest(BASE / TESTS) if Path(p).name.startswith("test_") and p.endswith(".py")],\n                "-q",')
(H/'tests/verify.py').write_text(verify)
subprocess.run([str(R/'.venv/bin/ruff'),'format',str(H/'tests')],check=True,capture_output=True)
B=S/'baseline'; C=S/'dependency-build'
image=subprocess.check_output(['docker','image','inspect','maintenance-task-dependencies:lc-chroma-none-001','--format','{{.Id}}'],text=True).strip()
if not (S/'receipt.key').exists(): (S/'receipt.key').write_bytes(secrets.token_bytes(32))
(S/'receipt.key').chmod(0o600)
for role in ['agent','verifier']:
 context=S/role; context.mkdir(exist_ok=True)
 if not (context/'source').exists(): shutil.copytree(B,context/'source')
 shutil.copyfile(T/'reproduce.py',context/'reproduce.py')
 lines=['FROM maintenance-task-dependencies:lc-chroma-none-001','USER root','COPY source /workspace','COPY reproduce.py /task/reproduce.py']
 if role=='agent':
  shutil.copyfile(T/'isolation.py',context/'isolation.py'); shutil.copyfile(R/'sandbox/workspace_edit.py',context/'workspace_edit.py')
  (context/'edit-resources.json').write_text(json.dumps({'source_paths':[source]}))
  unchanged=context/'unchanged'/source; unchanged.parent.mkdir(parents=True,exist_ok=True)
  if not unchanged.exists(): shutil.copytree(B/source,unchanged)
  lines+=['COPY isolation.py /task/isolation.py','COPY workspace_edit.py /task/workspace_edit.py','COPY edit-resources.json /task/edit-resources.json','COPY unchanged /task/unchanged']
 else:
  if not (context/'baseline').exists(): shutil.copytree(B,context/'baseline')
  shutil.copytree(H/'tests',context/'tests',dirs_exist_ok=True); shutil.copyfile(S/'receipt.key',context/'tests/receipt.key')
  lines+=['COPY baseline /baseline','COPY tests /tests']
 lines+=['RUN chmod -R a-w /workspace /task && mkdir -p /logs/agent /logs/verifier /logs/artifacts && chmod 777 /logs/agent /logs/verifier /logs/artifacts']
 if role=='agent': lines += [f'RUN find /workspace/{source} -type d -exec chmod 777 {{}} + && find /workspace/{source} -type f -exec chmod 666 {{}} + && chmod 1777 /workspace/{tests}','USER 65532:65532']
 else: lines+=['RUN chmod -R a-w /baseline /tests && chmod 400 /tests/receipt.key','USER root']
 (context/'Dockerfile').write_text('\n'.join(lines)+'\n')
agent_image=build(S/'agent','maintenance-chroma-agent:v1'); verifier_image=build(S/'verifier','maintenance-chroma-verifier:v1')
config=toml.loads((OLD/'task.toml').read_text()); config['metadata'].update(task_id=ID,starting_commit=COMMIT,baseline_checksum=digest(manifest(B)),experimental_setting=PREFIX,difficulty='Level 2')
config['environment'].update(docker_image=agent_image,workdir='/workspace/'+package)
config['verifier']['environment']['docker_image']=verifier_image
config['verifier']['collect'][0]['command']=f"rm -f /logs/artifacts/candidate.tar && tar --mode='u+rwX,go+rX' -cf /logs/artifacts/candidate.tar -C /workspace {source} {tests}"
(H/'task.toml').write_text(toml.dumps(config))
# Reference contains only the focused implementation from the original historical fix.
reference=subprocess.check_output(['git','-C',str(R / ".local/upstream/langchain"),'show','03aa9d65fd861fff4c709f8ca56b52a9ede9c74a:'+source+'/vectorstores.py'],text=True)
start=reference.index('def safe_results_to_docs_and_scores'); end=reference.index('Matrix =',start); reference=reference[:start]+reference[end:]
regression=accept.replace('Independent offline acceptance checks; never provided to the coding agent.','Regression checks for absent document content.')
normal=S/'normalization'; normal.mkdir(exist_ok=True)
(normal/'vectorstores.py').write_text(reference); (normal/'test_missing_document_content.py').write_text(regression)
for name in ['vectorstores.py','test_missing_document_content.py']:
 subprocess.run(['docker','run','--rm','--network=none','--user=0:0','-v',str(B)+':/workspace:ro','-v',str(normal)+':/reference',image,'ruff','check','--fix','--no-cache','--config','/workspace/'+package+'/pyproject.toml','/reference/'+name],capture_output=True)
 subprocess.run(['docker','run','--rm','--network=none','--user=0:0','-v',str(B)+':/workspace:ro','-v',str(normal)+':/reference',image,'ruff','format','--no-cache','--config','/workspace/'+package+'/pyproject.toml','/reference/'+name],check=True,capture_output=True)
(S/'reference-fixture.json').write_text(json.dumps({source+'/vectorstores.py':(normal/'vectorstores.py').read_text(),tests+'/test_missing_document_content.py':(normal/'test_missing_document_content.py').read_text()}))
job=json.loads((R/'configs/harbor-workspace-deliverable-v2-model.json').read_text()); policy=job['agents'][0]['kwargs']['policy']
policy.update(source_provenance={'starting_commit':COMMIT,'baseline_checksum':digest(manifest(B))},receipt_key=str(S/'receipt.key'),reference_fixture=str(S/'reference-fixture.json'),reproduction={'expected_failure':'ValidationError'})
for mode in ['baseline','reference','reference_budget','reference_error','model']:
 new=copy.deepcopy(job); new['job_name']=PREFIX+'-'+mode; new['tasks']=[{'path':str(H)}]
 if mode!='model': new['agents'][0]['kwargs']['fixture']=mode; new['jobs_dir']=str(R/'reports/harbor-validation')
 (R/f'configs/{PREFIX}-{mode}.json').write_text(json.dumps(new,indent=2)+'\n')
state=json.loads((S/'preparation-state.json').read_text()); state.update(agent_image=agent_image,verifier_image=verifier_image,dependency_image=image)
(S/'preparation-state.json').write_text(json.dumps(state,indent=2)); print(json.dumps(state,indent=2))
