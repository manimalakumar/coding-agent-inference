import json,copy,shutil,secrets,subprocess
from pathlib import Path
import toml
from harbor.models.task.config import TaskConfig
from harbor.models.job.config import JobConfig
from harbor_integration.prepare_resources import build
from sandbox.docker import digest,manifest
R=Path.cwd();ID='LC-CONTEXT-COUNTER-001';L='context-counter-feature-v1';S=R / ".local/resources"/ID/'feature-v1';B=S/'baseline';T=R/'tasks'/ID;H=R/'harbor_tasks/settings'/L/ID;O=R/'harbor_tasks/settings/comparison-4x2x3-v1/LC-CHROMA-NONE-001';P='libs/langchain_v1';SRC=P+'/langchain/agents/middleware';TEST=P+'/tests/unit_tests'
for d in ['tests','environment']:(H/d).mkdir(parents=True,exist_ok=True)
shutil.copyfile(T/'instruction.md',H/'instruction.md')
for p in ['test.sh','docker-compose.yaml','workspace_policy.py','completion-policy.json']:shutil.copyfile(O/'tests'/p,H/'tests'/p)
shutil.copyfile('/tmp/context_acceptance.py',H/'tests/test_acceptance.py')
v=(O/'tests/verify.py').read_text().replace('libs/partners/chroma/langchain_chroma',SRC).replace('libs/partners/chroma/tests/unit_tests',TEST).replace('libs/partners/chroma',P)
v=v.replace('*[\n                    str(WORK / TESTS / p)\n                    for p in manifest(BASE / TESTS)\n                    if Path(p).name.startswith("test_") and p.endswith(".py")\n                ],','"tests/unit_tests/agents/middleware/implementations/test_context_editing.py",')
(H/'tests/verify.py').write_text(v)
# Public probe exercises counting through the wrapper, not merely accepting a keyword.
(T/'reproduce.py').write_text('''"""Offline feature probe using a custom counter."""
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
''')
key=S/'receipt.key'
if not key.exists():key.write_bytes(secrets.token_bytes(32));key.chmod(0o600)
base=subprocess.check_output(['docker','image','inspect','maintenance-context-deps:v1','--format','{{.Id}}'],text=True).strip()
images={}
for role in ['agent','verifier']:
 C=S/role;C.mkdir(exist_ok=True)
 if not (C/'source').exists():shutil.copytree(B,C/'source')
 shutil.copyfile(T/'reproduce.py',C/'reproduce.py')
 lines=['FROM maintenance-context-deps:v1','USER root','COPY source /workspace','COPY reproduce.py /task/reproduce.py']
 if role=='agent':
  shutil.copyfile(T/'isolation.py',C/'isolation.py');shutil.copyfile(R/'sandbox/workspace_edit.py',C/'workspace_edit.py');(C/'edit-resources.json').write_text(json.dumps({'source_paths':[SRC]}))
  unchanged=C/'unchanged'/SRC;unchanged.parent.mkdir(parents=True,exist_ok=True)
  if not unchanged.exists():shutil.copytree(B/SRC,unchanged)
  lines += ['COPY isolation.py /task/isolation.py','COPY workspace_edit.py /task/workspace_edit.py','COPY edit-resources.json /task/edit-resources.json','COPY unchanged /task/unchanged']
 else:
  if not (C/'baseline').exists():shutil.copytree(B,C/'baseline')
  shutil.copytree(H/'tests',C/'tests',dirs_exist_ok=True);shutil.copyfile(key,C/'tests/receipt.key');lines+=['COPY baseline /baseline','COPY tests /tests','RUN chmod -R a-w /baseline /tests && chmod 400 /tests/receipt.key']
 lines += ['RUN chmod -R a-w /workspace /task && mkdir -p /logs/agent /logs/verifier /logs/artifacts && chmod 777 /logs/agent /logs/verifier /logs/artifacts']
 if role=='agent':lines += [f'RUN find /workspace/{SRC} -type d -exec chmod 777 {{}} + && find /workspace/{SRC} -type f -exec chmod 666 {{}} + && chmod 1777 /workspace/{TEST}']
 lines += [f'WORKDIR /workspace/{P}','USER '+('65532:65532' if role=='agent' else 'root')]
 (C/'Dockerfile').write_text('\n'.join(lines)+'\n');images[role]=build(C,'maintenance-context-'+role+':v1')
cfg=toml.loads((O/'task.toml').read_text());cfg['metadata'].update(task_id=ID,starting_commit=json.loads((T/'task.json').read_text())['commit'],baseline_checksum=digest(manifest(B)),experimental_setting=L,task_type='feature_request');cfg['metadata'].pop('difficulty',None)
cfg['environment'].update(docker_image=images['agent'],workdir='/workspace/'+P);cfg['verifier']['environment']['docker_image']=images['verifier'];cfg['verifier']['collect'][0]['command']=cfg['verifier']['collect'][0]['command'].replace('libs/partners/chroma/langchain_chroma',SRC).replace('libs/partners/chroma/tests/unit_tests',TEST);TaskConfig.model_validate(cfg);(H/'task.toml').write_text(toml.dumps(cfg))
compose=(O/'environment/docker-compose.yaml').read_text().replace('libs/partners/chroma/langchain_chroma',SRC).replace('libs/partners/chroma/tests/unit_tests',TEST);(H/'environment/docker-compose.yaml').write_text(compose)
# Trusted reference uses the upstream production implementation only, without exposing PR metadata or tests.
content=subprocess.check_output(['git','-C',str(R / ".local/upstream/langchain"),'show','37f266278d780cd7ebdfbf1891ba92066192d687:'+SRC+'/context_editing.py'],text=True)
# Add a separate regression file; existing tests remain untouched.
reg='''"""Regression for the optional custom counter."""
from langchain.agents.middleware.context_editing import ContextEditingMiddleware


def test_custom_token_counter() -> None:
    """A configured counter can be supplied without replacing defaults globally."""
    middleware = ContextEditingMiddleware(token_counter=lambda _messages: 0)
    assert middleware is not None
'''
reg=subprocess.check_output([str(R/'.venv/bin/ruff'),'format','--config',str(B/P/'pyproject.toml'),'--stdin-filename',TEST+'/test_custom_token_counter_reference.py','-'],input=reg,text=True)
fixture={SRC+'/context_editing.py':content,TEST+'/test_custom_token_counter_reference.py':reg};(S/'reference-fixture.json').write_text(json.dumps(fixture,indent=2))
job=json.loads((R/'configs/comparison-4x2x3-v1-LC-CHROMA-NONE-001-minimax-model.json').read_text());policy=job['agents'][0]['kwargs']['policy'];policy['receipt_key']=str(key);policy['reference_fixture']=str(S/'reference-fixture.json');policy['source_provenance']={'starting_commit':cfg['metadata']['starting_commit'],'baseline_checksum':cfg['metadata']['baseline_checksum']};policy['reproduction']={'expected_failure':"unexpected keyword argument 'token_counter'"}
for mode in ['model','baseline','reference','reference_budget','reference_error']:
 j=copy.deepcopy(job);j['job_name']=L+'-'+mode;j['tasks']=[{'path':str(H)}]
 if mode!='model':j['agents'][0]['kwargs']['fixture']=mode;j['jobs_dir']=str(R/'reports/harbor-validation')
 JobConfig.model_validate(j);(R/f'configs/{L}-{mode}.json').write_text(json.dumps(j,indent=2)+'\n')
record={'task_id':ID,'setting':L,'starting_commit':cfg['metadata']['starting_commit'],'baseline_checksum':cfg['metadata']['baseline_checksum'],'agent_image':images['agent'],'verifier_image':images['verifier'],'dependency_image':base,'state_directory':str(S),'task_directory':str(H),'setup_model_requests':0}
(R/f'reports/{L}-preparation.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
