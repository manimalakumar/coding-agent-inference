import json,hashlib,copy,shutil,time
from pathlib import Path
from datetime import datetime,UTC
from sandbox.docker import digest,manifest
from experiments.launch import verify_freeze
R=Path.cwd(); ID='LC-CHROMA-NONE-001'; PREFIX='harbor-chroma-none-workspace-v1'; S=R / ".local/resources"/ID/'workspace-v1'
state=json.loads((S/'preparation-state.json').read_text()); controls={}
for mode in ['baseline','reference','reference_budget','reference_error']:
 job='chroma-none-workspace-v1-'+mode+'-control-002'
 trial=next((R/'reports/harbor-validation'/job).glob('LC-*'))
 checks=json.loads((trial/'verifier/checks.json').read_text()); rewards=json.loads((trial/'verifier/reward.json').read_text()); receipt=checks['controller_receipt']
 assert not receipt['model_requests']
 assert rewards['task_success']==int(mode!='baseline'), (mode,checks)
 assert rewards['explicit_submission']==int(mode in ['baseline','reference'])
 isolation=json.loads((trial/'agent/isolation.json').read_text()); assert all(isolation.values())
 controls[mode]={'trial':str(trial.relative_to(R)),'rewards':rewards,'termination_reason':checks['termination_reason'],'checks':{k:v['exit_code'] for k,v in checks['checks'].items()},'candidate_sha256':checks['candidate_sha256'],'isolation':isolation,'model_requests':receipt['model_requests']}
validation={'task_id':ID,'setting':PREFIX,'model_requests':0,'controls':controls,'project_tests':'98 passed','validated_at':datetime.now(UTC).isoformat()}
(R/f'reports/{PREFIX}-validation-summary.json').write_text(json.dumps(validation,indent=2)+'\n')
setup_checks=json.loads((S/'environment-validation.json').read_text()); assert setup_checks['existing_tests']['exit_code']==0
(S/'environment.json').write_text(setup_checks['environment']['output']); (S/'requirements.lock').write_text(setup_checks['dependencies']['output'])
state.update(setup_status='validated',setup_model_calls=0,setup_elapsed_seconds=time.time()-state['setup_started_at'],difficulty='Level 2',reference_commit='03aa9d65fd861fff4c709f8ca56b52a9ede9c74a',reference_scope='Only Chroma implementation; remove unused untyped helper and normalize formatting; regression only in a new file.',validation_report=f'reports/{PREFIX}-validation-summary.json',dependency_manifest_sha256=hashlib.sha256((S/'requirements.lock').read_bytes()).hexdigest(),environment_sha256=hashlib.sha256((S/'environment.json').read_bytes()).hexdigest(),source_provenance_sha256=hashlib.sha256((S/'provenance.json').read_bytes()).hexdigest())
assert digest(manifest(S/'baseline'))==state['baseline_checksum']
(R/f'reports/{PREFIX}-preparation.json').write_text(json.dumps(state,indent=2)+'\n');(S/'setup.json').write_text(json.dumps(state,indent=2)+'\n')
# Archive trusted preparation scripts; never mounted into evaluated containers.
archive=R/'reports/task-preparation/LC-CHROMA-NONE-001'; archive.mkdir(parents=True,exist_ok=True)
for name in ['prepare_chroma_task','finalize_chroma_task','validate_chroma_environment','chroma_controls','freeze_chroma_task']:
 shutil.copyfile(Path('/tmp')/(name+'.py'),archive/(name+'.py'))
job=json.loads((R/f'configs/{PREFIX}-model.json').read_text()); policy=job['agents'][0]['kwargs']['policy']
old=json.loads((R/'configs/LC-ANNOTATED-TOOLS-001.workspace-deliverable-v2.frozen-setting.json').read_text())
for field in ['budgets','model','mini_config','completion_policy']:
 assert policy[field]==old['policy'][field], 'Unexpected workflow drift: '+field
files={p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in old['source_and_config_sha256']}
extra=[*list((R/'tasks'/ID).rglob('*')),*list((R/'harbor_tasks/settings'/PREFIX).rglob('*')),*list(archive.rglob('*.py')),R/f'reports/{PREFIX}-validation-summary.json',R/f'reports/{PREFIX}-preparation.json',*list((R/'configs').glob(PREFIX+'-*.json'))]
for path in extra:
 if path.is_file(): files[str(path.relative_to(R))]=hashlib.sha256(path.read_bytes()).hexdigest()
setting=copy.deepcopy(old)
setting.update(task_id=ID,difficulty='Level 2',experimental_setting=PREFIX,configuration_version=1,frozen_at=datetime.now(UTC).isoformat(),starting_commit=state['starting_commit'],baseline_checksum=state['baseline_checksum'],agent_image=state['agent_image'],verifier_image=state['verifier_image'],policy=policy,source_and_config_sha256=files,validation_report=f'reports/{PREFIX}-validation-summary.json',parent_setting_id=old['setting_id'],task_runtime_versions={'langchain-chroma':'0.2.4','langchain-core':state['core_version'],'pydantic':'2.12.5','chromadb':'1.0.9'},interpretation='First Level 2 Chroma pilot under the same workspace delivery policy and MiniMax model/provider/budgets; task-specific snapshot, dependencies and checks form a distinct experimental setting.',reference_pr='https://github.com/langchain-ai/langchain/pull/31377')
setting.pop('setting_id',None); setting['setting_id']=digest(setting)
filename=f'configs/{ID}.workspace-v1.frozen-setting.json'; (R/filename).write_text(json.dumps(setting,indent=2)+'\n')
(R/f'configs/{PREFIX}.sha256').write_text(''.join(f'{sha}  {p}\n' for p,sha in sorted(files.items()))+f'{hashlib.sha256((R/filename).read_bytes()).hexdigest()}  {filename}\n')
catalog=json.loads((R/'configs/study.json').read_text()); catalog['active_study']='chroma-none-workspace-v1';catalog['studies']['chroma-none-workspace-v1']={'task_ids':[ID],'job_config':f'configs/{PREFIX}-model.json','freeze_manifest':f'configs/{PREFIX}.sha256','frozen_setting':filename};(R/'configs/study.json').write_text(json.dumps(catalog,indent=2)+'\n')
assert verify_freeze(R,f'configs/{PREFIX}.sha256',filename)==setting['setting_id']
print(json.dumps({'setting_id':setting['setting_id'],'setup_seconds':state['setup_elapsed_seconds'],'controls':{k:v['rewards'] for k,v in controls.items()}},indent=2))
