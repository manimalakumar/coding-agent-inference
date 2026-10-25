import json,hashlib,shutil,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
from sandbox.docker import digest,manifest
from experiments.launch import verify_freeze
R=Path.cwd();L='context-counter-feature-v1';ID='LC-CONTEXT-COUNTER-001';S=R / ".local/resources"/ID/'feature-v1';prep=json.loads((R/f'reports/{L}-preparation.json').read_text());validation=json.loads((R/f'reports/{L}-validation-summary.json').read_text());assert validation['status']=='passed'
assert len(validation['controls'])==4 and all(v['status']=='passed' for v in validation['controls'].values())
for name,path in [('environment.json','/opt/environment.json'),('requirements.lock','/opt/requirements.lock')]:
 data=subprocess.check_output(['docker','run','--rm','--network=none',prep['dependency_image'],'cat',path]);(S/name).write_bytes(data);(R/'tasks'/ID/name).write_bytes(data)
job=json.loads((R/f'configs/{L}-model.json').read_text());policy=job['agents'][0]['kwargs']['policy']
recipe=R/'reports/task-preparation'/ID;recipe.mkdir(parents=True,exist_ok=True)
for name in ['prepare_context_task.py','build_context_bundle.py','validate_context_controls.py','freeze_context_setting.py']:shutil.copyfile('/tmp/'+name,recipe/name)
# Keep references and all solution-bearing setup recipes strictly host-side. Neither is mounted into the agent image.
assert digest(manifest(S/'baseline'))==prep['baseline_checksum']
old=json.loads((R/'configs/comparison-4x2x3-v1.frozen-setting.json').read_text())
paths=set(policy['code_fingerprints'])|{'configs/completion-workspace-v1.json','configs/mini-swe-workspace-v1.yaml','experiments/launch.py','experiments/bookkeeping.py','experiments/records.py','experiments/report.py','experiments/storage.py','agent/mini_runtime.py','sandbox/docker.py','evaluator/workspace_policy.py'}
paths|={str(p.relative_to(R)) for root in [R/'harbor_tasks/settings'/L,R/'tasks'/ID,recipe] for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
paths|={f'configs/{L}-{m}.json' for m in ['model','baseline','reference','reference_budget','reference_error']}
files={p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in sorted(paths)}
record={'schema_version':1,'experimental_setting':L,'task_id':ID,'task_type':'feature_request','issue_url':'https://github.com/langchain-ai/langchain/issues/39757','frozen_at':datetime.now(timezone.utc).isoformat(),'starting_commit':prep['starting_commit'],'baseline_checksum':prep['baseline_checksum'],'agent_image':prep['agent_image'],'verifier_image':prep['verifier_image'],'dependency_image':prep['dependency_image'],'task_python':'3.14.6','packages':old['packages'],'policy':policy,'source_and_config_sha256':files,'validation_report':f'reports/{L}-validation-summary.json','task_versions':{'langchain':'1.3.15','langchain-core':'1.5.6'},'setup_model_requests':0,'comparison_scope':'Separate feature-task setting. Frozen controller, tools, model, provider, budgets and completion policy retained; new task snapshot, dependencies, instructions and checks.'}
record['setting_id']=digest(record);setting=f'configs/{L}.frozen-setting.json';man=f'configs/{L}.sha256';assert not (R/setting).exists();(R/setting).write_text(json.dumps(record,indent=2)+'\n');(R/man).write_text(''.join(f'{sha}  {p}\n' for p,sha in sorted(files.items()))+f'{hashlib.sha256((R/setting).read_bytes()).hexdigest()}  {setting}\n')
catalog=json.loads((R/'configs/study.json').read_text());catalog['studies'][L]={'task_ids':[ID],'job_config':f'configs/{L}-model.json','freeze_manifest':man,'frozen_setting':setting};(R/'configs/study.json').write_text(json.dumps(catalog,indent=2)+'\n');assert verify_freeze(R,man,setting)==record['setting_id']
state=json.loads((S/'preparation-state.json').read_text());setup={'task_id':ID,'setup_status':'validated','task':json.loads((R/'tasks'/ID/'task.json').read_text()),'commit':prep['starting_commit'],'baseline_checksum':prep['baseline_checksum'],'image_id':prep['dependency_image'],'setup_model_calls':0,'setup_inference_cost_usd':0,'setup_seconds':time.time()-state['setup_started_at'],'setup_timing_scope':'Trusted setup wall time from completed snapshot export through environment, controls and freeze; earlier issue research/export precedes this timer','source_provenance':str(S/'provenance.json'),'environment_manifest':str(S/'environment.json'),'frozen_setting_id':record['setting_id']};(S/'setup.json').write_text(json.dumps(setup,indent=2)+'\n');(R/f'reports/{L}-setup.json').write_text(json.dumps(setup,indent=2)+'\n')
print(json.dumps({'setting_id':record['setting_id'],'study':L,'model':policy['model'],'setup_seconds':setup['setup_seconds']},indent=2))
