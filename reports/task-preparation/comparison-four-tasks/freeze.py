import hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
from sandbox.docker import digest
from experiments.launch import verify_freeze
R=Path.cwd();L='comparison-4x2x3-v1';setting_path=f'configs/{L}.frozen-setting.json';manifest_path=f'configs/{L}.sha256';plan_path=f'configs/{L}.plan.json'
assert not (R/setting_path).exists()
validation=json.loads((R/f'reports/{L}-validation-summary.json').read_text());assert validation['status']=='passed' and len(validation['controls'])==16
preparation=json.loads((R/f'reports/{L}-preparation.json').read_text());bundles=preparation['task_bundles'];assert len(bundles)==4
compat_path='reports/compatibility/glm-5-z-ai-fp8-v1/result.json';compat=json.loads((R/compat_path).read_text());assert compat['status']=='passed' and compat['model_requests']==2 and compat['reasoning_tokens_reported']==[0,0]
common=json.loads((R/'configs/completion-workspace-v1.json').read_text());models={'minimax':json.loads((R/'configs/minimax-m2.5-mini-v5.json').read_text()),'glm5':json.loads((R/'configs/glm-5-comparison-v1.json').read_text())}
assert hashlib.sha256((R/'configs/glm-5-comparison-v1.json').read_bytes()).hexdigest()==compat['model_config_sha256']
catalog=json.loads((R/'configs/study.json').read_text())
for entry in catalog['studies'].values():verify_freeze(R,entry['freeze_manifest'],entry['frozen_setting'])
archive=R/'reports/settings/pre-comparison-4x2x3-v1';archive.mkdir(parents=True,exist_ok=True)
for relative in ['main.ipynb','configs/study.json','reports/NEXT-MODEL-INVOKE.md']:
 destination=archive/Path(relative).name
 if not destination.exists():shutil.copyfile(R/relative,destination)
schedule=[];study_ids={}
for b in bundles:
 ID=b['task_id'];assert b['hidden_acceptance_sha256']==b['original_hidden_acceptance_sha256']
 policies=[]
 for model in models:
  path=f'configs/{L}-{ID}-{model}-model.json';job=json.loads((R/path).read_text());policy=job['agents'][0]['kwargs']['policy'];assert policy['model']==models[model] and policy['completion_policy']==common
  assert policy['budgets']==common['agent_budgets']
  assert job['agents'][0]['import_path']=='harbor_integration.host_comparison:ComparisonAgent'
  stripped=policy.copy();stripped.pop('model');policies.append(stripped)
  study=f'comparison-v1-{ID.lower()}-{model}';study_ids[ID,model]=study
  catalog['studies'][study]={'task_ids':[ID],'job_config':path,'freeze_manifest':manifest_path,'frozen_setting':setting_path,'comparison_round':L}
 assert policies[0]==policies[1]
for repeat in range(1,4):
 for ti,b in enumerate(bundles):
  for model in (['minimax','glm5'] if (repeat+ti)%2 else ['glm5','minimax']):
   ID=b['task_id'];job=f'comparison-v1-{ID.lower()}-{model}-attempt-{repeat:03d}';study=study_ids[ID,model]
   schedule.append({'order':len(schedule)+1,'task_id':ID,'model_key':model,'model_id':models[model]['model'],'repeat':repeat,'job_name':job,'study_id':study,'command':['.venv/bin/python','-m','experiments','execute','--study',study,'--confirm-paid-run','--job-name',job]})
assert len(schedule)==24 and set(Counter((s['task_id'],s['model_key']) for s in schedule).values())=={3}
plan={'schema_version':1,'round_id':L,'status_at_freeze':'ready_not_started','planned_measured_runs':24,'task_count':4,'model_count':2,'fresh_attempts_per_cell':3,'models':models,'agent_budgets':common['agent_budgets'],'evaluation_seconds':300,'completion_policy':'workspace-deliverable-v1','runtime':'harbor_integration.host_comparison:ComparisonAgent','runtime_revision':'comparison-civil-deadline-v1','task_bundles':bundles,'schedule':schedule,'compatibility_result':compat_path,'pilot_reuse':False,'fresh_context_and_snapshot_every_run':True,'automatic_retries':0,'infrastructure_policy':{'retain_every_attempt_and_cost':True,'primary_end_to_end_economics_includes_all_scheduled_attempt_inference':True,'coding_only_exclusions_require_explicit_reason_and_separate_denominator':True,'excluded_spend_retained_as_operational_expense':True,'no_silent_replacement_or_rerun':True,'additional_attempts_require_separate_ids_and_authorization':True},'freeze_manifest':manifest_path,'frozen_setting':setting_path}
(R/plan_path).write_text(json.dumps(plan,indent=2)+'\n')
setup={'schema_version':1,'phase':'pre-measurement compatibility/setup','entries':[{'model':models['glm5']['model'],'provider':models['glm5']['provider'],'model_requests':compat['model_requests'],'cost_usd':compat['reported_cost_usd'],'result':compat_path,'included_in_measured_round':False}],'total_reported_cost_usd':compat['reported_cost_usd'],'resource_preparation_model_requests':0,'validation_controls_model_requests':0,'measured_round_model_requests':0}
(R/'reports/setup-inference-costs.json').write_text(json.dumps(setup,indent=2)+'\n')
recipe=R/'reports/task-preparation'/L;recipe.mkdir(parents=True,exist_ok=True);shutil.copyfile('/tmp/validate_comparison_controls.py',recipe/'validate_controls.py');shutil.copyfile('/tmp/freeze_comparison_v1.py',recipe/'freeze.py')
old=json.loads((R/'configs/LC-CHROMA-NONE-001.workspace-v1.frozen-setting.json').read_text());files={p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in old['source_and_config_sha256']}
extra=[*list((R/'harbor_tasks/settings'/L).rglob('*')),*list((R/'configs').glob(L+'-*.json')),R/plan_path,R/'configs/glm-5-comparison-v1.json',R/'harbor_integration/host_comparison.py',R/'harbor_integration/comparison_setting.py',R/'experiments/compatibility.py',R/'tests/test_comparison_runtime.py',*list(recipe.rglob('*.py')),*list((R/'reports/compatibility/glm-5-z-ai-fp8-v1').rglob('*')),R/f'reports/{L}-preparation.json',R/f'reports/{L}-validation-summary.json',R/'reports/setup-inference-costs.json']
for path in extra:
 if path.is_file():files[str(path.relative_to(R))]=hashlib.sha256(path.read_bytes()).hexdigest()
setting={'schema_version':1,'experimental_setting':L,'frozen_at':datetime.now(timezone.utc).isoformat(),'packages':old['packages'],'agent_image':bundles[0]['agent_image'],'verifier_image':bundles[0]['verifier_image'],'task_python':'Task-specific: OpenAI tasks 3.14.6; Core and Chroma 3.13.16','task_bundles':bundles,'models':models,'policy':{'budgets':common['agent_budgets'],'completion_policy':common,'runtime_revision':plan['runtime_revision']},'planned_measured_runs':24,'schedule':schedule,'source_and_config_sha256':files,'validation_report':f'reports/{L}-validation-summary.json','compatibility_result':compat_path,'setup_inference_report':'reports/setup-inference-costs.json','pilot_reuse':False,'source_snapshot_and_acceptance_checks_preserved':True,'infrastructure_policy':plan['infrastructure_policy']}
setting['setting_id']=digest(setting);(R/setting_path).write_text(json.dumps(setting,indent=2)+'\n');(R/manifest_path).write_text(''.join(f'{sha}  {p}\n' for p,sha in sorted(files.items()))+f'{hashlib.sha256((R/setting_path).read_bytes()).hexdigest()}  {setting_path}\n')
# Keep active historical view unchanged; the new round is an explicit selection.
(R/'configs/study.json').write_text(json.dumps(catalog,indent=2)+'\n');assert verify_freeze(R,manifest_path,setting_path)==setting['setting_id'];print(json.dumps({'setting_id':setting['setting_id'],'planned_runs':24,'zero_inference_controls':16,'compatibility_calls':2,'compatibility_cost':compat['reported_cost_usd'],'measured_runs_started':0},indent=2))
