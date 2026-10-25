import json,os,subprocess,time,concurrent.futures
from pathlib import Path
R=Path.cwd();LABEL='comparison-4x2x3-v1'; prep=json.loads((R/f'reports/{LABEL}-preparation.json').read_text());started=time.time()
def validate_task(bundle):
 ID=bundle['task_id'];result={}
 for mode in ['baseline','reference','reference_budget','reference_error']:
  job=f'comparison-v1-{ID.lower()}-{mode}-control-001';config=R/f'configs/{LABEL}-{ID}-minimax-{mode}.json';env=os.environ|{'MSWEA_SILENT_STARTUP':'1','MSWEA_GLOBAL_CONFIG_DIR':str(R/'.state/mini')}
  p=subprocess.run([str(R/'.venv/bin/harbor'),'run','-c',str(config),'--job-name',job],env=env,capture_output=True,text=True,timeout=450);(R/f'reports/{job}.log').write_text(p.stdout+p.stderr)
  folder=next((R/'reports/harbor-validation'/job).glob('LC-*'));checksfile=folder/'verifier/checks.json'
  if not checksfile.exists():
   result[mode]={'status':'setup_failure','trial':str(folder.relative_to(R)),'native_result':json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else None};print(ID,mode,'MISSING CHECKS',flush=True);break
  d=json.loads(checksfile.read_text());reward=json.loads((folder/'verifier/reward.json').read_text());iso=json.loads((folder/'agent/isolation.json').read_text());receipt=d.get('controller_receipt',{})
  ok=reward.get('task_success')==int(mode!='baseline') and reward.get('explicit_submission')==int(mode in ['baseline','reference']) and all(iso.values()) and receipt.get('model_requests')==0
  result[mode]={'status':'passed' if ok else 'failed','trial':str(folder.relative_to(R)),'rewards':reward,'check_exit_codes':{k:v['exit_code'] for k,v in d['checks'].items()},'error':d.get('error'),'isolation':iso,'model_requests':receipt.get('model_requests'),'termination_reason':receipt.get('termination_reason'),'candidate_sha256':d.get('candidate_sha256'),'source_snapshot':json.loads((folder/'agent/source-snapshot.json').read_text()),'clock_policy':receipt.get('clock_policy')}
  print(ID,mode,result[mode]['status'],result[mode]['check_exit_codes'],flush=True)
  (R/f'reports/{LABEL}-{ID}-controls.json').write_text(json.dumps(result,indent=2)+'\n')
  if not ok:break
 return ID,result
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 results=dict(pool.map(validate_task,prep['task_bundles']))
summary={'setting':LABEL,'model_requests':0,'measured_round_started':False,'controls':{ID+':'+mode:v for ID,modes in results.items() for mode,v in modes.items()},'elapsed_seconds':time.time()-started,'status':'passed' if all(len(v)==4 and all(x['status']=='passed' for x in v.values()) for v in results.values()) else 'failed'}
(R/f'reports/{LABEL}-validation-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('FINAL',summary['status'],flush=True)
