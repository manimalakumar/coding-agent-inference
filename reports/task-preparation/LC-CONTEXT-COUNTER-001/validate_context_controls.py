import json,os,subprocess,time
from pathlib import Path
R=Path.cwd();L='context-counter-feature-v1';ID='LC-CONTEXT-COUNTER-001';started=time.monotonic();result={}
for mode in ['baseline','reference','reference_budget','reference_error']:
 job=L+'-'+mode+'-control-002';config=R/f'configs/{L}-{mode}.json';env=os.environ|{'MSWEA_SILENT_STARTUP':'1','MSWEA_GLOBAL_CONFIG_DIR':str(R/'.state/mini')}
 p=subprocess.run([str(R/'.venv/bin/harbor'),'run','-c',str(config),'--job-name',job],env=env,capture_output=True,text=True,timeout=450);(R/f'reports/{job}.log').write_text(p.stdout+p.stderr)
 folder=next((R/'reports/harbor-validation'/job).glob('LC-*'));checksfile=folder/'verifier/checks.json'
 if not checksfile.exists():
  result[mode]={'status':'setup_failure','trial':str(folder.relative_to(R))};print(mode,'MISSING CHECKS',flush=True);break
 d=json.loads(checksfile.read_text());reward=json.loads((folder/'verifier/reward.json').read_text());iso=json.loads((folder/'agent/isolation.json').read_text());receipt=d.get('controller_receipt',{})
 ok=reward.get('task_success')==int(mode!='baseline') and reward.get('explicit_submission')==int(mode in ['baseline','reference']) and all(iso.values()) and receipt.get('model_requests')==0
 result[mode]={'status':'passed' if ok else 'failed','trial':str(folder.relative_to(R)),'rewards':reward,'check_exit_codes':{k:v['exit_code'] for k,v in d['checks'].items()},'error':d.get('error'),'isolation':iso,'model_requests':receipt.get('model_requests'),'termination_reason':receipt.get('termination_reason'),'candidate_sha256':d.get('candidate_sha256'),'source_snapshot':json.loads((folder/'agent/source-snapshot.json').read_text())}
 print(mode,result[mode]['status'],result[mode]['check_exit_codes'],flush=True)
 (R/f'reports/{L}-validation-summary.json').write_text(json.dumps({'setting':L,'status':'passed' if len(result)==4 and all(x['status']=='passed' for x in result.values()) else 'pending_or_failed','controls':result,'model_requests':0,'elapsed_seconds':time.monotonic()-started},indent=2)+'\n')
 if not ok:break
