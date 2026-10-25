import subprocess,json,sys,os
from pathlib import Path
R=Path.cwd(); prefix='harbor-chroma-none-workspace-v1'; results={}
for mode in sys.argv[1:]:
 job='chroma-none-workspace-v1-'+mode+'-control-002'
 env=os.environ|{'MSWEA_SILENT_STARTUP':'1','MSWEA_GLOBAL_CONFIG_DIR':str(R/'.state/mini')}
 p=subprocess.run([str(R/'.venv/bin/harbor'),'run','-c',str(R/f'configs/{prefix}-{mode}.json'),'--job-name',job],env=env,capture_output=True,text=True,timeout=450)
 (R/f'reports/{job}.log').write_text(p.stdout+p.stderr)
 trials=list((R/'reports/harbor-validation'/job).glob('LC-*'))
 result={'command_exit':p.returncode}
 for trial in trials:
  f=trial/'verifier/checks.json'
  if f.exists():
   d=json.loads(f.read_text()); result.update(checks={k:v['exit_code'] for k,v in d['checks'].items()},error=d.get('error'),task_success=d.get('task_success'),status=d.get('status'),trial=str(trial.relative_to(R)))
   print(mode,json.dumps(result),flush=True)
  else: print(mode,'MISSING CHECKS',p.stdout[-1000:],p.stderr[-1000:],flush=True)
 results[mode]=result
 (R/f'reports/{prefix}-controls-progress.json').write_text(json.dumps(results,indent=2))
