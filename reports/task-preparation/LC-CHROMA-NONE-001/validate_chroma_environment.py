import json,subprocess,shlex,hashlib,time
from pathlib import Path
R=Path.cwd(); S=R / ".local/resources/LC-CHROMA-NONE-001/workspace-v1"; B=S/'baseline'; T=R/'tasks/LC-CHROMA-NONE-001'
image='maintenance-task-dependencies:lc-chroma-none-001'; checks={}
commands={'imports':"python -c \"import langchain_chroma,langchain_core,pydantic,sys;print(langchain_chroma.__file__,langchain_core.__file__,pydantic.__version__,sys.version);assert langchain_chroma.__file__.startswith('/workspace/');assert langchain_core.__file__.startswith('/workspace/');assert pydantic.__version__=='2.12.5'\"",'reproduction':'python /task/reproduce.py','existing_tests':'python -m pytest tests/unit_tests -q --disable-socket --allow-unix-socket','environment':'cat /opt/environment.json','dependencies':'cat /opt/requirements.lock'}
for name,command in commands.items():
 args=['docker','run','--rm','--network=none','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges','--user=65532:65532','--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=512m,mode=1777','-v',str(B)+':/workspace:ro','-v',str(T/'reproduce.py')+':/task/reproduce.py:ro',image,'bash','-o','pipefail','-c',command]
 p=subprocess.run(args,capture_output=True,text=True,timeout=120)
 (S/(name+'.log')).write_text(p.stdout+p.stderr); checks[name]={'exit_code':p.returncode,'output':p.stdout+p.stderr}
 print(name,p.returncode,(p.stdout+p.stderr)[-1800:])
(S/'environment-validation.json').write_text(json.dumps(checks,indent=2))
assert checks['imports']['exit_code']==0
assert checks['reproduction']['exit_code']==1 and 'ValidationError' in checks['reproduction']['output']
assert checks['existing_tests']['exit_code']==0
