import json,subprocess
from pathlib import Path
R=Path.cwd();source=(R / ".local/resources/LC-CONTEXT-COUNTER-001/feature-v1/baseline/libs/langchain_v1/langchain/agents/middleware/context_editing.py").read_text()
args=json.dumps({'command':"cat > /tmp/offline_payload_probe.py <<'EOF'\n"+source+"\nEOF"});reason=[]
for label in ['context-counter-feature-v1','context-counter-incremental-files-v1']:
 p=next((R/'runs/harbor').glob(label+'-minimax-attempt-001/LC-*/agent/requests.jsonl'))
 for line in p.read_text().splitlines():
  e=json.loads(line)
  if e['event']=='model_response':
   for c in e['response'].get('choices',[]):reason.append(c.get('message',{}).get('reasoning') or '')
image=json.loads((R/'configs/comparison-4x2x3-v1-LC-TIMEOUT-001-minimax-model.json').read_text())['tasks'][0]['path'];import toml
image=toml.loads((Path(image)/'task.toml').read_text())['environment']['docker_image']
script='''import json,sys,tiktoken
p=json.load(sys.stdin);out={}
for name in ['cl100k_base','o200k_base']:
 e=tiktoken.get_encoding(name)
 out[name]={'representative_complete_tool_args_proxy_tokens':len(e.encode(p['arguments'])),'maximum_observed_textual_reasoning_proxy_tokens':max(len(e.encode(r)) for r in p['reason'])}
print(json.dumps(out))
'''
p=subprocess.run(['docker','run','--rm','--interactive','--network=none','--read-only',image,'python','-c',script],input=json.dumps({'arguments':args,'reason':reason}),text=True,capture_output=True,timeout=60);assert p.returncode==0,p.stderr
counts=json.loads(p.stdout)
report={'model_requests':0,'planned_response_tokens':6000,'cumulative_output_tokens_unchanged':10000,'representative_payload':'JSON-encoded bash tool arguments containing the unchanged baseline context_editing.py; no reference patch','tokenizers':'offline cl100k_base/o200k_base proxies, not MiniMax-native billing or guaranteed native counts','proxy_measurements':counts,'planned_reasoning_allowance_tokens':1536,'tool_envelope_margin_tokens':256,'reasoning_setting_unchanged':'omitted for MiniMax; zero reported reasoning does not establish absence of reasoning','remaining_cumulative_budget_after_one_maximum_response':4000,'inference_cost_usd':0}
for row in counts.values():
 row['reasoning_allowance_tokens']=max(1536,row['maximum_observed_textual_reasoning_proxy_tokens'])
 row['planned_combined_proxy_tokens']=row['representative_complete_tool_args_proxy_tokens']+row['reasoning_allowance_tokens']+256
 row['headroom_to_6000']=6000-row['planned_combined_proxy_tokens'];assert row['headroom_to_6000']>0
(R/'reports/context-counter-response-6000-v1-headroom.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
