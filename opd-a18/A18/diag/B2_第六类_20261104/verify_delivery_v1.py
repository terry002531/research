"""Final read-back verification of A18-B2. New evidence files only."""
from pathlib import Path
import sys,json,csv,hashlib,time,zipfile
from datetime import datetime,timezone
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;WORK=OUT.parents[2]
REPORT=WORK/'A18/HANDOFF_A18-B2.md'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def readcsv(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
before=json.loads((OUT/'existing_files_before.json').read_text(encoding='utf-8'))
original_changes={}
for key,v in before.items():
    p=WORK/key
    if not p.exists() or p.stat().st_size!=v['size'] or p.stat().st_mtime_ns!=v['mtime_ns']:
        original_changes[key]=v
assert not original_changes,original_changes
resume_hash=json.loads((OUT/'resume_input_hashes_v2.json').read_text(encoding='utf-8'))
hash_changes={key:old for key,old in resume_hash.items() if not (OUT/key).exists() or sha(OUT/key)!=old}
assert not hash_changes,hash_changes

import numpy as np
import torch
metrics=readcsv(OUT/'family_metrics_complete_resume_v2.csv')
oracle=readcsv(OUT/'oracle_metrics_complete_resume_v2.csv')
null=readcsv(OUT/'nullspace_metrics_complete_resume_v2.csv')
per=readcsv(OUT/'per_reading_complete_resume_v2.csv')
assert len(metrics)==len(oracle)==len(null)==6
assert len(per)==720
oe=np.load(OUT/'oracle_errors_complete_resume_v2.npz')
net=json.loads((OUT/'network_summary.json').read_text(encoding='utf-8'))
for model in net['models']:
    ck=torch.load(OUT/(model['name']+'.pt'),map_location='cpu',weights_only=False)
    last=torch.load(OUT/(model['name']+'_last.pt'),map_location='cpu',weights_only=False)
    assert ck['step']==model['selected_step'] and last['step']==30000
    assert sum(v.numel() for v in ck['state'].values())==model['parameters']
for j,row in enumerate(metrics):
    assert np.isclose(oe[f'dg{j}'].mean(),float(row['oracle']),atol=1e-12,rtol=0)
    current=[v for v in per if v['family']==row['family']]
    assert len(current)==120
    for name in ('B','A','oracle','specialist_512x4','specialist_768x6'):
        assert abs(np.mean([float(v[name]) for v in current])-float(row[name]))<1e-12
    assert abs(float(row['null_energy_percent'])-100*float(null[j]['null_energy_fraction']))<1e-12
    assert float(oracle[j]['quadrature_max_spectrum_difference'])<1e-6
    assert abs(float(oracle[j]['quadrature_mean_nrmse_difference']))<5e-8

raw=REPORT.read_bytes();doc=raw.decode('utf-8')
assert b'\r' not in raw,'Unexpected carriage return / escaped Windows path'
assert '\ufffd' not in doc and '???' not in doc
assert 'AI（Codex）' in doc and 'VSEED+13=31451 是 A 的 SS 调参集' in doc
assert '偏乐观' in doc and '不是严格下界' in doc and '仿真' in doc
assert doc.count('```')%2==0
for row in metrics:
    for name in ('B','A','oracle','specialist_512x4','specialist_768x6'):
        assert f"{float(row[name]):.5f}" in doc
for name in ('allowed_data.npz','oracle_errors_complete_resume_v2.npz','five_subtype_missing_predictions_resume_v2.npz'):
    with zipfile.ZipFile(OUT/name) as z:assert z.testzip() is None
imagefile=OUT/'B2_comparison_resume_v1.png'
from PIL import Image
im=Image.open(imagefile);im.verify()
assert imagefile.stat().st_size>20000

# The numerical functions used are still the exact B1-source revisions.
protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
for key,digest in protocol['source_sha256'].items():assert sha(WORK/key)==digest
bpath=WORK/'算法对照/45_Uddin2024_真实R矩阵/results/a15_nets/cnet0.pt'
assert sha(bpath)==protocol['B_sha256']

now=datetime.now(timezone.utc);start=datetime(2026,11,4,2,26,59,tzinfo=timezone.utc)
pause=json.loads((OUT/'timing_report_v1.json').read_text(encoding='utf-8'))['approximate_interruption_seconds']
elapsed=(now-start).total_seconds()
manifest={str(p.relative_to(WORK)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in OUT.rglob('*') if p.is_file()}
manifest[str(REPORT.relative_to(WORK))]=dict(bytes=REPORT.stat().st_size,sha256=sha(REPORT))
with (OUT/'delivery_manifest_v1.json').open('x',encoding='utf-8') as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
result=dict(task='A18-B2',author='AI（Codex）',verified_utc=now.isoformat(),
    original_files_checked=len(before),original_file_changes=original_changes,
    pre_resume_hashes_checked=len(resume_hash),pre_resume_hash_changes=hash_changes,
    report_utf8=True,report_numbers_match=True,SS_selection_bias_disclosed=True,
    networks_checkpoint_and_counts_verified=True,all_six_oracles_complete=True,
    total_spectra=180,total_individual_readings=720,simulation_only=True,
    figure_visually_inspected_separately=True,test_data_not_used=True,
    original_source_and_baseline_hashes_match=True,
    elapsed_minutes=elapsed/60,approximate_pause_minutes=pause/60,approximate_active_minutes=(elapsed-pause)/60,
    evidence_time='推算；时钟时间戳已核实，中断起点取末条旧日志')
with (OUT/'delivery_verification_v1.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False,indent=2))
