"""AI (Codex): descriptive uncertainty for the reported aggregate, never selection."""
from pathlib import Path
import csv
import hashlib
import json
import sys
import time
import numpy as np

sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent
assert (OUT/'summary.json').exists(), 'Run only after the frozen evaluation has completed.'
assert not (OUT/'aggregate_uncertainty.json').exists(), 'Preserve existing inference.'
samples=list(csv.DictReader((OUT/'per_reading_metrics.csv').open(encoding='utf-8-sig')))
families=list(dict.fromkeys(r['family'] for r in samples))
keys=('B_raw','B_consistent','fusion_pre_q','final','hard_raw','hard_consistent')
assert len(samples)==15600 and len(families)==26
data=[]
for family in families:
    rows=sorted((r for r in samples if r['family']==family),key=lambda r:(int(r['spectrum']),int(r['draw'])))
    assert [(int(r['spectrum']),int(r['draw'])) for r in rows]==[(i,d) for i in range(150) for d in range(4)]
    data.append(np.array([[float(r[k]) for k in keys] for r in rows]).reshape(150,4,len(keys)).mean(1))
x=np.stack(data)
assert x.shape==(26,150,6) and np.isfinite(x).all() and (x>0).all()
summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
point=np.exp(np.log(x.mean(1)).mean(0))
assert np.allclose(point,[summary['geometric_means'][k] for k in keys],rtol=1e-12,atol=1e-13)
comparisons=(('fusion_pre_q','B_raw'),('final','B_raw'),('hard_raw','B_raw'),
             ('hard_consistent','B_consistent'),('final','fusion_pre_q'))
nboot=200000
rng=np.random.default_rng(1901122)
boot=np.empty((nboot,len(keys)))
t0=time.time()
for start in range(0,nboot,200):
    ids=rng.integers(0,150,(min(200,nboot-start),26,150))
    means=x[np.arange(26)[None,:,None],ids].mean(2)
    boot[start:start+len(ids)]=np.exp(np.log(means).mean(1))
results=[]
for method,reference in comparisons:
    i,j=keys.index(method),keys.index(reference)
    gain=100*(1-boot[:,i]/boot[:,j])
    lo,hi=np.quantile(gain,[.025,.975])
    results.append(dict(method=method,reference=reference,
        improvement_percent=float(100*(1-point[i]/point[j])),
        percentile_95_ci_percent=[float(lo),float(hi)]))
result=dict(author='AI（Codex）',purpose='Descriptive aggregate uncertainty only; no B/q/routing/seed selection.',
    preregistered_primary_criterion=False,
    statistic='Percent reduction of geometric mean of 26 family arithmetic mean NRMSE values.',
    unit='Spectrum, retaining all four paired draws; independent stratified resampling within each family.',
    bootstrap_n=nboot,bootstrap_seed=1901122,interval='Unadjusted two-sided 95 percent percentile intervals; no simultaneous-coverage claim.',
    comparisons=results,seconds=time.time()-t0,
    input_sha256={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest()
                  for name in ('per_reading_metrics.csv','summary.json')})
(OUT/'aggregate_uncertainty.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('input_sha256',)},ensure_ascii=False,indent=2))
