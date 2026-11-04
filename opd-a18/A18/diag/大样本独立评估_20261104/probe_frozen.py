"""Reproduce archived readings before registering or generating the new set."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import frozen_runtime as r
import csv,json,time,hashlib
np,torch = r.np,r.torch
O,W = r.OUT,r.WORK
start = time.perf_counter()
old = W/'A18/diag/B3_新种子_20261104'
with (old/'per_reading_metrics.csv').open(encoding='utf-8-sig') as f:
    archived = {(x['family'],int(x['spectrum']),int(x['draw'])):x for x in csv.DictReader(f)}
ds = r.ORIGINAL_FAMS(31600,30,r.P,True)
rng = np.random.default_rng(8191)
rows=[]; max_a=0.;max_b=0.;pointwise=0.;batch_seconds=0.;old_seconds=0.
for i,(key,S) in enumerate(ds.items()):
    Y = r.AD.meas(S.T,rng,'real')
    s=S
    t=time.perf_counter(); A,post = r.decode_A(Y,large_batch=True,return_posteriors=True)
    batch_seconds += time.perf_counter()-t
    B=r.decode_B(Y)
    ea,eb=r.errors(A,s),r.errors(B,s)
    name=key.replace('留出·','')
    da=max(abs(float(ea[z,j])-float(archived[name,j,z]['A'])) for j in range(30) for z in range(4))
    db=max(abs(float(eb[z,j])-float(archived[name,j,z]['B_raw'])) for j in range(30) for z in range(4))
    max_a=max(max_a,da);max_b=max(max_b,db)
    if i == 0:
        t=time.perf_counter(); standard=r.decode_A(Y,large_batch=False)
        old_seconds += time.perf_counter()-t
        delta=float(np.max(np.abs(A-standard)));pointwise=max(pointwise,delta)
        assert np.allclose(A,standard,rtol=1e-8,atol=1e-9),(name,delta)
    assert da < 1e-6 and db < 1e-6,(name,da,db)
    assert np.isfinite(post).all() and np.allclose(post.sum(-1),1,atol=1e-8)
    rows.append(dict(family=name,max_A_error_difference=da,max_B_error_difference=db))
    print(f'{name}: archived A/B errors reproduced; A diff={da:.3g}, B diff={db:.3g}',flush=True)
result=dict(passed=True,new_evaluation_spectra_generated=False,archived_seed=31600,
    spectra_checked=780,readings_checked=3120,max_A_error_difference=max_a,max_B_error_difference=max_b,
    batch16_vs_original_pointwise_max=pointwise,numerical_batch_change_only=True,
    current_A_includes_same_B_auxiliary=True,family_groups=r.GROUPS,mixture_component_count=len(r.GF.classes),
    benchmark_batch16_seconds_for_3120_readings=batch_seconds,
    benchmark_original_seconds_for_120_readings=old_seconds,seconds=time.perf_counter()-start,
    peak_cuda_GiB=torch.cuda.max_memory_allocated()/2**30,
    noise=r.AD.ER,P0_W=r.AD.P0,source_read_paths=sorted(str(p) for p in r.PROJECT_READS),
    families=rows)
(O/'runtime_probe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:result[k] for k in ('passed','max_A_error_difference','max_B_error_difference','batch16_vs_original_pointwise_max','family_groups','benchmark_batch16_seconds_for_3120_readings','peak_cuda_GiB')},ensure_ascii=False,indent=2),flush=True)
