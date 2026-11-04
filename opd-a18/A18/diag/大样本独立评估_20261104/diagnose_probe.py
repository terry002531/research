from pathlib import Path
import sys,time,json,csv
sys.path.insert(0,str(Path(__file__).resolve().parent))
import frozen_runtime as r
np=r.np
print('Groups: '+json.dumps(r.GROUPS,ensure_ascii=False),flush=True)
ds=r.ORIGINAL_FAMS(31600,30,r.P,True)
k,S=next(iter(ds.items()))
Y=r.AD.meas(S.T,np.random.default_rng(8191),'real')
with (r.WORK/'A18/diag/B3_新种子_20261104/per_reading_metrics.csv').open(encoding='utf-8-sig') as f:
    rows=[x for x in csv.DictReader(f) if x['family']==k]
refa=np.array([[float(next(x for x in rows if int(x['draw'])==d and int(x['spectrum'])==i)['A']) for i in range(30)] for d in range(4)])
refb=np.array([[float(next(x for x in rows if int(x['draw'])==d and int(x['spectrum'])==i)['B_raw']) for i in range(30)] for d in range(4)])
t=time.perf_counter();a=r.decode_A(Y,large_batch=True);fast=time.perf_counter()-t
b=r.decode_B(Y)
t=time.perf_counter();standard=r.decode_A(Y,large_batch=False);slow=time.perf_counter()-t
asub=r.decode_A(Y[:,:,:4],large_batch=True)
bsub=r.decode_B(Y[:,:,:4])
out=dict(groups=r.GROUPS,full_batch_A_error_max=float(np.max(np.abs(r.errors(a,S)-refa))),
    full_batch_B_error_max=float(np.max(np.abs(r.errors(b,S)-refb))),
    full_batch_A_fast_vs_standard_point_max=float(np.max(np.abs(a-standard))),
    A_subbatch_vs_full_error_max=float(np.max(np.abs(r.errors(asub,S[:4])-r.errors(a,S)[:,:4]))),
    B_subbatch_vs_full_error_max=float(np.max(np.abs(r.errors(bsub,S[:4])-r.errors(b,S)[:,:4]))),
    fast_seconds=fast,standard_seconds=slow,peak_GiB=r.torch.cuda.max_memory_allocated()/2**30)
(r.OUT/'probe_diagnosis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2),flush=True)
