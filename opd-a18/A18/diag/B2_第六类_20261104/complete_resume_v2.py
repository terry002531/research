"""Resume A18-B2 without overwriting ANY pre-existing file.
Only reconstruct the missing per-reading errors for the selected five-subtype
prior. Reuse complete controls, network runs, and the other family results.
"""
from pathlib import Path
import sys,os,time,json,hashlib,csv,ast
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent
WORK=OUT.parents[2]
started=time.time()

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
existing={str(p.resolve()):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
          for p in WORK.rglob('*') if p.is_file()}
protected=set(existing)
old_hashes={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}
def immutable(event,args):
    if event=='open':
        path,mode,flags=args
        if not isinstance(path,(str,bytes,os.PathLike)):return
        if os.fsdecode(path).lower() in ('nul',os.devnull.lower()):return
        p=Path(os.fsdecode(path)).resolve()
        write=isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if write and (str(p) in protected or not p.is_relative_to(OUT)):
            raise PermissionError(f'Resume refuses old file or outside write: {p}')
sys.addaudithook(immutable)
(OUT/'resume_input_hashes_v2.json').write_text(json.dumps(old_hashes,indent=2,ensure_ascii=False),encoding='utf-8')

# Original B1 initialization sanitizer, with ALL output names redirected to new
# resume files. Existing numerical functions and source files stay unchanged.
boot=(OUT/'bootstrap_b2.py').read_text(encoding='utf-8')
boot=boot.replace('existing_files_before.json','resume_initial_outside_snapshot_v2.json')
boot=boot.replace('run_validated.log','resume_execution_v2.log')
boot=boot.replace('strict_validation_adapter.json','strict_validation_adapter_resume_v2.json')
exec(compile(boot,str(OUT/'bootstrap_b2.py'),'exec'),globals())
from scipy.linalg import cho_factor,cho_solve,solve_triangular
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
threadpool_limits(limits=4)
tree=ast.parse((OUT/'run_oracles.py').read_text(encoding='utf-8'))
funcs=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ('posterior','metric','invisibility')]
exec(compile(ast.Module(body=funcs,type_ignores=[]),str(OUT/'run_oracles.py'),'exec'),globals())

def readcsv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def newcsv(name,rows):
    with (OUT/name).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def newjson(name,obj):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)

data=np.load(OUT/'allowed_data.npz')
assert data['full_train'].shape==(87000,176)
assert list(data['family_names'])==FAMILIES
assert np.all(np.isfinite(data['full_train']))
for j in range(6):
    assert data[f'dg{j}'].shape==(30,176) and data[f'yd{j}'].shape==(4,41,30)
    assert data[f'va{j}'].shape==(40,176) and data[f'yv{j}'].shape==(4,41,40)

latest={v['family']:v for v in readcsv(OUT/'oracle_metrics.csv')}
assert len(latest)==4 and all(k in latest for k in TARGETS)
previous=json.loads((OUT/'oracle_summary.json').read_text(encoding='utf-8'))
oldmetrics={v['family']:v for v in previous['metrics']}
stored_errors=np.load(OUT/'oracle_errors.npz')
oracle_rows=[];oracle_errors={};null_rows=[]
null_detail=json.loads((OUT/'nullspace_detail.json').read_text(encoding='utf-8'))
R=data['RN'][:,BM];noise=data['DEVCOV']
missing=time.perf_counter()
for j,k in enumerate(FAMILIES):
    source=latest[k] if k in latest else oldmetrics[k]
    row={n:(str(v) if n in ('family','prior_kind') else float(v)) for n,v in source.items()}
    row.setdefault('prior_kind','spectral_kmeans')
    priorfile=OUT/f'prior_{j}.npz'
    prior=np.load(priorfile)
    assert 1 <= len(prior['mu']) <= int(row['selected_K'])
    row['actual_K']=len(prior['mu'])
    assert np.isclose(prior['weight'].sum(),1) and np.all(prior['weight']>0)
    assert np.max(np.abs(prior['mu'].sum(1)-1))<1e-12
    if j==3:
        # The interrupted run wrote this family's metrics/prior but never its
        # final errors archive. Nothing is refitted or reselected here.
        y=data['yd3'].transpose(0,2,1).reshape(-1,41)
        pred=posterior(prior,y,R,noise)
        e=metric(pred,np.tile(data['dg3'],(4,1))).reshape(4,30)
        invariant=posterior(prior,y*np.linspace(.3,3,len(y))[:,None],R,noise)
        scale_difference=float(np.max(np.abs(pred-invariant)))
        assert scale_difference<1e-10,scale_difference
        np.savez_compressed(OUT/'five_subtype_missing_predictions_resume_v2.npz',prediction=pred,errors=e)
        origin='latest four-row CSV and prior_3; missing per-reading posterior recomputed, no fitting'
    else:
        e=stored_errors[f'dg{j}']
        origin='latest four-row CSV + complete prior archive errors' if j<3 else 'earlier complete oracle_summary + prior/error archives; no rerun'
    assert e.shape==(4,30) and np.all(np.isfinite(e))
    assert abs(float(e.mean())-row['oracle_nrmse'])<1e-10,(k,e.mean(),row)
    oracle_errors[f'dg{j}']=e
    row.update(source=origin,evidence='已核实（仿真）；近似oracle，非严格下界')
    oracle_rows.append(row)
    nu=null_detail[k]
    null_rows.append(dict(family=k,rank=nu['rank'],rank_relative_2pct=nu['rank_relative_2pct'],
        null_energy_fraction=nu['null_energy_fraction'],weak_2pct_energy_fraction=nu['weak_2pct_energy_fraction'],
        centered_area_null_fraction=nu['centered_area_null_fraction'],
        centered_area_noise_unidentifiable_fraction=nu['centered_area_noise_unidentifiable_fraction'],
        noise_SNR_below_one_directions=nu['noise_SNR_below_one_directions'],evidence='推算（仿真真谱投影）'))
np.savez_compressed(OUT/'oracle_errors_complete_resume_v2.npz',**oracle_errors)
newcsv('oracle_metrics_complete_resume_v2.csv',oracle_rows)
newcsv('nullspace_metrics_complete_resume_v2.csv',null_rows)
log('Complete oracle results assembled; only missing same-five per-reading posterior computed.')

b1dir=WORK/'A18/diag/B1_诊断_20261104'
b1={v['family']:v for v in readcsv(b1dir/'family_metrics.csv')}
baseline=np.load(OUT/'baseline_errors.npz')
netall=json.loads((OUT/'network_summary.json').read_text(encoding='utf-8'))
neterr={m['name']:np.load(OUT/(m['name']+'_errors.npz')) for m in netall['models']}
aerrors={k:np.empty((4,30)) for k in FAMILIES}
for v in readcsv(b1dir/'per_reading_metrics.csv'):
    if v['family'] in aerrors:aerrors[v['family']][int(v['draw']),int(v['spectrum'])]=float(v['A'])

rows=[];stats=[];perreading=[]
for j,k in enumerate(FAMILIES):
    eb=baseline[f'dg{j}'];ea=aerrors[k];eo=oracle_errors[f'dg{j}']
    assert abs(eb.mean()-float(b1[k]['B_raw']))<1e-6
    assert abs(ea.mean()-float(b1[k]['A']))<1e-12
    es={n:z[f'dg{j}'] for n,z in neterr.items()}
    for model in netall['models']:
        name=model['name']
        assert abs(es[name].mean()-model['diagnostic'][k])<1e-12
        curve=readcsv(OUT/(name+'_curve.csv'))
        assert int(curve[-1]['step'])==30000
        selected=min(curve,key=lambda v:float(v['validation_equal_family']))
        assert int(selected['step'])==model['selected_step']
    row=dict(family=k,B=float(eb.mean()),A=float(ea.mean()),oracle=float(eo.mean()),
        null_energy_percent=100*null_rows[j]['null_energy_fraction'],
        noise_weak_variation_percent=100*null_rows[j]['centered_area_noise_unidentifiable_fraction'],
        specialist_512x4=float(es['specialist_512x4'].mean()),
        specialist_768x6=float(es['specialist_768x6'].mean()),
        oracle_reduction_vs_B_percent=100*(1-eo.mean()/eb.mean()),
        evidence_errors='已核实（仿真）',evidence_projection_and_reductions='推算',
        A_SS_selection_bias='VSEED+13 used for SS selection; optimistic, especially steps/parts')
    rows.append(row)
    indices=np.random.default_rng(20261104).integers(0,30,(5000,30))
    allerrors={'A':ea,'oracle':eo,**es}
    for name,ev in allerrors.items():
        delta=(ev-eb).mean(0);boot=delta[indices].mean(1);ci=np.quantile(boot,[.025,.975])
        stats.append(dict(family=k,comparison=name+' minus B',delta=float(delta.mean()),
            relative_change_percent=100*(ev.mean()/eb.mean()-1),ci_low=float(ci[0]),ci_high=float(ci[1]),
            unit='spectrum, all four noise readings kept together',resamples=5000,
            evidence='推算；单诊断种子条件下的成对区间，未校正多重比较'))
    for draw in range(4):
        for sid in range(30):
            perreading.append(dict(family=k,spectrum=sid,draw=draw,B=float(eb[draw,sid]),A=float(ea[draw,sid]),
                oracle=float(eo[draw,sid]),**{n:float(v[draw,sid]) for n,v in es.items()}))
newcsv('family_metrics_complete_resume_v2.csv',rows)
newcsv('paired_comparisons_resume_v2.csv',stats)
newcsv('per_reading_complete_resume_v2.csv',perreading)

newjson('completion_inventory_resume_v2.json',dict(
    complete_reused=['allowed_data.npz','protocol.json','overlap.json','B1_reproduction.csv',
        'baseline_errors.npz','network_summary.json','specialist_512x4*','specialist_768x6*',
        'latest oracle_metrics.csv four target rows and prior_0..3',
        'previous oracle_summary.json control rows and prior_4..5 and oracle_errors.npz controls',
        'nullspace_detail.json six families'],
    stale_or_partial=['oracle_metrics.csv contains only four latest rows',
        'oracle_summary.json/oracle_errors.npz same-five entry is older K=256; superseded by latest K=320 subtype mixture',
        'oracle_candidates.csv has complete four target families, interrupted controls are absent',
        'nullspace_metrics.csv has only four latest rows'],
    recomputed_only='latest same-five selected-prior per-reading posterior and scale-invariance verification; no training or prior refit',
    controls='both already complete in previous run; checked shapes, selected prior sizes and error means; reused without rerunning',
    no_old_file_overwritten=True))

changed={p:dict(before=st,after=(dict(size=Path(p).stat().st_size,mtime_ns=Path(p).stat().st_mtime_ns) if Path(p).exists() else None))
         for p,st in existing.items() if not Path(p).exists() or Path(p).stat().st_size!=st['size'] or Path(p).stat().st_mtime_ns!=st['mtime_ns']}
hashchanged={p:dict(before=h,after=sha(OUT/p)) for p,h in old_hashes.items() if sha(OUT/p)!=h}
assert not changed and not hashchanged,(changed,hashchanged)
newjson('resume_verification_v2.json',dict(old_file_metadata_changes=changed,old_B2_hash_changes=hashchanged,
    recompute_seconds=time.perf_counter()-missing,resume_script_seconds=time.time()-started,
    gain_nodes=15,radial_nodes=48,scale_invariance_max_abs=scale_difference,
    incomplete_same5_fixed=True,networks_complete=True,control_oracles_reused=True,
    no_test_data_generated=True,old_files_checked=len(existing),old_B2_hashes_checked=len(old_hashes)))
for r in rows:print(json.dumps(r,ensure_ascii=False),flush=True)
