"""A18-B2, AI (Codex). Training/validation/diagnostic simulation only.
Run with: py -I -X utf8 A18/diag/B2_第六类_20261104/run_networks.py
"""
from pathlib import Path
import sys, time, json, hashlib
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap_b2 import *

def prepare():
    P = AD.H.Parts(AD.RN, AD.BAND)
    lib = AD.all_fams(31337, AD.NTR, P, True)
    extra = AD.D.gen5(3*len(lib['同分布五类']),31344,AD.BAND)[0]
    full = np.concatenate(list(lib.values())+[extra])
    train = {k:np.concatenate([lib[k],extra]) if k=='同分布五类' else lib[k] for k in FAMILIES}
    va = AD.all_fams(31438,40,P,True)
    dg = AD.all_fams(31451,30,P,True)
    # Advance exactly the complete 26-family B1 diagnostic RNG stream.
    gd=np.random.default_rng(4321); gv=np.random.default_rng(54321)
    yd={k:AD.meas(s.T,gd,'real') for k,s in dg.items()}
    yv={k:AD.meas(s.T,gv,'real') for k,s in va.items()}
    def hashes(s):
        return {hashlib.sha256(row.tobytes()).hexdigest() for row in np.round(s,9)}
    htr=hashes(full); hva=hashes(np.concatenate(list(va.values())))
    hdg=hashes(np.concatenate(list(dg.values())))
    overlap=dict(train_vs_validation=len(htr&hva),train_vs_diagnostic=len(htr&hdg),
                 validation_vs_diagnostic=len(hva&hdg),decimals=9,test_generated=False)
    assert not any(overlap[k] for k in ('train_vs_validation','train_vs_diagnostic','validation_vs_diagnostic'))
    dump('overlap.json',overlap)
    payload={'full_train':full,'family_names':np.array(FAMILIES),'RN':AD.RN,'BM':AD.BM,'WL':AD.WL,
             'DEVCOV':AD.DEVCOV,'P0':np.array(AD.P0),'SCALE':np.array(AD.SCALE)}
    for j,k in enumerate(FAMILIES):
        payload.update({f'tr{j}':train[k],f'va{j}':va[k],f'dg{j}':dg[k],f'yv{j}':yv[k],f'yd{j}':yd[k]})
    np.savez_compressed(OUT/'allowed_data.npz',**payload)
    sb=full[:,BM]
    r=float(np.mean(sb**2)/np.mean(np.diff(sb,axis=1)**2))
    dump('protocol.json',dict(task='A18-B2',author='AI（Codex）',simulation_only=True,
        training_seed=31337,extra_training_seed=31344,validation_seed=31438,diagnostic_seed=31451,
        diagnostic_noise_seed=4321,validation_noise_seed=54321,n_draws=4,mode='real',
        train_power='log_uniform(0.5,2)*P0',evaluation_power='P0',
        noise=AD.ER,full_library_size=len(full),family_train_counts={k:len(train[k]) for k in FAMILIES},
        loss_r=r,steps=30000,batch_size=2048,init_seed=11,lr=1e-3,weight_decay=1e-4,
        checkpoint_rule='minimum equal-family mean real-mode shape NRMSE on VSEED=31438 every 1000 steps',
        variants=[dict(name='specialist_512x4',width=512,blocks=4),dict(name='specialist_768x6',width=768,blocks=6)],
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        source_sha256={str(p.relative_to(WORK)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()},
        B_sha256=hashlib.sha256((Path(M.NETDIR)/'cnet0.pt').read_bytes()).hexdigest(),
        matmul_precision=torch.get_float32_matmul_precision()))
    return payload,r

def predict(net,Y):
    y=Y.transpose(0,2,1).reshape(-1,Y.shape[1])
    with torch.no_grad():
        xx=M.nrm_t(torch.nn.functional.softplus(net(M.l1n(torch.tensor(y,dtype=torch.float32,device=M.DEV))))).cpu().numpy()
    x=np.zeros((len(xx),len(BM)));x[:,BM]=xx
    return x.reshape(Y.shape[0],Y.shape[2],len(BM)).transpose(0,2,1)

def errs(X,S):
    return np.array([AD.LD.band_nrmse(x,S.T) for x in X])

def evaluate(net,data,split):
    results={}
    for j,k in enumerate(FAMILIES):
        ss=data[f'{"va" if split=="validation" else "dg"}{j}']
        yy=data[f'{"yv" if split=="validation" else "yd"}{j}']
        results[k]=errs(predict(net,yy),ss)
    return results

def train_one(name,w,nb,data,r):
    S=np.concatenate([data[f'tr{j}'] for j in range(4)])
    st=torch.tensor(S,dtype=torch.float32,device=M.DEV)
    def init():
        torch.manual_seed(11)
        net=M.CNet(AD.N_RD,int(BM.sum()),w=w,nb=nb).to(M.DEV)
        opt=torch.optim.AdamW(net.parameters(),1e-3,weight_decay=1e-4)
        sch=torch.optim.lr_scheduler.OneCycleLR(opt,1e-3,total_steps=30000)
        return net,opt,sch
    net,opt,sch=init()
    def step(it):
        rng=np.random.default_rng(1100000+it-it%50)
        s=st[torch.randint(0,len(st),(2048,),device=M.DEV)]
        y=AD.H.meas_t(torch,AD.RN,s,rng,M.DEV)
        x=M.nrm_t(torch.nn.functional.softplus(net(M.l1n(y))))
        z=s[:,BM]
        loss=((x-z)**2).mean()+0.5*r*((x.diff(dim=1)-z.diff(dim=1))**2).mean()
        opt.zero_grad();loss.backward();opt.step();sch.step()
        return float(loss)
    # Small measured pilot; refuse the full run if its estimate exceeds 30 min.
    for it in range(20):step(it)
    torch.cuda.synchronize();t0=time.perf_counter()
    for it in range(20,220):step(it)
    torch.cuda.synchronize();estimate=(time.perf_counter()-t0)/200*30000
    log(f'{name}: 30000-step measured estimate {estimate:.1f}s')
    dump(name+'_budget.json',dict(pilot_steps=200,estimated_seconds=estimate,limit_seconds=1800))
    if estimate>1800:
        dump(name+'_skipped.json',dict(reason='estimated training exceeds 30 minutes',seconds=estimate))
        return None
    net,opt,sch=init();best=float('inf');curve=[];t0=time.perf_counter()
    for it in range(30000):
        loss=step(it)
        if (it+1)%1000==0:
            net.eval();ev=evaluate(net,data,'validation')
            score=float(np.mean([ev[k].mean() for k in TARGETS]))
            row=dict(step=it+1,loss_batch=loss,validation_equal_family=score,seconds=time.perf_counter()-t0,
                     **{f'val_{k}':float(v.mean()) for k,v in ev.items()})
            curve.append(row);csvwrite(name+'_curve.csv',curve)
            if score<best:
                best=score;beststep=it+1
                torch.save(dict(state=net.state_dict(),width=w,blocks=nb,step=beststep,validation=score,
                    r=r,source='training only; selected on VSEED=31438'),OUT/(name+'.pt'))
            log(f'{name}: step {it+1}, validation {score:.6f}, best {best:.6f} @ {beststep}, {row["seconds"]:.0f}s')
            net.train()
    elapsed=time.perf_counter()-t0
    torch.save(dict(state=net.state_dict(),width=w,blocks=nb,step=30000),OUT/(name+'_last.pt'))
    ck=torch.load(OUT/(name+'.pt'),weights_only=False);net.load_state_dict(ck['state']);net.eval()
    ee=evaluate(net,data,'diagnostic');vv=evaluate(net,data,'validation')
    np.savez_compressed(OUT/(name+'_errors.npz'),**{f'dg{j}':ee[k] for j,k in enumerate(FAMILIES)},
                        **{f'va{j}':vv[k] for j,k in enumerate(FAMILIES)})
    # Fixed 400 training spectra per family, real noise, no weight update.
    gap=[]
    for j,k in enumerate(FAMILIES):
        ss=data[f'tr{j}'];idx=np.random.default_rng(11).choice(len(ss),min(400,len(ss)),replace=False);ss=ss[idx]
        yy=AD.meas(ss.T,np.random.default_rng(810000+100*j),'real')
        gap.append(dict(family=k,train_real=float(errs(predict(net,yy),ss).mean()),
                        validation_real=float(vv[k].mean()),diagnostic_real=float(ee[k].mean())))
    csvwrite(name+'_gap.csv',gap)
    res=dict(name=name,parameters=sum(p.numel() for p in net.parameters()),width=w,blocks=nb,
             seconds=elapsed,selected_step=beststep,validation_equal_family=best,
             diagnostic={k:float(v.mean()) for k,v in ee.items()},peak_cuda_GB=torch.cuda.max_memory_allocated()/2**30)
    dump(name+'_summary.json',res)
    return res

if __name__=='__main__':
    t0=time.perf_counter();data,r=prepare();log('Allowed data prepared; no test spectra generated.')
    base=M.CNet(AD.N_RD,int(BM.sum())).to(M.DEV)
    base.load_state_dict(torch.load(Path(M.NETDIR)/'cnet0.pt',weights_only=True));base.eval()
    ev=evaluate(base,data,'diagnostic')
    with (WORK/'A18/diag/B1_诊断_20261104/family_metrics.csv').open(encoding='utf-8-sig') as f:
        old={v['family']:v for v in csv.DictReader(f)}
    checks=[]
    for k in FAMILIES:
        e=float(ev[k].mean());delta=e-float(old[k]['B_raw'])
        assert abs(delta)<1e-6,(k,e,old[k]['B_raw'])
        checks.append(dict(family=k,B2_reproduced_B=e,B1_B=float(old[k]['B_raw']),difference=delta))
    csvwrite('B1_reproduction.csv',checks)
    np.savez_compressed(OUT/'baseline_errors.npz',**{f'dg{j}':ev[k] for j,k in enumerate(FAMILIES)})
    log('B1 baseline reproduced within 1e-6 on identical diagnostic readings (float32 GPU tolerance).')
    del base
    results=[]
    for name,w,nb in [('specialist_512x4',512,4),('specialist_768x6',768,6)]:
        if (OUT/(name+'_summary.json')).exists():
            results.append(json.loads((OUT/(name+'_summary.json')).read_text(encoding='utf-8')))
        else:results.append(train_one(name,w,nb,data,r))
    dump('network_summary.json',dict(models=results,total_seconds=time.perf_counter()-t0))
    log('Network phase complete.')
