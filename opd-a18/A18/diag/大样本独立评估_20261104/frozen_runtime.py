"""Load the unchanged A/B numerical functions and an already-frozen A state."""
from pathlib import Path
import os, sys, ast, json, time, hashlib, importlib.abc, importlib.util
OUT = Path(__file__).resolve().parent
WORK = OUT.parents[2]
ALG = WORK / '算法对照'
sys.dont_write_bytecode = True
for key in ('TMP','TEMP','MPLCONFIGDIR','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR'):
    p = OUT/'runtime'/key
    p.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(p)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
PROJECT_READS = set()
def audit(event,args):
    if event == 'open':
        path,mode,flags = args
        if not isinstance(path,(str,bytes,os.PathLike)): return
        if os.fsdecode(path).lower() in ('nul',os.devnull.lower()): return
        p = Path(os.fsdecode(path)).absolute()
        write = bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)) if isinstance(flags,int) else False
        if write and not p.is_relative_to(OUT):
            raise PermissionError('Write outside new evaluation directory: '+str(p))
        norm = str(p).replace('\\','/').lower()
        if not write and ('/a18/results/' in norm or any(s in norm for s in ('a18_seed6161','a18_seed4242','testset_','peakfix.json','qnet.pt'))):
            raise PermissionError('Forbidden historical test/result access: '+str(p))
        if not write and p.is_relative_to(WORK.parent) and not p.is_relative_to(OUT):
            PROJECT_READS.add(p)
    elif event in ('os.remove','os.rmdir','os.rename'):
        ps = args[:2] if event == 'os.rename' else args[:1]
        if any(not Path(p).absolute().is_relative_to(OUT) for p in ps):
            raise PermissionError('Destructive operation outside new evaluation directory')
    elif event == 'os.mkdir':
        p = Path(args[0]).absolute()
        if not p.is_relative_to(OUT) and not p.is_dir():
            raise PermissionError('Directory outside evaluation')
sys.addaudithook(audit)
import numpy as np
import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(2)
assert torch.cuda.is_available()
import warnings
warnings.filterwarnings('ignore',category=UserWarning)

# Execute exactly the reviewed B3 import sanitizer, ending before its fitting,
# generation, fingerprint, logging, or evaluation drivers.
source = WORK/'A18/diag/B3_新种子_20261104/diagnose_b3_newseed.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == 'f111ceaed910e926b9d46419f541045e7d97d2a04670edfdbf6d5966ea4e64e6'
text = source.read_text(encoding='utf-8')
section = text[text.index('# Only in-memory initialization'):text.index('start = time.time()')]
exec(compile(section,str(source)+' [import adapter only]','exec'),globals())

def no_fit(*args,**kwargs):
    raise RuntimeError('Fitting/training is forbidden in this expanded evaluation')
M.train_cnet = M.fit_blend3 = M.gf_a16a = M.build3 = PF.fit_q = no_fit
FROZEN_PATH = WORK/'A18/diag/B2_新种子63862_20261104/frozen_A_state.pt'
assert hashlib.sha256(FROZEN_PATH.read_bytes()).hexdigest() == 'dbc9c93718cbe44d9d41fae9656610b6e563ada937cf6692f018674007fbfddb'
frozen = torch.load(FROZEN_PATH,map_location=M.DEV,weights_only=False)
def restore(cls,state):
    obj = cls.__new__(cls)
    obj.__dict__.update(state)
    obj.t = torch
    return obj
GF = restore(AD.Q.GaussX,frozen['GF'])
GM = restore(AD.S7.GaussMR,frozen['GM'])
PM = restore(AD.D.ParamE,frozen['PM'])
SV = restore(a16_ss.SS,frozen['SS'])
GSS = a16_ss.GateSS.__new__(a16_ss.GateSS)
GSS.SV, GSS.GM, GSS.R, GSS.idx = SV, GM, AD.RN, frozen['ss_idx']
assert frozen['sigma_e'] == 0.08 and frozen['ss_alpha'] == a16_ss.ALPHA
assert frozen['ss_CFG'] == a16_ss.CFG and np.array_equal(frozen['escale_grid'],a16_ss.CG)
P = AD.H.Parts(AD.RN,AD.BAND)
NT8 = dict(net=torch.nn.Sequential(torch.nn.Linear(AD.N_RD,512),torch.nn.ReLU(),
    torch.nn.Linear(512,512),torch.nn.ReLU(),torch.nn.Linear(512,512),torch.nn.ReLU(),
    torch.nn.Linear(512,P.na)).to(M.DEV), dev=M.DEV, torch=torch,
    mu=frozen['NT8_mu'],sd=frozen['NT8_sd'])
NT8['net'].load_state_dict(frozen['NT8'])
NT8['net'].eval()
M.XN = M.CNet(AD.N_RD,int(BM.sum())).to(M.DEV)
M.XN.load_state_dict(frozen['B'])
M.XN.eval()
bpath = Path(M.NETDIR)/'cnet0.pt'
reference_b = torch.load(bpath,map_location=M.DEV,weights_only=True)
assert all(torch.equal(v,reference_b[k]) for k,v in M.XN.state_dict().items())
del reference_b
ORIGINAL_FAMS, ORIGINAL_GEN5 = AD.all_fams, AD.D.gen5

# The optional larger numerical batch changes only vectorization, never the
# model, scale grid, solver, objective, covariance, or selection rule. Its
# pointwise equality is checked on historical data before new spectra exist.
node = ast.parse(source.read_text(encoding='utf-8')) # source provenance only
import inspect
escale_tree = ast.parse(inspect.getsource(a16_ss.escale))
escale_tree.body[0].name = 'escale_batch16'
class BatchOnly(ast.NodeTransformer):
    def visit_Assign(self,node):
        if any(isinstance(x,ast.Name) and x.id == 'ch' for x in node.targets):
            node.value = ast.Constant(16)
        return node
escale_tree = BatchOnly().visit(escale_tree)
ast.fix_missing_locations(escale_tree)
exec(compile(escale_tree,'<identical escale, batch 16>','exec'),a16_ss.__dict__)
GROUPS = list(dict.fromkeys(c.split('#')[0] for c in GF.classes))
POOL = np.zeros((len(GF.classes),len(GROUPS)))
for i,c in enumerate(GF.classes): POOL[i,GROUPS.index(c.split('#')[0])] = 1
def decode_A(Y,large_batch=True,return_posteriors=False):
    a = M.t3_infer(AD.RN,Y)[:,BM,:].max(1)
    yy = Y/np.maximum(a,1e-30)[:,None,:]
    c = (a16_ss.escale_batch16 if large_batch else a16_ss.escale)(GF,yy)[:,None,:]
    yy = yy*c
    xp = PM.solve(yy,0.08)[0]
    x8 = AD.H.solve_pick(P,yy,AD.H.infer(NT8,yy))[0]
    xf,df,wf = GF.solve_u(yy)
    gp = AD.DS.fuse_parts(AD.RN,xf,wf,df,[xp,x8],yy,GF,GF.classes)
    gate = np.clip(AD.DS.gate_mean(gp),0,None)
    gate = GSS.apply(yy,gate,wf,np.clip(M.t3_infer(AD.RN,yy),0,None))
    result = gate/c
    if return_posteriors:
        pooled = np.einsum('dmk,kg->dmg',wf,POOL)
        return result,pooled
    return result
def decode_B(Y):
    # Same input scaling used inside the existing A18 decode_ac path.
    a = M.t3_infer(AD.RN,Y)[:,BM,:].max(1)
    return np.clip(M.t3_infer(AD.RN,Y/np.maximum(a,1e-30)[:,None,:]),0,None)
def errors(X,S):
    return np.array([AD.LD.band_nrmse(x,S.T) for x in X])
def tensor_digest():
    h = hashlib.sha256()
    def add(value):
        if isinstance(value,torch.Tensor):
            x = value.detach().cpu().contiguous().numpy()
            h.update(str(x.dtype).encode());h.update(str(x.shape).encode());h.update(x.tobytes())
        elif isinstance(value,np.ndarray):
            h.update(str(value.dtype).encode());h.update(str(value.shape).encode());h.update(np.ascontiguousarray(value).tobytes())
        elif isinstance(value,dict):
            for k,v in sorted(value.items(),key=lambda x:str(x[0])):
                h.update(str(k).encode());add(v)
        elif isinstance(value,(list,tuple)):
            for v in value:add(v)
        elif isinstance(value,(str,int,float,bool,type(None))):
            h.update(repr(value).encode())
    for obj in (GF,GM,PM,SV):add(vars(obj))
    add(GSS.idx);add(NT8['net'].state_dict());add(M.XN.state_dict())
    return h.hexdigest()
