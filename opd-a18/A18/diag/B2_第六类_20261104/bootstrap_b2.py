"""A18-B2, AI (Codex). Read-only source execution; all writes stay beside this file.
Run: py -I -X utf8 A18/diag/B2_诊断_20261104/diagnose_b1.py
No B training and no project/author test-spectrum or test-current evaluation.
"""
from pathlib import Path
import ast, sys, os, json, time, hashlib, importlib.abc, importlib.util, csv

OUT = Path(__file__).resolve().parent
WORK = OUT.parents[2]
ALG = WORK / '算法对照'
PROJ = WORK.parent
sys.dont_write_bytecode = True
for key in ('TMP', 'TEMP', 'MPLCONFIGDIR', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR'):
    p = OUT / 'runtime' / key
    p.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(p)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def fingerprint():
    files = {}
    for p in WORK.rglob('*'):
        if not p.is_file() or p.is_relative_to(OUT):
            continue
        st = p.stat()
        files[str(p.relative_to(WORK))] = dict(size=st.st_size, mtime_ns=st.st_mtime_ns)
    return files

if not (OUT / 'existing_files_before.json').exists():
    dump('existing_files_before.json', fingerprint())

# Fail closed on writes outside the diagnostic directory, including imported modules.
def audit(event, args):
    if event == 'open':
        path, mode, flags = args
        if not isinstance(path, (str, bytes, os.PathLike)):
            return
        if os.fsdecode(path).lower() in ('nul', os.devnull.lower()):
            return  # Windows null device; no filesystem artifact.
        p = Path(os.fsdecode(path)).absolute()
        write = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)) if isinstance(flags, int) else False
        if write and not p.is_relative_to(OUT):
            raise PermissionError(f'B2 write outside diagnostic directory: {p}')
        norm = str(p).replace('\\', '/').lower()
        if not write and any(x in norm for x in ('a18_seed6161', 'a18_seed4242', 'testset_', 'peakfix.json', 'qnet.pt')) and not p.is_relative_to(OUT):
            raise PermissionError(f'B2 forbidden test-result read: {p}')
    elif event in ('os.remove', 'os.rename', 'os.rmdir'):
        ps = args[:2] if event == 'os.rename' else args[:1]
        if any(not Path(p).absolute().is_relative_to(OUT) for p in ps):
            raise PermissionError(f'B2 disallows destructive operation outside diagnostic directory: {event}')
    elif event == 'os.mkdir':
        p = Path(args[0]).absolute()
        if not p.is_relative_to(OUT) and not p.is_dir():
            raise PermissionError(f'B2 directory outside diagnostic directory: {p}')

sys.addaudithook(audit)
import numpy as np
import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(2)
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
assert torch.cuda.is_available(), 'CPU runtime exceeds the estimated budget; stop.'

# Only in-memory initialization changes: keep original numerical functions unchanged.
# Remove author test-current loading and use the previously fixed simulator power.
P0_FIXED = 1.0542339586846165e-05
edits = []
class Sanitize(ast.NodeTransformer):
    def __init__(self, name): self.name = name
    def visit_Expr(self, node):
        if self.name == 'a11_device' and ast.unparse(node) == "sys.modules.pop('paper_check')":
            return ast.copy_location(ast.Pass(), node)
        return self.generic_visit(node)
    def visit_Import(self, node):
        if self.name == 'a11_device' and any(a.name == 'paper_check' for a in node.names):
            edits.append([self.name, node.lineno, 'skip unused author test-check import'])
            return ast.copy_location(ast.Pass(), node)
        return node
    def visit_Assign(self, node):
        names = [x.id for x in node.targets if isinstance(x, ast.Name)]
        replacement = None
        if self.name == 'a11_device':
            if any(k in names for k in ('_pc', 'P_FIT')):
                return ast.copy_location(ast.Pass(), node)
            if 'P0' in names and node.lineno < 110: replacement = ast.Constant(P0_FIXED)
            if any(k in names for k in ('Y_REAL', 'LAM_TEST', 'TEST_ST', 'REC')): replacement = ast.Constant(None)
        if self.name in ('a11_peakfix', 'a15_three') and any(k in names for k in ('OUT',)):
            replacement = ast.Constant(str(OUT / 'runtime'))
        if self.name == 'a15_three' and any(isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name) and x.value.id == 'PF' and x.attr == 'OUT' for x in node.targets):
            replacement = ast.Constant(str(OUT / 'runtime'))
        if self.name == 'forward' and isinstance(node.targets[0], ast.Tuple):
            attrs = [x.attr for x in node.targets[0].elts if isinstance(x, ast.Attribute)]
            if attrs == ['lam_test', 'I_test']: replacement = ast.Tuple(elts=[ast.Constant(None), ast.Constant(None)], ctx=ast.Load())
        if replacement is not None:
            edits.append([self.name, node.lineno, ast.unparse(node.targets[0]) + ' initialization only'])
            node.value = replacement
        return self.generic_visit(node)

paths = {k: ALG / (k + '.py') for k in ('a15_three', 'a11_peakfix', 'a11_device')}
paths['forward'] = ALG / '45_Uddin2024_真实R矩阵/scripts/forward.py'
class Loader(importlib.abc.Loader):
    def __init__(self, name): self.name = name
    def create_module(self, spec): return None
    def exec_module(self, mod):
        p = paths[self.name]
        mod.__file__ = str(p)
        tree = Sanitize(self.name).visit(ast.parse(p.read_text(encoding='utf-8'), filename=str(p)))
        ast.fix_missing_locations(tree)
        exec(compile(tree, str(p), 'exec'), mod.__dict__)
class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == 'forward' and Path(sys.path[0]).resolve() != paths['forward'].parent.resolve():
            return None
        if name in paths: return importlib.util.spec_from_loader(name, Loader(name))
sys.meta_path.insert(0, Finder())
sys.path.insert(0, str(ALG))
sys.argv = ['a15_three.py', '--third', 'cnet0', '--no-b', '--a16a', '32', '--a16a-ss', '--a16a-vscale', '--a16a-escale', '--test-seed', '31438']
import a15_three as M
import a16_ss
AD, PF, P18, BM = M.AD, M.PF, M.P18, M.BM
assert P18.VSEED == 31438 and M.ARG.third == 'cnet0' and M.A18
assert AD.Y_REAL is None and AD.LAM_TEST is None
assert AD.DV.I_test is None and AD.DV.lam_test is None

start = time.time()
def log(s):
    msg = f'[{time.time()-start:.1f}s] {s}'
    print(msg, flush=True)
    with (OUT / 'run_validated.log').open('a', encoding='utf-8') as f: f.write(msg+'\n')
M.log = PF.log = log
original_fams = AD.all_fams
def safe_fams(seed, n, P, jitter):
    assert seed in (31337, 31438, 31451), seed
    assert jitter, 'Fixed CIE spectra would overlap test spectra.'
    return original_fams(seed, n, P, jitter)
AD.all_fams = safe_fams
# Legacy build3 uses a separate seed-3 tuning validation set. Replace only that
# data source with the EXACT five-family member of VSEED=31438 (40 spectra).
# Keep noise seed 1004 and the existing parameter-selection algorithm.
old_seeds = P18.SEEDS
P18.SEEDS = [(31439, old_seeds[0][1], old_seeds[0][2])]
gen5_original = AD.D.gen5
def gen5_allowed(n, seed, band):
    if n == 60 and seed == 31439:
        return gen5_original(AD.NVF, 31439, band)
    assert seed != 3, 'Legacy seed-3 validation is forbidden in this task.'
    return gen5_original(n, seed, band)
AD.D.gen5 = gen5_allowed
# Preserve the existing A18 ParamE hyperparameter read from the allowed historical
# diagnostic logs, rather than tune a different A while diagnosing the current B.
AD.A.SIG_E[:] = [0.08]
dump('strict_validation_adapter.json', dict(legacy_seed_excluded=3,source='VSEED=31438 same-distribution five-family member',
     actual_gen5_seed=31439,actual_gen5_n=AD.NVF,fixed_sigma_e=0.08,
     fixed_sigma_source='A18/diag/scale_ev.out and fuse_diag2.out',
     reason='Use only explicitly allowed validation spectra and retain existing ParamE sigma; not an old-run bitwise reproduction'))


# B2 does not build A or tune ParamE; retain B1 adapter metadata for provenance.
ALLOWED_GEN5 = {31338,31344,31439,31452}
def gen5_strict(n, seed, band, *args, **kwargs):
    assert seed in ALLOWED_GEN5, f"Forbidden spectral seed: {seed}"
    return gen5_original(n, seed, band, *args, **kwargs)
AD.D.gen5 = gen5_strict
TARGETS = ['阶梯谱','零件组合','随机光滑','同分布五类']
CONTROLS = ['单峰 60–200 nm','宽带通']
FAMILIES = TARGETS + CONTROLS
def csvwrite(name, rows):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
