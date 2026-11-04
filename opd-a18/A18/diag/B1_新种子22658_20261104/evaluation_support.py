"""AI (Codex): preregistered paired inference; no model fitting here."""
import csv
import hashlib
import json
import types
import numpy as np
from scipy.stats import t

NBOOT = 200000
BOOT_SEED = 1901122
FAMILY_ALPHA = .05 / 26


def paired(a, b, alpha=FAMILY_ALPHA):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape and a.ndim == 2 and a.shape[1] == 4
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert (a > 0).all() and (b > 0).all()
    x = np.log(a / b).mean(1)
    mean = float(x.mean())
    rng = np.random.default_rng(BOOT_SEED)
    boot = np.empty(NBOOT)
    for s in range(0, NBOOT, 2000):
        ids = rng.integers(0, len(x), (min(2000, NBOOT-s), len(x)))
        boot[s:s+len(ids)] = x[ids].mean(1)
    blo, bhi = np.quantile(boot, [alpha/2, 1-alpha/2])
    margin = t.ppf(1-alpha/2, len(x)-1) * x.std(ddof=1) / np.sqrt(len(x))
    tlo, thi = mean-margin, mean+margin
    lo, hi = min(blo, tlo), max(bhi, thi)
    ratio = float(np.exp(mean))
    return dict(n_spectra=len(x), log_ratio=mean, ratio=ratio,
                arithmetic_ratio=float(a.mean()/b.mean()),
                bootstrap_log_low=float(blo), bootstrap_log_high=float(bhi),
                t_log_low=float(tlo), t_log_high=float(thi),
                conservative_log_low=float(lo), conservative_log_high=float(hi),
                conservative_ratio_low=float(np.exp(lo)), conservative_ratio_high=float(np.exp(hi)),
                A_stable=bool(hi < 0 and ratio <= .95),
                B_stable=bool(lo > 0 and ratio >= 1/.95))


def arrays(samples, family, keys):
    rows = sorted((s for s in samples if s['family'] == family),
                  key=lambda s: (int(s['spectrum']), int(s['draw'])))
    ns = len(rows)//4
    assert len(rows) == ns*4
    assert [(int(s['spectrum']), int(s['draw'])) for s in rows] == [
        (i, d) for i in range(ns) for d in range(4)]
    return {k: np.array([float(s[k]) for s in rows]).reshape(ns, 4) for k in keys}


def annotate(rows, samples):
    for r in rows:
        a = arrays(samples, r['family'], ('A', 'B_raw', 'B_consistent'))
        for b in ('B_raw', 'B_consistent'):
            r.update({f'A_vs_{b}_{k}': v for k, v in paired(a['A'], a[b]).items()})
    return rows


def summarize(rows):
    result = dict(bootstrap_n=NBOOT, bootstrap_seed=BOOT_SEED,
                  unit='spectrum, retaining four paired readings',
                  statistic='mean_spectrum mean_draw log(A/B)',
                  alpha_per_family=FAMILY_ALPHA, correction='Bonferroni 26 families per comparison',
                  interval='envelope of percentile bootstrap and Student t',
                  point_requirement='A/B <= .95 or B/A <= .95')
    for b in ('B_raw', 'B_consistent'):
        for direction in ('A_stable', 'B_stable'):
            result[f'{direction}_vs_{b}'] = [r['family'] for r in rows if r[f'A_vs_{b}_{direction}']]
        result[f'uncertain_vs_{b}'] = [r['family'] for r in rows
            if not r[f'A_vs_{b}_A_stable'] and not r[f'A_vs_{b}_B_stable']]
    return result


def q_statistics(rows, samples):
    """Equal-family spectrum log ratio; stratified paired bootstrap and t CI."""
    xs = []
    for r in rows:
        a = arrays(samples, r['family'], ('q_on', 'q_off'))
        st = paired(a['q_on'], a['q_off'])
        r.update({f'paired_{k}': v for k, v in st.items() if k not in ('A_stable', 'B_stable')})
        xs.append(np.log(a['q_on']/a['q_off']).mean(1))
    x = np.stack(xs)
    nf, ns = x.shape
    rng = np.random.default_rng(BOOT_SEED)
    boot = np.empty(NBOOT)
    for s in range(0, NBOOT, 1000):
        ids = rng.integers(0, ns, (min(1000, NBOOT-s), nf, ns))
        boot[s:s+len(ids)] = x[np.arange(nf)[None, :, None], ids].mean((1, 2))
    lo, hi = np.quantile(boot, [.025, .975])
    means, var = x.mean(1), x.var(1, ddof=1)/ns/nf**2
    se = np.sqrt(var.sum())
    df = var.sum()**2 / (var**2/(ns-1)).sum()
    margin = t.ppf(.975, df)*se
    tlo, thi = means.mean()-margin, means.mean()+margin
    upper = max(hi, thi)
    worse = [r['family'] for r in rows if r['q_on']/r['q_off'] > 1.10]
    return dict(statistic='equal-family mean_spectrum mean_draw log(q_on/q_off)',
                log_ratio=float(means.mean()), ratio=float(np.exp(means.mean())),
                bootstrap_log_ci=[float(lo), float(hi)],
                t_log_ci=[float(tlo), float(thi)], t_df=float(df),
                conservative_log_ci=[float(min(lo,tlo)), float(upper)],
                bootstrap_n=NBOOT, bootstrap_seed=BOOT_SEED,
                overall_alpha=.05, family_alpha=FAMILY_ALPHA,
                family_degradation_definition='ratio of family arithmetic mean NRMSE > 1.10',
                any_family_worse_over_10_percent=worse,
                retain_q=bool(upper < 0 and not worse),
                decision_scope='B3 q use only; development comparison on reused validation spectra')


def csvwrite(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def numeric_state(roots):
    """Content fingerprints of decoder closures and inference globals, including priors."""
    import torch
    out, seen, keepalive = {}, {}, []
    def visit(v, path):
        if v is None or isinstance(v, (str, int, float, bool, np.generic)):
            out[path] = v.item() if isinstance(v, np.generic) else v
            return
        if isinstance(v, (types.ModuleType, type)):
            out[path] = dict(runtime_reference=getattr(v, '__name__', str(v)))
            return
        if id(v) in seen:
            out[path] = dict(reference=seen[id(v)])
            return
        seen[id(v)] = path
        keepalive.append(v)
        if isinstance(v, (np.ndarray, torch.Tensor)):
            a = v.detach().cpu().contiguous().numpy() if isinstance(v, torch.Tensor) else np.ascontiguousarray(v)
            assert a.dtype != object, path
            out[path] = dict(shape=list(a.shape), dtype=str(a.dtype), sha256=hashlib.sha256(a.tobytes()).hexdigest())
        elif isinstance(v, torch.nn.Module):
            for k, a in sorted(v.state_dict().items()): visit(a, path+'.state.'+k)
        elif isinstance(v, dict):
            for k in sorted(v, key=str): visit(v[k], path+'.'+str(k))
        elif isinstance(v, (list, tuple)):
            for i, a in enumerate(v): visit(a, path+'.'+str(i))
        elif isinstance(v, types.FunctionType):
            out[path+'.code'] = dict(module=v.__module__, name=v.__qualname__,
                bytecode_sha256=hashlib.sha256(v.__code__.co_code).hexdigest())
            if v.__closure__:
                for k, c in zip(v.__code__.co_freevars, v.__closure__): visit(c.cell_contents, path+'.closure.'+k)
        elif hasattr(v, '__dict__'):
            out[path+'.class'] = type(v).__module__+'.'+type(v).__qualname__
            for k, a in sorted(vars(v).items()): visit(a, path+'.'+k)
        else:
            out[path] = dict(runtime_reference=str(v))
    visit(roots, 'model')
    raw = json.dumps(out, sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf-8')
    return dict(sha256=hashlib.sha256(raw).hexdigest(), entries=out)
