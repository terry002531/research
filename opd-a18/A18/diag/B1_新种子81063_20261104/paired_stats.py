"""Preregistered spectrum-paired log-ratio bootstrap, 26-family Bonferroni."""
import numpy as np

NBOOT = 200000
BOOT_SEED = 1901122
ALPHA = 0.05/26

def paired(a, b):
    """a,b: spectra x draws; estimate E_spectrum E_draw log(a/b)."""
    a,b=np.asarray(a),np.asarray(b)
    assert a.shape == b.shape == (30,4)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    assert np.all(a>0) and np.all(b>0)
    log_ratios=np.log(a/b).mean(axis=1)
    rng=np.random.default_rng(BOOT_SEED)
    boot=np.empty(NBOOT)
    for start in range(0,NBOOT,10000):
        ids=rng.integers(0,len(a),(min(10000,NBOOT-start),len(a)))
        boot[start:start+len(ids)]=log_ratios[ids].mean(axis=1)
    low,high=np.quantile(boot,[ALPHA/2,1-ALPHA/2])
    mean=float(log_ratios.mean())
    ratio=float(np.exp(mean))
    return dict(log_ratio=mean,ratio=ratio,improvement_percent=100*(1-ratio),
                log_ci_low=float(low),log_ci_high=float(high),
                ratio_ci_low=float(np.exp(low)),ratio_ci_high=float(np.exp(high)),
                ratio_of_arithmetic_means=float(a.mean()/b.mean()),
                A_stable=bool(high<0 and ratio<=0.95),
                B_stable=bool(low>0 and ratio>=1/0.95))

def add_paired_statistics(rows, samples):
    for row in rows:
        ss=[s for s in samples if s['family']==row['family']]
        arrays={k:np.array([s[k] for s in ss]).reshape(30,4)
                for k in ('A','B_raw','B_consistent')}
        for compare in ('B_raw','B_consistent'):
            for key,value in paired(arrays['A'],arrays[compare]).items():
                row[f'A_vs_{compare}_{key}']=value
        row['bootstrap_n']=NBOOT
        row['bootstrap_seed']=BOOT_SEED
        row['bonferroni_alpha']=ALPHA
        row['bootstrap_unit']='spectrum (4 paired draws)'
    return rows

def summarize_statistics(rows):
    result=dict(bootstrap_n=NBOOT,bootstrap_seed=BOOT_SEED,
        statistic='mean over spectra of mean over draws log(NRMSE_A/NRMSE_B)',
        quantiles=[ALPHA/2,1-ALPHA/2],alpha_per_family=ALPHA,
        correction='Bonferroni 26 families, separately for B_raw and B_consistent',
        point_requirement='exp(mean log ratio) <= 0.95',
        interval_method='paired spectrum percentile bootstrap')
    for b in ('B_raw','B_consistent'):
        for which in ('A_stable','B_stable'):
            result[f'{which}_vs_{b}']=[r['family'] for r in rows if r[f'A_vs_{b}_{which}']]
        result[f'uncertain_vs_{b}']=[r['family'] for r in rows if not r[f'A_vs_{b}_A_stable'] and not r[f'A_vs_{b}_B_stable']]
    return result
