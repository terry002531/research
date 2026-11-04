"""Compare the already generated new spectra with all historical A fit inputs.
No new-seed generation, fit, inference, or model selection is performed.
"""
from pathlib import Path
import sys, json, hashlib
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent
OLD=OUT.parent/'B2_第六类_20261104'
boot=(OLD/'bootstrap_b2.py').read_text(encoding='utf-8')
boot=boot.replace('run_validated.log','overlap_audit.log')
boot=boot.replace('strict_validation_adapter.json','strict_validation_adapter_overlap.json')
exec(compile(boot,str(OLD/'bootstrap_b2.py'),'exec'),globals())
ALLOWED_GEN5.add(11)
assert not (OUT/'extended_overlap.json').exists()
P=AD.H.Parts(AD.RN,AD.BAND)
new=np.load(OUT/'diagnostic_spectra.npz',allow_pickle=False)
old=np.load(OLD/'allowed_data.npz',allow_pickle=False)
S=np.concatenate([new[k] for k in new.files])
def hs(s):return {hashlib.sha256(row.tobytes()).hexdigest() for row in np.round(s,9)}
h=hs(S)
sets={'full_training':old['full_train'],
      'full_fusion_validation':np.concatenate(list(AD.all_fams(31438,40,P,True).values())),
      'SS_tuning_31451_all26':np.concatenate(list(AD.all_fams(31451,30,P,True).values())),
      'A_Gaussian_fit_seed11':AD.D.gen5(AD.A.NTRAIN,11,AD.BAND)[0],
      'A_tuning_five_31439':AD.D.gen5(40,31439,AD.BAND)[0]}
g=np.random.default_rng(11)
sets['parts_network_original_training_pool']=np.concatenate([P.gen(1024*8,g)[0] for _ in range(8)])
counts={k:len(h&hs(s)) for k,s in sets.items()}
assert not any(counts.values()),counts
payload=dict(status='PASS',method='Exact rows after 9-decimal rounding',
    new_seed_spectra_read_from_file=True,new_seed_regenerated=False,
    new_seed_spectra=len(S),reported_six_family_spectra=180,
    compared_counts={k:len(s) for k,s in sets.items()},overlaps=counts,
    test_6161_4242_generated_or_loaded=False,results_81063_loaded=False,
    fitting_or_inference_performed=False)
(OUT/'extended_overlap.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(payload,ensure_ascii=False,indent=2))
