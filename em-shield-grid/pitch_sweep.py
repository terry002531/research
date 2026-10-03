"""Pitch sweep for the proposed anode / grid-cathode / anode structure (transparent grid).
EQE estimate = (1D-model EQE of the original structure) x (collection ratio new/original)."""
from collection_model import efficiency
tau_n, tau_p = 35e-6, 14e-6
EQE_orig = {('800',-0.1):43, ('800',-2.1):60, ('800',-5):63, ('1100',-0.1):36, ('1100',-2.1):49, ('1100',-5):52}
conds = list(EQE_orig)
pitches = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
for mun, label in [(3e-8, 'mu_n=3e-4 cm2/Vs (calibrated)'), (1e-8, 'mu_n=1e-4 cm2/Vs (slower electrons)')]:
    mup = 1e-7
    print(f"\n== {label}, mu_p=1e-3 cm2/Vs ==")
    print("condition    | orig  grid0.5 | " + " ".join(f"sym{p:>4}" for p in pitches))
    for k, V in conds:
        o = efficiency('orig', 1e-6, V, k, mun, mup, tau_n, tau_p)
        g = efficiency('grid', 0.5e-6, V, k, mun, mup, tau_n, tau_p)
        s = [efficiency('sym', p*1e-6, V, k, mun, mup, tau_n, tau_p) for p in pitches]
        e = EQE_orig[(k,V)]
        print(f"{k:>4} nm {V:>5} V | {e:4.0f}  {e*g/o:6.1f} | " + " ".join(f"{e*x/o:7.1f}" for x in s))
