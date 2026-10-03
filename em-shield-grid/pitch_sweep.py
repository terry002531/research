"""Pitch sweep: anode / grid cathode / anode (sym) vs continuous middle cathode (cont, group 07),
original two-electrode device (orig) and the current bottom-cathode grid design (grid).
655 nm device, real TMM generation profile, grid line (w+20) x 40 nm at 290 nm.
EQE estimate = (1D-model EQE of the original structure, from the 2D CHARGE run) x (collection ratio)."""
from collection_model import efficiency
tau_n, tau_p = 35e-6, 14e-6
EQE_orig = {('800',-0.1):43, ('800',-2.1):60, ('800',-5):63, ('1100',-0.1):36, ('1100',-2.1):49, ('1100',-5):52}
pitches = [0.3, 0.4, 0.5, 0.6, 0.7, 1.0, 1.5, 2.0]
for mun, mup, label in [(1e-7, 2.5e-7, 'calibrated: mu_n=1e-3, mu_p=2.5e-3 cm2/Vs'),
                        (3e-8, 1e-7,   'slower: mu_n=3e-4, mu_p=1e-3 cm2/Vs')]:
    print(f"\n== {label} (effective values, grid line w = 50 nm) ==")
    print("condition       | orig grid300 grid400 | cont | " + " ".join(f"sym{p:>4}" for p in pitches))
    for (k, V), e in EQE_orig.items():
        o = efficiency('orig', 1e-6, V, k, mun, mup, tau_n, tau_p)
        f = lambda s, p=1e-6: e*efficiency(s, p, V, k, mun, mup, tau_n, tau_p, w=50e-9)/o
        row = [f('grid', 0.3e-6), f('grid', 0.4e-6), f('cont')] + [f('sym', p*1e-6) for p in pitches]
        print(f"{k:>4} nm {V:>5} V | {e:4.0f} {row[0]:7.1f} {row[1]:7.1f} | {row[2]:4.1f} | " + " ".join(f"{x:7.1f}" for x in row[3:]))
