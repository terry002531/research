"""Fig.6b 示意：d–V 可行域图。

参数全部是“示意值”，只用于检查各边界函数能否画在同一张图上，
不代表任何文献体系。换成目标体系的输入后重新运行即可。
运行：python3 demo_fig6b.py  → 输出 fig6b_demo.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import opd_bounds as b

# ---------------- 示意参数（需替换）
lam_t = 1100e-9                 # 目标波长
k_t, k_vis = 0.05, 0.6          # 目标波长与可见区消光系数
n_eff = 1.9
V0 = 0.6                        # 补偿电压/内建电势
mutau_e, mutau_h = 1e-12, 5e-14 # 有效 μτ（空穴为慢载流子；光从电子接触侧入射，慢载流子走全程）
mu_slow = 1e-7
eps_r, A_dev, R_load = 3.5, 4e-6, 50.0
A_target, eta_c_target = 0.30, 0.9
f_target = 5e5
J_target = 1e-4                 # 10 nA/cm²
dark = dict(J_inj0=1.08e11, phi_inj_eV=1.00, eps_r=eps_r, g_bulk=5e19)
V_noise = 5.0                   # 实测噪声开始恶化的反偏（输入）
photon_flux, eta_gen = 5e19, 0.5  # 约 1 mW/cm²@1100 nm
RR_target = 0.5                 # CCN 判据：EQE(λ_vis)/EQE(λ*) < 0.5

a_t = b.alpha_from_k(k_t, lam_t)
a_v = b.alpha_from_k(k_vis, 600e-9)
d = np.linspace(50e-9, 2000e-9, 240)
V = np.linspace(0, 8, 161)
DD, VV = np.meshgrid(d, V, indexing="ij")
Veff = V0 + VV

# ---------------- 各边界
d_abs = b.d_abs_floor(a_t, A_target)                                   # A1/A5
d_coll = np.array([b.d_collection(np.sqrt(mutau_e * mutau_h), V0 + v, eta_c_target) for v in V])  # K3
d_sc = b.d_space_charge(mu_slow, V0 + V, photon_flux, eta_gen, eps_r)  # K5'
d_tr = b.d_transit_max(mu_slow, V0 + V, f_target)                      # B4
d_rc = b.d_RC_min(R_load, eps_r, A_dev, f_target)                      # B5
d_dk = np.array([b.d_dark(V0 + v, J_target, **dark) for v in V])        # N2
d_res = [b.d_resonance(m, lam_t, n_eff) for m in range(1, 6)]          # A3

# CCN 区：由同一个 Hecht 分布积分直接算谱选择比
RR = np.zeros_like(DD)
for i, di in enumerate(d):
    for j, vj in enumerate(V):
        e_t = b.eta_A_incoherent(a_t, di) * b.hecht_profile(di, a_t, mutau_e, mutau_h, V0 + vj, light_on_hole_side=False, n=120)
        e_v = b.eta_A_incoherent(a_v, di) * b.hecht_profile(di, a_v, mutau_e, mutau_h, V0 + vj, light_on_hole_side=False, n=120)
        RR[i, j] = e_v / e_t
ccn = RR < RR_target

feasible = ((DD >= d_abs)
            & (DD <= d_coll[None, :])
            & (DD <= d_tr[None, :]) & (DD >= d_rc)
            & (DD >= np.nan_to_num(d_dk, nan=np.inf)[None, :])
            & (VV <= V_noise))

fig, ax = plt.subplots(figsize=(7, 5))
ax.contourf(V, d * 1e9, feasible.astype(float), levels=[0.5, 1.5], colors=["#cfe8d5"])
ax.contourf(V, d * 1e9, ccn.astype(float), levels=[0.5, 1.5], colors=["#f3d9c4"], alpha=0.7)
ax.axhline(d_abs * 1e9, color="#1f77b4", label="absorption floor (A1)")
ax.plot(V, d_coll * 1e9, color="#2ca02c", label="collection √(ξ*μτV_eff) (K3)")
ax.plot(V, d_sc * 1e9, "--", color="#2ca02c", label="space-charge (K5')")
ax.plot(V, d_tr * 1e9, color="#9467bd", label="transit limit (B4)")
ax.plot(V, d_dk * 1e9, color="#d62728", label="dark-current thickness (N2)")
ax.axvline(V_noise, color="#d62728", ls=":", label="noise bias limit")
for m, dm in enumerate(d_res, 1):
    ax.axhline(dm * 1e9, color="gray", lw=0.6, ls="-.")
    ax.text(7.9, dm * 1e9 + 8, f"m={m}", fontsize=7, ha="right", color="gray")
ax.set_xlabel("reverse bias V (V)")
ax.set_ylabel("active-layer thickness d (nm)")
ax.set_ylim(d[0] * 1e9, d[-1] * 1e9)
ax.set_title("Fig.6b demo (illustrative parameters) — green: feasible, orange: CCN")
ax.legend(fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig("fig6b_demo.png", dpi=150)
print("d_abs=%.0f nm, d_rc=%.1f nm" % (d_abs * 1e9, d_rc * 1e9))
print("d_coll(0V)=%.0f nm, d_coll(5V)=%.0f nm" % (d_coll[0] * 1e9, d_coll[100] * 1e9))
print("d_tr(5V)=%.0f nm, d_dark(5V)=%.0f nm, d_sc(5V)=%.0f nm" % (d_tr[100] * 1e9, d_dk[100] * 1e9, d_sc[100] * 1e9))
print("feasible cells:", feasible.sum(), " CCN cells:", ccn.sum())
