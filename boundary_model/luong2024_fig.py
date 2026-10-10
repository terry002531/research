"""用 Luong et al. 2024（Hoang Mai Luong 等，ACS Energy Lett.；作者稿 arXiv 2309.07370）
的公开数据标定通用边界模型，并画出 Fig.6c 与验证点。

数据全部来自作者稿正文与 SI（页码见 luong2024_说明.md）；拟合得到的参数是“有效参数”，
只在本器件堆栈（glass/ITO/ZnO/PTB7-Th:COTIC-4F/MoOx/Ag，面积 4.7 mm²）、λ=1100 nm 下有意义。
运行：python3 luong2024_fig.py → luong2024_fig6c.png、luong2024_dV.png，并打印拟合参数。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, brentq
import opd_bounds as b

C_BLUE, C_ORANGE, C_AQUA, C_VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
C_INK, C_MUTED, C_GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": C_MUTED, "axes.labelcolor": C_INK,
                     "xtick.color": C_MUTED, "ytick.color": C_MUTED, "axes.grid": True,
                     "grid.color": C_GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 2})

# ============================================================ 原文数据
LAM = 1100e-9
AREA = 4.7e-6                      # 有效面积 4.7 mm²（Experimental）
EPS_R = 3.5                        # 假设值（原文未给）
RES_M = np.array([2, 3, 4])
RES_D = np.array([245, 458, 655]) * 1e-9   # 实测共振厚度 @1100 nm
EQE_IDEAL = {2: 0.61, 4: 0.78}     # FDTD 理想 EQE（无电学损失）
EQE_5V = {2: 0.53, 4: 0.59}        # 实测 EQE @5 V
RC_MEAS = {245e-9: 0.87, 655e-9: 0.84}     # R_c = EQE(0 V)/EQE(5 V) @1100 nm
F3DB = [(245e-9, 0.1, 550e3), (245e-9, 5, 900e3), (655e-9, 0.1, 60e3), (655e-9, 5, 820e3)]
JD = [(245e-9, 0.1, 5.5), (245e-9, 5, 456.0), (655e-9, 0.1, 3.0), (655e-9, 5, 38.2)]  # nA/cm²
JD_THRESH_5V = (300e-9, 350e-9)    # 5 V 下暗电流趋平的厚度区间（Fig. S8b）
ENH_280 = 5.0                      # 280 nm BHJ 在器件中比独立薄膜吸收高 5 倍（Fig.1b）
R_BACK_GUESS = None                # 两镜反射率都由拟合得到

nA = 1e-5                          # 1 nA/cm² = 1e-5 A/m²

# ============================================================ 1 光学：共振位置 → n_eff, φ
slope, icpt = np.polyfit(RES_M, RES_D, 1)
N_EFF = LAM / (2 * slope)
PHI = -icpt / slope * 2 * np.pi
d_m = lambda m: b.d_resonance(m, LAM, N_EFF, PHI)

# 2 光学：α 由 5 倍增强约束，R_f、R_b 由两个理想 EQE 峰拟合
ALPHA = -np.log(1 - EQE_IDEAL[2] / ENH_280) / 280e-9
sig = lambda z: 1 / (1 + np.exp(-z))


def A_peak(Rf, Rb, d):
    x = np.exp(-ALPHA * d)
    return (1 - Rf) * (1 - x) * (1 + Rb * x) / (1 - np.sqrt(Rf * Rb) * x) ** 2


fit = least_squares(lambda p: [A_peak(sig(p[0]), sig(p[1]), d_m(2)) - EQE_IDEAL[2],
                               A_peak(sig(p[0]), sig(p[1]), d_m(4)) - EQE_IDEAL[4]], [0, 1])
RF, RB = sig(fit.x[0]), sig(fit.x[1])
A_of_d = lambda d: b.eta_A_cavity(ALPHA, d, LAM, N_EFF, RF, RB, PHI)

# ============================================================ 3 带宽：μ_s, V0, R
def f_model(p, d, V):
    mu, V0, R = np.exp(p)
    return b.f_3dB(b.f_RC(R, EPS_R, AREA, d), b.f_transit(mu, V + V0, d))


best = None
for g in [(1e-8, 0.3, 50), (1e-7, 0.3, 200), (1e-7, 0.05, 500), (3e-8, 1, 100)]:
    s = least_squares(lambda p: [np.log(f_model(p, d, V) / f) for d, V, f in F3DB], np.log(g))
    best = s if best is None or s.cost < best.cost else best
MU, V0, R_EFF = np.exp(best.x)

# ============================================================ 4 收集：每个厚度各自的 μτ_eff
def Rc(mt, d):
    return b.hecht_uniform(d, mt, mt, V0) / b.hecht_uniform(d, mt, mt, V0 + 5)


MUTAU = {d: 10 ** brentq(lambda L: Rc(10 ** L, d) - r, -17, -10) for d, r in RC_MEAS.items()}

# ============================================================ 5 暗电流：注入项（经验 β）+ 平台项
J_FLOOR = {0.1: 3.0 * nA, 5: 38.2 * nA}     # 取 655 nm 器件为平台值


def j_floor(V):
    """平台项：在 0.1 V 与 5 V 两点间按 V_eff 做对数线性插值（中间偏压为外推）。"""
    v1, v2 = V0 + 0.1, V0 + 5
    t = (V0 + np.asarray(V) - v1) / (v2 - v1)
    return J_FLOOR[0.1] * (J_FLOOR[5] / J_FLOOR[0.1]) ** t


d_thr = np.mean(JD_THRESH_5V)
F1, F2 = np.sqrt((V0 + 5) / 245e-9), np.sqrt((V0 + 5) / d_thr)
J_inj_245 = 456 * nA - J_FLOOR[5]
BETA = np.log(J_inj_245 / J_FLOOR[5]) / (F1 - F2)        # 阈值处注入 = 平台
J_A = J_inj_245 / np.exp(BETA * F1)


def j_dark(d, V):
    return J_A * np.exp(BETA * np.sqrt((V0 + V) / d)) + j_floor(V)


# ============================================================ 打印参数
beta_schottky = np.sqrt(b.q / (4 * np.pi * b.eps0 * EPS_R)) / b.kT()
print("== 光学 ==")
print(f"共振间距 {slope*1e9:.0f} nm → n_eff = {N_EFF:.2f}，φ/2π = {PHI/2/np.pi:.2f}，m=1 预测 {d_m(1)*1e9:.0f} nm")
print(f"α(1100) = {ALPHA/100:.0f} cm⁻¹ (k = {ALPHA*LAM/4/np.pi:.3f})，R_f = {RF:.2f}，R_b = {RB:.2f}")
print("== 带宽 ==")
print(f"μ_s = {MU*1e4:.2e} cm²/Vs，V0 = {V0:.2f} V，R_eff = {R_EFF:.0f} Ω")
for d, V, f in F3DB:
    print(f"  d={d*1e9:.0f} V={V}: 实测 {f/1e3:.0f} kHz，模型 {f_model(best.x, d, V)/1e3:.0f} kHz "
          f"(f_RC {b.f_RC(R_EFF, EPS_R, AREA, d)/1e3:.0f}, f_tr {b.f_transit(MU, V+V0, d)/1e3:.0f})")
print("== 收集 ==")
for d, mt in MUTAU.items():
    print(f"  d={d*1e9:.0f}: μτ_eff = {mt:.2e} m²/V（τ_eff = {mt/MU*1e6:.1f} μs）")
print(f"  用 245 nm 的 μτ 预测 655 nm 的 R_c = {Rc(MUTAU[245e-9], 655e-9):.2f}（实测 0.84）")
print("== 暗电流 ==")
print(f"  β = {BETA:.2e} (m/V)^½，Schottky 理论 {beta_schottky:.2e}，比值 {BETA/beta_schottky:.1f}")
for d, V, j in JD:
    print(f"  d={d*1e9:.0f} V={V}: 实测 {j} nA/cm²，模型 {j_dark(d, V)/nA:.1f}")

# ============================================================ 图 6c：四联图
fig, ax = plt.subplots(2, 2, figsize=(9.2, 7.2))

# (a) 光学
a = ax[0, 0]
dd = np.linspace(20e-9, 900e-9, 900)
a.plot(dd * 1e9, A_of_d(dd), color=C_BLUE, label="calibrated cavity model η_A")
for m in RES_M:
    a.axvline(d_m(m) * 1e9, color=C_MUTED, lw=0.6, ls=":")
    a.text(d_m(m) * 1e9 + 6, 0.93, f"m={m}", color=C_MUTED, fontsize=8)
a.scatter([245, 655], [EQE_IDEAL[2], EQE_IDEAL[4]], s=60, facecolor="white", edgecolor=C_BLUE, lw=2, zorder=3, label="FDTD ideal EQE (paper)")
a.scatter([245, 655], [EQE_5V[2], EQE_5V[4]], s=60, color=C_ORANGE, edgecolor="white", lw=1.5, zorder=4, label="measured EQE @5 V")
a.set(xlabel="BHJ thickness d (nm)", ylabel="absorption / EQE at 1100 nm", ylim=(0, 1),
      title="(a) Resonance: absorption vs thickness")
a.legend(fontsize=7.5, loc="upper right", bbox_to_anchor=(1, 0.9), frameon=False)

# (b) 带宽
a = ax[0, 1]
VV = np.linspace(0, 6, 200)
for d, col, lab in [(245e-9, C_BLUE, "245 nm"), (655e-9, C_ORANGE, "655 nm")]:
    a.plot(VV, f_model(best.x, d, VV) / 1e3, color=col, label=f"{lab} model")
    a.axhline(b.f_RC(R_EFF, EPS_R, AREA, d) / 1e3, color=col, lw=1, ls="--")
    pts = [(V, f) for dd_, V, f in F3DB if dd_ == d]
    a.scatter([p[0] for p in pts], [p[1] / 1e3 for p in pts], s=60, color=col, edgecolor="white", lw=1.5, zorder=3)
a.text(5.9, b.f_RC(R_EFF, EPS_R, AREA, 245e-9) / 1e3 * 1.08, "RC limit 245 nm", ha="right", fontsize=7.5, color=C_MUTED)
a.text(5.9, b.f_RC(R_EFF, EPS_R, AREA, 655e-9) / 1e3 * 1.08, "RC limit 655 nm", ha="right", fontsize=7.5, color=C_MUTED)
a.set(yscale="log", xlabel="reverse bias V (V)", ylabel="f₃dB (kHz)", ylim=(20, 5000),
      title="(b) Bandwidth: RC vs transit")
a.legend(fontsize=7.5, loc="lower right", frameon=False)

# (c) 暗电流
a = ax[1, 0]
dd = np.linspace(150e-9, 900e-9, 300)
for V, col in [(5, C_ORANGE), (0.1, C_BLUE)]:
    a.plot(dd * 1e9, j_dark(dd, V) / nA, color=col, label=f"model {V} V")
    pts = [(d, j) for d, V_, j in JD if V_ == V]
    a.scatter([p[0] * 1e9 for p in pts], [p[1] for p in pts], s=60, color=col, edgecolor="white", lw=1.5, zorder=3)
a.axvspan(*(np.array(JD_THRESH_5V) * 1e9), color=C_ORANGE, alpha=0.12, lw=0)
a.text(325, 1.6e3, "paper: plateau\nonset @5 V", ha="center", fontsize=7.5, color=C_MUTED)
a.set(yscale="log", xlabel="BHJ thickness d (nm)", ylabel="J_d (nA cm⁻²)", ylim=(1, 5e3),
      title="(c) Dark current: injection + plateau")
a.legend(fontsize=7.5, loc="upper right", frameon=False)

# (d) 收集
a = ax[1, 1]
VV = np.linspace(0, 6, 200)
for d, col in [(245e-9, C_BLUE), (655e-9, C_ORANGE)]:
    mt = MUTAU[d]
    eta = b.hecht_uniform(d, mt, mt, V0 + VV) / b.hecht_uniform(d, mt, mt, V0 + 5)
    a.plot(VV, eta, color=col, label=f"{d*1e9:.0f} nm, own μτ={mt:.1e}")
    a.scatter([0], [RC_MEAS[d]], s=60, color=col, edgecolor="white", lw=1.5, zorder=3)
mt = MUTAU[245e-9]
eta = b.hecht_uniform(655e-9, mt, mt, V0 + VV) / b.hecht_uniform(655e-9, mt, mt, V0 + 5)
a.plot(VV, eta, color=C_ORANGE, ls="--", lw=1.5, label="655 nm with 245-nm μτ")
a.set(xlabel="reverse bias V (V)", ylabel="EQE(V) / EQE(5 V) at 1100 nm", ylim=(0.3, 1.05),
      title="(d) Collection: one μτ cannot fit both")
a.legend(fontsize=7.5, loc="lower right", frameon=False)

fig.suptitle("Luong 2024 (PTB7-Th:COTIC-4F, 1100 nm) — general boundary model calibrated to published data",
             fontsize=10, color=C_INK)
fig.tight_layout()
fig.savefig("luong2024_fig6c.png", dpi=170)

# ============================================================ d–V 可行域（Fig.6b 中的 Luong 验证点）
A_TARGET = 0.55       # 吸收目标（对应 EQE ≈ 50%）
ETA_C = 0.85          # 收集目标（相对 5 V）
F_TARGET = 500e3      # 带宽目标
DSTAR = 1e13          # 散粒噪声 D* 目标（Jones）
EQE_FOR_D = 0.5
R_resp = b.responsivity(EQE_FOR_D, LAM)
J_TARGET = (R_resp / (DSTAR * 1e-2)) ** 2 / (2 * b.q)   # D* 用 cm 单位换算

d = np.linspace(100e-9, 900e-9, 400)
V = np.linspace(0, 6, 241)
D, VG = np.meshgrid(d, V, indexing="ij")
ok_abs = A_of_d(D) >= A_TARGET
ok_f = f_model(best.x, D, VG) >= F_TARGET
ok_j = j_dark(D, VG) <= J_TARGET
dc = {k: b.d_collection(mt, V0 + V, ETA_C) for k, mt in MUTAU.items()}
ok_c_245 = D <= dc[245e-9][None, :]
ok_c_655 = D <= dc[655e-9][None, :]
feas_lo = ok_abs & ok_f & ok_j & ok_c_245
feas_hi = ok_abs & ok_f & ok_j & ok_c_655

fig, a = plt.subplots(figsize=(7.4, 5.4))
a.contourf(V, d * 1e9, feas_lo.astype(float), levels=[0.5, 1.5], colors=[C_AQUA], alpha=0.6)
a.contourf(V, d * 1e9, ok_abs.astype(float), levels=[0.5, 1.5], colors=[C_BLUE], alpha=0.10)
a.contour(V, d * 1e9, f_model(best.x, D, VG), levels=[F_TARGET], colors=[C_VIOLET], linewidths=2)
a.contour(V, d * 1e9, j_dark(D, VG), levels=[J_TARGET], colors=[C_ORANGE], linewidths=2)
a.plot(V, dc[245e-9] * 1e9, color=C_INK, lw=1.5)
a.plot(V, dc[655e-9] * 1e9, color=C_INK, lw=1.5, ls="--")
for dd_, VV_, f in F3DB:
    jd = [j for d2, V2, j in JD if d2 == dd_ and V2 == VV_][0]
    a.scatter(VV_, dd_ * 1e9, s=70, color="white", edgecolor=C_INK, lw=1.8, zorder=5)
    a.annotate(f"{f/1e3:.0f} kHz\n{jd:g} nA/cm²", (VV_, dd_ * 1e9), xytext=(8, 6),
               textcoords="offset points", fontsize=7.5, color=C_INK)
a.text(1.75, 700, "collection η≥0.85\n(μτ from 245 nm)", fontsize=7.5, ha="right")
a.text(0.22, 860, "same, μτ from 655 nm", fontsize=7.5, color=C_MUTED)
a.text(0.15, 140, f"blue bands: η_A ≥ {A_TARGET}", fontsize=7.5, color=C_BLUE)
a.text(3.0, 320, f"f₃dB = {F_TARGET/1e3:.0f} kHz", fontsize=7.5, color=C_VIOLET)
a.text(0.45, 600, f"J_d = {J_TARGET/nA:.1f} nA/cm²\n(D*_shot = 1e13 Jones)", fontsize=7.5, color=C_ORANGE)
a.set(xlabel="reverse bias V (V)", ylabel="BHJ thickness d (nm)", xlim=(0, 6), ylim=(100, 900),
      title="Luong 2024 on the d–V map: feasible window (aqua) and the four measured devices")
fig.tight_layout()
fig.savefig("luong2024_dV.png", dpi=170)
print(f"J_target = {J_TARGET/nA:.2f} nA/cm²；可行格点（245 μτ / 655 μτ）: {feas_lo.sum()} / {feas_hi.sum()}")
for lo, hi in [(0, 0.5), (0.5, 2), (2, 6)]:
    sel = (V >= lo) & (V < hi)
    rows = np.nonzero(feas_hi[:, sel].any(axis=1))[0]
    if rows.size:
        print(f"  V∈[{lo},{hi}): 可行厚度 {d[rows.min()]*1e9:.0f}–{d[rows.max()]*1e9:.0f} nm（含不连续段）")
    else:
        print(f"  V∈[{lo},{hi}): 无可行点")
