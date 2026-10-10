"""用其他文献做独立检验（不用 Luong 2024，避免循环验证）。

原则：
1. 预测只用文献中“独立测得”的输入（迁移率、电容、负载电阻、厚度、面积），
   不用被预测的那个量去拟合任何参数；
2. 文献没给的量（内建电势 V0、寿命 τ、吸收系数 α）用先验区间，输出预测区间而不是单点；
3. 公式按《42种策略_通用边界模型.md》预先固定：渡越 B1 用 κ=3.5、全厚度 d；
   “平均渡越距离 d/2” 变体只作诊断，不作为通过依据。

文献：
- A14：Armin et al., Laser Photonics Rev. 8, 924 (2014)，Swansea 机构库作者稿 cronfa38600
- N24：CH17 OPD, Natl. Sci. Rev. 11, nwad311 (2024)，PMC10833469
- Y19：Yazmaciyan, Meredith, Armin, Adv. Opt. Mater. 1801543 (2019)，Swansea 机构库作者稿 cronfa48973
运行：python3 external_tests.py → 打印结果，输出 external_tests.png、external_ccn.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import opd_bounds as b

C_BLUE, C_ORANGE, C_AQUA, C_INK, C_MUTED, C_GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
TOL = 2.0
results = []   # (标签, 预测下限, 预测上限, 实测, 说明)


def f3db_from_C(mu, Veff, d, R, C, kappa=3.5, d_eff=None):
    """带宽：RC 用实测电容，渡越用 B1。d_eff 为渡越距离（默认 = d）。"""
    de = d if d_eff is None else d_eff
    f_tr = kappa * mu * Veff / (2 * np.pi * de**2)
    f_rc = 1 / (2 * np.pi * R * C)
    return b.f_3dB(f_rc, f_tr)


# ============================================================ A14 带宽（700 nm，C60 阻挡层）
# 独立输入：μh = 3e-4 cm²/Vs（photo-CELIV/双注入，慢载流子）、C = 3.7 nF/cm²、A = 0.2 cm²、R = 100 Ω
# 先验：V0 ∈ [0.8, 1.0] V（PCDTBT:PC70BM 的 Voc 约 0.88 V；原文未给 V_bi）
A14 = dict(mu=3e-8, d=700e-9, C=3.7e-9 / 1e-4 * 0.2e-4, R=100.0)
for V, f_meas in [(0.0, 120e3), (5.0, 1.0e6)]:
    lo = f3db_from_C(A14["mu"], 0.8 + V, A14["d"], A14["R"], A14["C"])
    hi = f3db_from_C(A14["mu"], 1.0 + V, A14["d"], A14["R"], A14["C"])
    results.append((f"A14 f3dB 700 nm @{V:g} V", lo, hi, f_meas, "μh 来自 photo-CELIV/DI"))
    lo2 = f3db_from_C(A14["mu"], 0.8 + V, A14["d"], A14["R"], A14["C"], d_eff=A14["d"] / 2)
    hi2 = f3db_from_C(A14["mu"], 1.0 + V, A14["d"], A14["R"], A14["C"], d_eff=A14["d"] / 2)
    results.append((f"  diag: same, d_eff = d/2", lo2, hi2, f_meas, "诊断"))

# ============================================================ N24 带宽（CH17，95 nm，0 V）
# 独立输入：SCLC μ = 4.5e-4 cm²/Vs（e/h = 1.2，取较慢者 μ/1.2 到 μ），C = 0.7 nF（0.04 cm²，10 kHz），R = 50 Ω
# 先验：V0 ∈ [0.7, 0.9] V。实测给的是上升时间 t_r，换算 f3dB = 0.35/t_r（单极点）
N24_mu = (4.5e-8 / 1.2, 4.5e-8)
for area, C, tr in [(0.04, 0.7e-9, 168e-9), (0.01, 0.7e-9 / 4, 91e-9)]:
    vals = [f3db_from_C(mu, v0, 95e-9, 50.0, C) for mu in N24_mu for v0 in (0.7, 0.9)]
    results.append((f"N24 f3dB 95 nm {area} cm² @0 V", min(vals), max(vals), 0.35 / tr, "t_r→f3dB"))
    vals2 = [f3db_from_C(mu, v0, 95e-9, 50.0, C, d_eff=95e-9 / 2) for mu in N24_mu for v0 in (0.7, 0.9)]
    results.append(("  diag: same, d_eff = d/2", min(vals2), max(vals2), 0.35 / tr, "诊断"))
# 变温：373 K 时电子迁移率 1e-3 cm²/Vs（原文 Fig. S13），实测 f3dB ≈ 4 MHz（面积按 0.04 cm²）
vals = [f3db_from_C(mu, v0, 95e-9, 50.0, 0.7e-9) for mu in (1e-7 / 1.2, 1e-7) for v0 in (0.7, 0.9)]
results.append(("N24 f3dB 95 nm 0.04 cm² @373 K", min(vals), max(vals), 4e6, "面积为推断"))
# Y6 对照器件：只有 C = 1.9 nF，厚度未知 → 只能给 RC 上界
f_rc_y6 = 1 / (2 * np.pi * 50 * 1.9e-9)
results.append(("N24 Y6 device: RC upper bound only", 0.0, f_rc_y6, 0.35 / 386e-9, "上界检验"))

# ============================================================ 与 κ、d_eff 约定无关的标度检验
# A14：渡越主导时 f(5 V)/f(0 V) = (V0+5)/V0（RC 修正后略小）
lo = f3db_from_C(A14["mu"], 1.0 + 5, A14["d"], A14["R"], A14["C"]) / f3db_from_C(A14["mu"], 1.0, A14["d"], A14["R"], A14["C"])
hi = f3db_from_C(A14["mu"], 0.8 + 5, A14["d"], A14["R"], A14["C"]) / f3db_from_C(A14["mu"], 0.8, A14["d"], A14["R"], A14["C"])
results.append(("A14 ratio f3dB(5 V)/f3dB(0 V)", lo, hi, 1.0e6 / 120e3, "与 κ 无关"))
# N24：面积 0.04→0.01 cm² 的带宽比（实测 = t_r 比 168/91）
r = [f3db_from_C(mu, v0, 95e-9, 50.0, 0.7e-9 / 4) / f3db_from_C(mu, v0, 95e-9, 50.0, 0.7e-9) for mu in N24_mu for v0 in (0.7, 0.9)]
results.append(("N24 ratio f3dB(0.01)/f3dB(0.04 cm²)", min(r), max(r), 168 / 91, "与 κ 无关"))
r2 = [f3db_from_C(mu, v0, 95e-9, 50.0, 0.7e-9 / 4, d_eff=95e-9 / 2) / f3db_from_C(mu, v0, 95e-9, 50.0, 0.7e-9, d_eff=95e-9 / 2) for mu in N24_mu for v0 in (0.7, 0.9)]
results.append(("  diag: same, d_eff = d/2", min(r2), max(r2), 168 / 91, "诊断"))

# ============================================================ A14 暗电流：厚度留一法（Table 1）
# 每个偏压、每个系列：用其余 3 个厚度精确拟合 (J_A, β, J_floor)，预测被留出的中间厚度
A14_JD = {  # thickness: {V: (无 C60, 有 C60)}  [nA/cm²]
    100: {0.2: (160, 60), 0.5: (50, 6), 1.0: (315, 17)},
    200: {0.2: (5.8, 1.8), 0.5: (17.0, 6.5), 1.0: (100, 30)},
    300: {0.2: (2.0, 0.7), 0.5: (7.0, 1.8), 1.0: (55, 4)},
    700: {0.2: (1.35, 0.16), 0.5: (3.00, 0.42), 1.0: (6.6, 0.8)},
}
V0_A14 = 0.9


def fit3(ds, Js, V):
    """J = J_A·exp(β√(V_eff/d)) + J_floor，三点精确求解（对 β 一维求根）。"""
    x = np.sqrt((V0_A14 + V) / np.asarray(ds))
    Js = np.asarray(Js, float)
    o = np.argsort(x)          # x 从小到大 = 厚到薄
    x, Js = x[o], Js[o]

    def resid(beta):
        e = np.exp(beta * x)
        JA = (Js[2] - Js[0]) / (e[2] - e[0])
        Jf = Js[0] - JA * e[0]
        return JA * e[1] + Jf - Js[1], JA, Jf

    betas = np.geomspace(1e-7, 60 / x.max(), 4000)   # 上限避免 exp 溢出
    r = np.array([resid(bb)[0] for bb in betas])
    idx = np.nonzero(np.sign(r[:-1]) != np.sign(r[1:]))[0]
    if idx.size == 0:
        return None
    from scipy.optimize import brentq
    beta = brentq(lambda bb: resid(bb)[0], betas[idx[0]], betas[idx[0] + 1])
    _, JA, Jf = resid(beta)
    return beta, JA, Jf


jd_rows = []
for series, si in (("no C60", 0), ("C60", 1)):
    for V in (0.2, 0.5, 1.0):
        for leave in (200, 300):
            train = [t for t in (100, 200, 300, 700) if t != leave]
            p = fit3([t * 1e-9 for t in train], [A14_JD[t][V][si] for t in train], V)
            meas = A14_JD[leave][V][si]
            if p is None or p[2] < 0 or p[1] < 0:
                jd_rows.append((series, V, leave, np.nan, meas, "需要负的平台项或负的 J_A"))
                results.append((f"A14 J_d {leave} nm @-{V} V {series} (LOO)", np.nan, np.nan, meas, "形式失败"))
                continue
            beta, JA, Jf = p
            pred = JA * np.exp(beta * np.sqrt((V0_A14 + V) / (leave * 1e-9))) + Jf
            jd_rows.append((series, V, leave, pred, meas, ""))
            results.append((f"A14 J_d {leave} nm @-{V} V {series} (LOO)", pred, pred, meas, ""))

# ============================================================ 打印
print(f"{'检验':44s} {'预测下限':>10s} {'预测上限':>10s} {'实测':>10s}  结论")
for lab, lo, hi, m, note in results:
    if np.isnan(lo):
        continue
    if lab.startswith("  diag"):
        verdict = "诊断"
    elif "upper bound" in lab:
        verdict = "通过" if m <= hi else "不通过"
    else:
        tol = 1.25 if lab.startswith(("A14 ratio", "N24 ratio")) else TOL
        verdict = "通过" if (lo / tol <= m <= hi * tol) else "不通过"
        inside = lo <= m <= hi
        verdict += "（落在区间内）" if inside else ""
    print(f"{lab:44s} {lo:10.3g} {hi:10.3g} {m:10.3g}  {verdict}  {note}")
for s, V, t, p, m, n in jd_rows:
    if n:
        print(f"A14 J_d {t} nm @-{V} V {s}: 三点精确拟合不存在物理解（{n}），实测 {m}；记为模型形式失败")

# ============================================================ Y19：CCN 分类检验
# 观测：常规结构 100、700 nm 宽带；常规 2600 nm 窄带（CCN）；反转 700 nm 窄带（CCN）
# 输入：μe = 1e-3、μh = 1e-5 cm²/Vs（Y19 引用值）或 μh = 3e-4（A14 实测）；V0 = 0.9 V，0 V 偏压
# α 先验：吸收峰（~450 nm）1e5 cm⁻¹，吸收边（~650 nm）5e3 cm⁻¹；τ（μτ 中的寿命）为扫描量
# 判据（预先固定）：谱选择比 RR = EQE(峰)/EQE(边)；RR ≥ 1 记为宽带，RR < 0.5 记为 CCN
a_pk, a_edge = 1e7, 5e5   # m⁻¹
V0 = 0.9
cases = [("conv 100 nm", 100e-9, True, "broad"), ("conv 700 nm", 700e-9, True, "broad"),
         ("conv 2600 nm", 2600e-9, True, "CCN"), ("inverted 700 nm", 700e-9, False, "CCN")]
taus = np.geomspace(1e-9, 1e-3, 241)


def classify(RR):
    return "broad" if RR >= 1 else ("CCN" if RR < 0.5 else "mid")


fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
for ax, mu_h, ttl in [(axs[0], 1e-9, "μh = 1e-5 cm²/Vs (cited in Y19)"),
                      (axs[1], 3e-8, "μh = 3e-4 cm²/Vs (measured in A14)")]:
    mu_e = 1e-7
    ok_all = np.ones_like(taus, bool)
    for k, (name, d, hole_side, obs) in enumerate(cases):
        RR = []
        for tau in taus:
            # 常规结构：光从 ITO 阳极（空穴接触）入射；反转结构：光从阴极（电子接触）入射
            e_pk = b.eta_A_incoherent(a_pk, d) * b.hecht_profile(d, a_pk, mu_e * tau, mu_h * tau, V0, light_on_hole_side=hole_side, n=200)
            e_ed = b.eta_A_incoherent(a_edge, d) * b.hecht_profile(d, a_edge, mu_e * tau, mu_h * tau, V0, light_on_hole_side=hole_side, n=200)
            RR.append(e_pk / e_ed)
        RR = np.array(RR)
        ok = np.array([classify(r) == obs for r in RR])
        ok_all &= ok
        ax.plot(taus, RR, color=[C_BLUE, C_AQUA, C_ORANGE, C_INK][k], label=f"{name} (obs: {obs})")
    ax.axhline(1, color=C_MUTED, lw=0.8, ls=":")
    ax.axhline(0.5, color=C_MUTED, lw=0.8, ls=":")
    if ok_all.any():
        ax.axvspan(taus[ok_all].min(), taus[ok_all].max(), color=C_AQUA, alpha=0.15, lw=0)
        print(f"Y19 CCN, {ttl}: 四个观测同时满足的 τ 区间 = {taus[ok_all].min():.2e}–{taus[ok_all].max():.2e} s")
    else:
        print(f"Y19 CCN, {ttl}: 不存在同时满足四个观测的 τ")
    ax.set(xscale="log", yscale="log", xlabel="carrier lifetime τ (s)", title=ttl)
    ax.grid(color=C_GRID)
axs[0].set_ylabel("RR = EQE(peak) / EQE(edge)")
axs[0].legend(fontsize=7, loc="lower right", frameon=False)
# 稳健性：对 α_pk、α_edge、V0、μh 的先验网格扫描
import itertools
n_tot = n_strict = n_loose = 0
for a1, a2, v0, muh in itertools.product([5e6, 1e7, 2e7], [1e5, 5e5, 2e6], [0.6, 0.9], [1e-9, 1e-8, 3e-8]):
    for tau in np.geomspace(1e-9, 1e-3, 61):
        n_tot += 1
        RR = []
        for name, d, hs, obs in cases:
            ep = b.eta_A_incoherent(a1, d) * b.hecht_profile(d, a1, 1e-7 * tau, muh * tau, v0, light_on_hole_side=hs, n=150)
            ee = b.eta_A_incoherent(a2, d) * b.hecht_profile(d, a2, 1e-7 * tau, muh * tau, v0, light_on_hole_side=hs, n=150)
            RR.append(ep / ee)
        n_strict += RR[0] >= 1 and RR[1] >= 1 and RR[2] < 0.5 and RR[3] < 0.5
        n_loose += RR[0] >= 1 and RR[1] >= 1 and RR[2] < 1 and RR[3] < 1
print(f"Y19 CCN 稳健性：{n_tot} 组先验组合中，严格判据（CCN: RR<0.5）命中 {n_strict}，宽松判据（RR<1）命中 {n_loose}")

fig.suptitle("Yazmaciyan 2019 CCN classification: aqua = τ range reproducing all four observations", fontsize=9.5)
fig.tight_layout()
fig.savefig("external_ccn.png", dpi=170)

# ============================================================ 汇总图
rows = [r for r in results]
fig, ax = plt.subplots(figsize=(8.6, 0.32 * len(rows) + 1.4))
y = np.arange(len(rows))[::-1]
ax.axvspan(1 / TOL, TOL, color=C_BLUE, alpha=0.07, lw=0)
ax.axvline(1, color=C_MUTED, lw=1)
for yi, (lab, lo, hi, m, note) in zip(y, rows):
    diag = lab.startswith("  diag")
    col = C_MUTED if diag else C_BLUE
    if np.isnan(lo):
        ax.text(0.012, yi, "model form cannot fit (needs negative floor)", va="center", ha="left", fontsize=7, color=C_INK)
        ax.text(60, yi, "FAIL", va="center", fontsize=7.5, fontweight="bold", color=C_INK)
        continue
    if not diag:
        if "upper bound" in lab:
            ok = m <= hi
        else:
            tol = 1.25 if lab.startswith(("A14 ratio", "N24 ratio")) else TOL
            ok = lo / tol <= m <= hi * tol
        ax.text(60, yi, "pass" if ok else "FAIL", va="center", fontsize=7.5,
                fontweight="normal" if ok else "bold", color=C_INK)
    if "upper bound" in lab:
        ax.annotate("", xy=(m / hi, yi), xytext=(1e-2, yi), arrowprops=dict(arrowstyle="-", color=col, lw=2))
        ax.scatter(m / hi, yi, s=40, color=col, zorder=3)
        continue
    ax.plot([lo / m, hi / m], [yi, yi], color=col, lw=3, solid_capstyle="round")
    if lo == hi:
        ax.scatter(lo / m, yi, s=40, color=col, zorder=3)
ax.set_yticks(y, [r[0] for r in rows], fontsize=7.5)
ax.set_ylim(-0.8, len(rows) - 0.2)
ax.set_xscale("log")
ax.set_xlim(1e-2, 1e2)
ax.set_xlabel("predicted / measured (bar = prediction interval from priors); shaded = factor 2")
ax.set_title("Out-of-sample tests on other papers — grey: diagnostic d/2 variant; ratio rows judged at ±25%", fontsize=9)
ax.grid(axis="x", color=C_GRID)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig("external_tests.png", dpi=170)
