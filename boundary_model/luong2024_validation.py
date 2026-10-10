"""循环论证审查：区分“拟合残差”（样本内）和“真正的预测”（样本外），并对比原文结果。

样本外数据来源：
- 留一法（LOO）：带宽 4 点中每次去掉 1 点重拟合，再预测被去掉的点；
- SI Table S1 的散粒噪声列 i_sh = sqrt(2qI_d) → 720、810 nm 器件的 J_d（标定时没用）；
- 正文“EQE 比非共振条件提高约 5 倍”（实验值，标定时用的是 Fig.1b 的 FDTD 5 倍）；
- 交叉预测：用 245 nm 的 μτ 预测 655 nm 的 R_c。
运行：python3 luong2024_validation.py → luong2024_validation.png
"""
import contextlib
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import opd_bounds as b

with contextlib.redirect_stdout(io.StringIO()):
    import luong2024_fig as L

TOL = 2.0          # 判定容差：预测/实测在 [1/2, 2] 内算通过（审查时设定）
TOL_ABS = 0.10     # 有界量（收集比 R_c ∈ [0,1]）用绝对容差 ±0.10，倍数容差对它没有意义
A_CM2 = 0.047      # 4.7 mm²
rows = []          # (标签, 预测, 实测, 是否样本外, 说明)

# ---------------- 带宽：样本内残差
for d, V, f in L.F3DB:
    rows.append((f"f3dB {d*1e9:.0f} nm @{V} V (fit residual)", L.f_model(L.best.x, d, V), f, False, ""))

# ---------------- 带宽：留一法
for k, (d, V, f) in enumerate(L.F3DB):
    tr = [x for i, x in enumerate(L.F3DB) if i != k]
    best = None
    for g in [(1e-8, 0.3, 50), (1e-7, 0.3, 200), (1e-7, 0.05, 500), (3e-8, 1, 100)]:
        s = least_squares(lambda p: [np.log(L.f_model(p, dd, VV) / ff) for dd, VV, ff in tr], np.log(g))
        best = s if best is None or s.cost < best.cost else best
    rows.append((f"f3dB {d*1e9:.0f} nm @{V} V (leave-one-out)", L.f_model(best.x, d, V), f, True, ""))

# ---------------- 暗电流：Table S1 散粒噪声反推
S1 = [(245, 0.1, 9.12e-15), (655, 0.1, 6.72e-15), (720, 0.1, 7.46e-15), (810, 0.1, 7.38e-15),
      (245, 5, 8.28e-14), (655, 5, 2.43e-14), (720, 5, 2.12e-14), (810, 5, 2.15e-14)]
FITTED_JD = {(245, 5), (655, 5), (655, 0.1)}
for d, V, ish in S1:
    J_meas = ish**2 / (2 * b.q) / A_CM2 * 1e9        # nA/cm²
    J_pred = L.j_dark(d * 1e-9, V) / L.nA
    oos = (d, V) not in FITTED_JD
    rows.append((f"J_d {d} nm @{V} V" + ("" if oos else " (fit residual)"), J_pred, J_meas, oos, ""))

# ---------------- 光学：峰谷比（实验约 5 倍）
for m, lo, hi in [(2, 100e-9, 400e-9), (4, 550e-9, 760e-9)]:
    dd = np.linspace(lo, hi, 3000)
    A = L.A_of_d(dd)
    rows.append((f"on/off-resonance ratio m={m}", A.max() / A.min(), 5.0, True, ""))

# ---------------- 收集：交叉预测
rows.append(("R_c 655 nm from 245-nm μτ (±0.10 abs.)", L.Rc(L.MUTAU[245e-9], 655e-9), 0.84, True, "abs"))

# ---------------- 打印
print(f"{'检验项':45s} {'预测':>10s} {'实测':>10s} {'比值':>7s}  类型      结论")
def passes(p, m, kind):
    return abs(p - m) <= TOL_ABS if kind == "abs" else 1 / TOL <= p / m <= TOL


for lab, p, m, oos, kind in rows:
    r = p / m
    ok = passes(p, m, kind)
    print(f"{lab:45s} {p:10.3g} {m:10.3g} {r:7.2f}  {'样本外' if oos else '样本内'}  {'通过' if ok else '不通过'}")

# ---------------- 图：预测/实测比值点图
C_IN, C_OUT, C_INK, C_MUTED, C_GRID = "#8a8986", "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
fig, ax = plt.subplots(figsize=(8.2, 6.6))
y = np.arange(len(rows))[::-1]
ax.axvspan(1 / TOL, TOL, color="#2a78d6", alpha=0.07, lw=0)
ax.axvline(1, color=C_MUTED, lw=1)
for yi, (lab, p, m, oos, kind) in zip(y, rows):
    r = p / m
    ok = passes(p, m, kind)
    if oos:
        ax.scatter(r, yi, s=64, color=C_OUT, edgecolor="white", lw=1.5, zorder=3)
    else:
        ax.scatter(r, yi, s=64, facecolor="white", edgecolor=C_IN, lw=2, zorder=3)
    ax.text(70, yi, ("pass" if ok else "FAIL") if oos else "in-sample", va="center", fontsize=8,
            color=C_INK if oos else C_MUTED, fontweight="bold" if (oos and not ok) else "normal")
ax.set_yticks(y, [r[0] for r in rows], fontsize=8)
ax.set_xscale("log")
ax.set_xlim(0.1, 120)
ax.set_xlabel("predicted / measured (log scale); shaded band = factor-2 tolerance")
ax.set_title("Luong 2024: fit residuals (hollow) vs genuine out-of-sample predictions (filled)", fontsize=10)
ax.grid(axis="x", color=C_GRID)
ax.grid(axis="y", visible=False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig("luong2024_validation.png", dpi=170)
