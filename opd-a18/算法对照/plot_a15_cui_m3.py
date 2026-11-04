# -*- coding: utf-8 -*-
"""画 a15_cui_m3.py 的结果：Cui 2025 器件 · M3 读数，A15 和 A11 / A12 / Xie MLP 等全部算法对比。
  fig_demo_a15_m3.png   26 族重建示例（每族第 1 条测试谱、第 1 次 M3 抽样，和 fig_demo_allalgos_m3.png 同一份读数）
    黑 = 真值；红 = A15；蓝 = A11（M3 训练）；紫 = Xie MLP；赭 / 绿 = 传统方法里该族误差中位最低者；各曲线峰值归一
  fig_table_a15_m3.png  误差表：行 = 26 族（+ 各族中位），列 = 各方法，格内 = 该族形状误差平均（40 条 × 4 次），每行最好的加粗加框
  py -I -X utf8 plot_a15_cui_m3.py
"""
import os, sys, json, pickle
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm

ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import plot_all_algos as PA                                                       # 只借配色
plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
RES = os.path.join(ROOT, '46_Cui2025_真实R矩阵', 'results'); OUT = os.path.join(RES, 'a15_M3')
J = json.load(open(os.path.join(OUT, 'a15.json'), encoding='utf-8')); D15 = pickle.load(open(os.path.join(OUT, 'demo.pkl'), 'rb'))
A12 = json.load(open(os.path.join(RES, 'a12_M3', 'a12.json'), encoding='utf-8'))
XJ = json.load(open(os.path.join(RES, 'all_algos_M3', 'all_algos.json'), encoding='utf-8'))
DA = pickle.load(open(os.path.join(RES, 'all_algos_M3', 'demo.pkl'), 'rb'))
WL, b, demo, NM = DA['WL'], DA['BM'], DA['demo'], DA['name']; wl = WL[b]
TRAD = ['A1', 'T0', 'T0NN', 'Ridge', 'LapNN', 'Hao', 'CS', 'Gauss1', 'Pinv']
C15 = '#d6402a'
nrm = lambda x: x / x.max() if x.max() > 0 else x
fams = list(demo)
E = lambda m, k: J['fam'][k]['A15'] if m == 'A15' else A12['fam'][k][m] if m == 'A12' else XJ['fam'][k][m]

# ------------------------------------------------------------------ 1. 重建示例
dmax = max(np.abs(nrm(DA['demo'][k]['A11'][b]) - nrm(D15[k]['two'][b])).max() for k in fams)  # 两支融合 = A11：同一份读数的核对
nc = 6; nr = int(np.ceil(len(fams) / nc))
fig, axs = plt.subplots(nr, nc, figsize=(3.1 * nc, 2.35 * nr), sharex=True)
for ax, nm in zip(axs.flat, fams):
    d = demo[nm]; e = d['_err']; bt = min(TRAD, key=lambda m: XJ['fam'][nm][m]); s = d['truth']
    ex = lambda x: float(np.sqrt(np.mean((nrm(x[b]) - nrm(s[b])) ** 2)) / np.ptp(nrm(s[b])))
    assert abs(ex(d['A11']) - e['A11']) < 1e-9, (nm, ex(d['A11']), e['A11'])
    e15 = float(np.sqrt(np.mean((nrm(D15[nm]['A15'][b]) - nrm(s[b])) ** 2)) / np.ptp(nrm(s[b])))
    ax.plot(wl, nrm(s[b]), color=PA.INK, lw=2.2, zorder=5)
    ax.plot(wl, nrm(d[bt][b]), color=PA.COL[bt], lw=1.1, zorder=2)
    ax.plot(wl, nrm(d['XieMLP'][b]), color=PA.COL['XieMLP'], lw=1.1, zorder=3)
    ax.plot(wl, nrm(d['A11'][b]), color=PA.COL['A11'], lw=1.3, zorder=4)
    ax.plot(wl, nrm(D15[nm]['A15'][b]), color=C15, lw=1.5, zorder=6)
    ax.set_ylim(-0.25, 1.3); ax.set_xlim(wl[0], wl[-1]); ax.grid(color=PA.GRID, lw=0.5); ax.tick_params(labelsize=7)
    ax.set_title(nm, fontsize=9.5, color=PA.INK, pad=15)
    ax.text(0.5, 1.02, f'A15 {e15:.3f} · A11 {e["A11"]:.3f} · Xie {e["XieMLP"]:.3f} · {NM[bt]} {e[bt]:.3f}',
            transform=ax.transAxes, ha='center', va='bottom', fontsize=6.2, color=PA.INK2)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
for ax in list(axs.flat)[len(fams):]:
    ax.axis('off')
for ax in axs[-1]:
    ax.set_xlabel('波长（nm）', fontsize=8)
h = [Line2D([], [], color=PA.INK, lw=2.2, label='真值'), Line2D([], [], color=C15, lw=1.5, label='A15（三支融合 + 峰位校正，M3 训练）'),
     Line2D([], [], color=PA.COL['A11'], lw=1.3, label='A11（M3 训练）'), Line2D([], [], color=PA.COL['XieMLP'], lw=1.1, label='Xie MLP（Xie 2026b）'),
     Line2D([], [], color=PA.COL['T0'], lw=1.1, label='传统方法里该族误差中位最低者（名字见格顶误差行末项）'),
     Line2D([], [], color=PA.COL['A1'], lw=1.1, label='同上，恰好是经典平滑非负（41 号）时')]
fig.legend(handles=h, loc='upper left', bbox_to_anchor=(0.01, 0.975), ncol=6, fontsize=9, frameon=False)
fig.suptitle('Cui 2025 器件 · M3 读数（recon 档白噪声 + 逐 Vgs 帧增益漂移，a = 1）：A15 与其他算法的各谱族重建示例（每族第 1 条测试谱、第 1 次抽样；曲线峰值归一）\n'
             '格顶小字 = 这一条的误差（峰值归一均方根 / 真值极差，越小越好）；要训练的方法都在 M3 读数上训练',
             fontsize=11, color=PA.INK, x=0.01, y=1.0, ha='left')
plt.tight_layout(rect=(0, 0, 1, 0.955)); f = os.path.join(OUT, 'fig_demo_a15_m3.png'); fig.savefig(f, dpi=140, bbox_inches='tight'); plt.close(fig)
print(f, f'  示例读数核对：两支融合 vs all_algos 的 A11 峰值归一最大差 {dmax:.1e}')

# ------------------------------------------------------------------ 2. 误差表
COLS = ['A15', 'A12', 'A11', 'XieMLP', 'NSF'] + sorted(TRAD, key=lambda m: np.median([XJ['fam'][k][m] for k in fams]))
LAB = dict(NM, A15='A15', A12='A12', A11='A11', XieMLP='Xie MLP', NSF='Wang NSF')
rows = sorted(fams, key=lambda k: E('A15', k))
V = np.array([[E(m, k) for m in COLS] for k in rows]); med = np.median(V, 0)
V = np.vstack([V, med]); rl = rows + ['各族中位']
fig, ax = plt.subplots(figsize=(1.05 * len(COLS) + 2.6, 0.34 * len(rl) + 1.6))
ax.imshow(V, cmap='YlOrBr', norm=LogNorm(vmin=0.015, vmax=0.6), aspect='auto')
for i in range(len(rl)):
    j0 = int(np.argmin(V[i]))
    for j in range(len(COLS)):
        v = V[i, j]; dark = v > 0.2
        ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=7.6, color='white' if dark else PA.INK, fontweight='bold' if j == j0 else 'normal')
    ax.add_patch(plt.Rectangle((j0 - 0.5, i - 0.5), 1, 1, fill=False, ec=PA.INK, lw=1.6))
ax.axhline(len(rows) - 0.5, color=PA.INK, lw=1.2); ax.axvline(4.5, color=PA.INK, lw=1.0)
ax.set_xticks(range(len(COLS))); ax.set_xticklabels([LAB[m] for m in COLS], rotation=40, ha='right', fontsize=8.5)
ax.set_yticks(range(len(rl))); ax.set_yticklabels(rl, fontsize=8.5); ax.tick_params(length=0)
for t, m in zip(ax.get_xticklabels(), COLS):
    if m == 'A15':
        t.set_color(C15); t.set_fontweight('bold')
nb = {m: int(sum(np.argmin(V[i]) == j for i in range(len(rows)))) for j, m in enumerate(COLS)}
ax.set_title('Cui 2025 器件 · M3 读数：各族形状误差（每族 40 条 × 4 次平均；峰值归一均方根 / 真值极差，越小越好）\n'
             '框 = 该行最好；行按 A15 从小到大排；竖实线左边是本项目的和文献机器学习方法，右边是传统方法（按各族中位排）\n'
             '各方法拿第一的族数：' + '，'.join(f'{LAB[m]} {nb[m]}' for m in COLS if nb[m]), fontsize=9.5, loc='left')
for sp in ax.spines.values():
    sp.set_visible(False)
plt.tight_layout(); f = os.path.join(OUT, 'fig_table_a15_m3.png'); fig.savefig(f, dpi=140, bbox_inches='tight'); print(f)
print('各族中位：', {m: round(float(v), 4) for m, v in zip(COLS, med)}); print('拿第一族数：', nb)
