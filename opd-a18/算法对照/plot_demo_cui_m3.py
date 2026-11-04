# -*- coding: utf-8 -*-
"""画 Cui 2025 器件 · M3 读数下各谱族的重建示例（all_algos_cui_m3.py DEMO=1 出的 demo.pkl）：
  fig_demo_allalgos_m3.png  26 族各一格，每格 = 该族第 1 条测试谱、第 1 次 M3 抽样读数
    黑 = 真值；蓝 = A11（M3 训练）；紫实线 = Xie MLP；紫虚线 = Wang NSF；赭 = 传统方法里该族误差中位最低的那个（按 all_algos.json 逐族挑）
    各曲线峰值归一；格顶小字 = 这一条的误差（峰值归一均方根 / 真值极差）
  py -I -X utf8 plot_demo_cui_m3.py
"""
import os, sys, json, pickle
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import plot_all_algos as PA                                                       # 只借配色
plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
OUT = os.path.join(ROOT, '46_Cui2025_真实R矩阵', 'results', 'all_algos_M3')
D = pickle.load(open(os.path.join(OUT, 'demo.pkl'), 'rb')); J = json.load(open(os.path.join(OUT, 'all_algos.json'), encoding='utf-8'))
WL, b, demo, NM = D['WL'], D['BM'], D['demo'], D['name']; wl = WL[b]
TRAD = ['A1', 'T0', 'T0NN', 'Ridge', 'LapNN', 'Hao', 'CS', 'Gauss1', 'Pinv']
LINES = [('A11', PA.COL['A11'], '-', 1.6), ('XieMLP', PA.COL['XieMLP'], '-', 1.2), ('NSF', PA.COL['NSF'], '--', 1.2)]
nrm = lambda x: x / x.max() if x.max() > 0 else x

fams = list(demo); nc = 6; nr = int(np.ceil(len(fams) / nc))
fig, axs = plt.subplots(nr, nc, figsize=(3.1 * nc, 2.35 * nr), sharex=True)
for ax, nm in zip(axs.flat, fams):
    d, e = demo[nm], demo[nm]['_err']; bt = min(TRAD, key=lambda m: J['fam'][nm][m])
    ax.plot(wl, nrm(d['truth'][b]), color=PA.INK, lw=2.2, zorder=5)
    ax.plot(wl, nrm(d[bt][b]), color=PA.COL[bt], lw=1.2, zorder=2)
    for m, c, ls, lw in LINES:
        ax.plot(wl, nrm(d[m][b]), color=c, ls=ls, lw=lw, zorder=4 if m == 'A11' else 3)
    ax.set_ylim(-0.25, 1.3); ax.set_xlim(wl[0], wl[-1]); ax.grid(color=PA.GRID, lw=0.5); ax.tick_params(labelsize=7)
    ax.set_title(nm, fontsize=9.5, color=PA.INK, pad=15)                       # 误差写在标题下，不压曲线
    ax.text(0.5, 1.02, f'A11 {e["A11"]:.3f} · Xie {e["XieMLP"]:.3f} · NSF {e["NSF"]:.3f} · {NM[bt]} {e[bt]:.3f}',
            transform=ax.transAxes, ha='center', va='bottom', fontsize=6.2, color=PA.INK2)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
for ax in list(axs.flat)[len(fams):]:
    ax.axis('off')
for ax in axs[-1]:
    ax.set_xlabel('波长（nm）', fontsize=8)
h = [Line2D([], [], color=PA.INK, lw=2.2, label='真值'), Line2D([], [], color=PA.COL['A11'], lw=1.6, label=NM['A11']),
     Line2D([], [], color=PA.COL['XieMLP'], lw=1.2, label='Xie MLP（Xie 2026b）'), Line2D([], [], color=PA.COL['NSF'], ls='--', lw=1.2, label='Wang NSF（Wang 2024）'),
     Line2D([], [], color=PA.COL['T0'], lw=1.2, label='传统方法里该族误差中位最低者（名字见格顶误差行末项）'),
     Line2D([], [], color=PA.COL['A1'], lw=1.2, label='同上，恰好是经典平滑非负（41 号）时')]
fig.legend(handles=h, loc='upper left', bbox_to_anchor=(0.01, 0.975), ncol=6, fontsize=9, frameon=False)
fig.suptitle('Cui 2025 器件 · M3 读数（recon 档白噪声 + 逐 Vgs 帧增益漂移，a = 1）：各谱族重建示例（每族第 1 条测试谱、第 1 次抽样；曲线峰值归一）\n'
             '格顶小字 = 这一条的误差（峰值归一均方根 / 真值极差，越小越好）；要训练的方法都在 M3 读数上训练',
             fontsize=11, color=PA.INK, x=0.01, y=1.0, ha='left')
plt.tight_layout(rect=(0, 0, 1, 0.955)); f = os.path.join(OUT, 'fig_demo_allalgos_m3.png'); fig.savefig(f, dpi=140, bbox_inches='tight')
print(f)
for nm in fams:
    print(nm, {m: round(v, 3) for m, v in demo[nm]['_err'].items()})
