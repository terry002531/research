# -*- coding: utf-8 -*-
"""读 a11_peakfix.py 的 peakfix.json，画 A12（= A11p，最后一步加峰位校正）对旧 A11 的对比：只用生成谱族（全模拟）。
  左：每族形状误差相对旧 A11 的变化（%）；右：两个单峰族的峰位 |误差| 中位，和两支、Xie MLP 并排。
  py -I -X utf8 plot_peakfix.py      输出 45_Uddin2024_真实R矩阵/results/a11_peakfix/fig_peakfix.png
"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
INK, INK2, GRID = '#1f1f1e', '#6b6a65', '#e4e3df'
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '45_Uddin2024_真实R矩阵', 'results')
ROWS = [('a11_peakfix', 'real', 'A12 对旧 A11')]                  # A11_R 两行已撤（用户 2026-11-02：A11_R 等实测数据再说）
MODE = {'real': '真实模式读数', 'realR': '真实模式 + R 标定误差读数'}
BARS = [('gate', '门控支', '#b9b8b2'), ('cons', '网络支', '#8d8c86'), ('A11', '旧 A11', '#9cc0ea'), ('A11p', 'A12', '#2a78d6'), ('XieMLP', 'Xie MLP', '#eb6834')]
PKF = ['窄线 4–10 nm', '单峰 15–60 nm']

XJ = json.load(open(os.path.join(RES, 'all_algos', 'all_algos.json'), encoding='utf-8'))
fg, axs = plt.subplots(len(ROWS), 2, figsize=(13, 4.6 * len(ROWS)), gridspec_kw=dict(width_ratios=[1.5, 1], wspace=0.35, hspace=0.45), squeeze=False)
for r, (d, mo, tag) in enumerate(ROWS):
    J = json.load(open(os.path.join(RES, d, 'peakfix.json'), encoding='utf-8'))
    F = {k: v for k, v in J['fam'][mo].items() if not k.startswith('_')}
    ch = {k: (v['A11p'] / v['A11'] - 1) * 100 for k, v in F.items()}; fam = sorted(ch, key=ch.get); y = np.arange(len(fam))
    ax = axs[r, 0]
    ax.barh(y, [ch[k] for k in fam], height=0.62, color=['#2a78d6' if ch[k] < 0 else '#c9c8c2' for k in fam])
    ax.axvline(0, color=INK2, lw=0.8); ax.set_yticks(y); ax.set_yticklabels(fam, fontsize=7.5, color=INK); ax.invert_yaxis()
    ax.grid(axis='x', color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(axis='x', labelsize=7.5)
    med = J['fam'][mo]['_中位']
    ax.set_xlabel(f'A12 形状误差相对旧 A11 的变化（%，负 = 变好）    各族中位：旧 A11 {med["A11"]:.4f} → A12 {med["A11p"]:.4f}，Xie MLP {med.get("XieMLP", np.nan):.4f}',
                  fontsize=8, color=INK2)
    ax.set_title(f'{tag} · {MODE[mo]}：26 族形状误差（λ = {J["lam"]:g}）', fontsize=9.5, color=INK, loc='left')
    bx = axs[r, 1]; w = 0.16
    for i, (m, nm, c) in enumerate(BARS):
        v = [J['fampk'][mo][f][m] if m != 'XieMLP' else XJ['fampk'][mo][f]['XieMLP'] for f in PKF]
        x = np.arange(len(PKF)) + (i - 2) * w; bx.bar(x, v, width=w * 0.9, color=c, label=nm)
        for xx, vv in zip(x, v):
            bx.text(xx, vv + 0.02, f'{vv:.2f}', ha='center', va='bottom', fontsize=6.5, color=INK)
    bx.set_ylim(0, bx.get_ylim()[1] * 1.3); bx.set_xticks(range(len(PKF))); bx.set_xticklabels(PKF, fontsize=8.5, color=INK); bx.tick_params(axis='y', labelsize=7.5)
    bx.set_ylabel('峰位 |误差| 中位（nm）', fontsize=8, color=INK2); bx.grid(axis='y', color=GRID, lw=0.6); bx.set_axisbelow(True)
    bx.set_title('单峰族峰位误差（越低越好）', fontsize=9.5, color=INK, loc='left')
    if r == 0:
        bx.legend(fontsize=7.5, frameon=False, ncol=3, loc='upper left')
    for a in (ax, bx):
        for sp in ('top', 'right'):
            a.spines[sp].set_visible(False)
fg.suptitle('A12 = A11 最后一步加峰位校正：Uddin 模拟器，生成谱族每族 40 条 × 4 次读数', fontsize=11, color=INK, y=0.915 if len(ROWS) > 1 else 1.0)
out = os.path.join(RES, 'a11_peakfix', 'fig_peakfix.png'); fg.savefig(out, dpi=150, bbox_inches='tight'); print(out)
