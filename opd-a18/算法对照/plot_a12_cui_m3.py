# -*- coding: utf-8 -*-
"""画 a12_cui_m3.py 的结果：Cui 2025 器件 · M3 读数，A12（A11 + 峰位校正）对 A11。
  fig_a12_vs_a11.png  左：各族形状误差 A12 / A11 − 1（%）；右：峰位误差（单峰 15–60 nm、窄线族中位；实测 5 束 RMSD；模拟 M3 5 束 RMSD / 中位）
  py -I -X utf8 plot_a12_cui_m3.py
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
CUI = os.path.join(ROOT, '46_Cui2025_真实R矩阵', 'results')
OUT = os.path.join(CUI, 'a12_M3'); J = json.load(open(os.path.join(OUT, 'a12.json'), encoding='utf-8'))
XJ = json.load(open(os.path.join(CUI, 'all_algos_M3', 'all_algos.json'), encoding='utf-8'))
C = {'gate': '#9a9993', 'A11': '#2a78d6', 'A12': '#0b3d82', 'XieMLP': '#8a5cd1'}; NB5 = 5
F = J['fam']; fams = sorted(F, key=lambda k: F[k]['A12'] / F[k]['A11'])
rel = np.array([(F[k]['A12'] / F[k]['A11'] - 1) * 100 for k in fams])

fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 7.5), gridspec_kw=dict(width_ratios=[1, 1.15]))
a1.barh(np.arange(len(fams)), rel, color=['#1baf7a' if v < -0.01 else '#d0574a' if v > 0.01 else '#b9b8b2' for v in rel])
a1.set_yticks(np.arange(len(fams))); a1.set_yticklabels(fams, fontsize=8.5); a1.invert_yaxis(); a1.axvline(0, color='k', lw=0.8)
for i, v in enumerate(rel):
    a1.text(max(v, 0) + 0.05, i, f'{v:+.1f} %', va='center', ha='left', fontsize=7.5)
a1.set_xlabel('形状误差 A12 / A11 − 1（%，负 = A12 更好）')
a1.set_title(f'各族形状误差：中位 A11 {J["med"]["A11"]:.4f} → A12 {J["med"]["A12"]:.4f}（Xie MLP {J["med"]["XieMLP"]:.4f}）', fontsize=10.5)
a1.grid(axis='x', color='#e4e3df', lw=0.6); a1.set_axisbelow(True)

rm = lambda a: float(np.sqrt(np.nanmean(np.square(np.asarray(a, float)))))
grp = ['单峰 15–60 nm\n族峰位中位', '窄线 4–10 nm\n族峰位中位', '实测 5 束\n峰位 RMSD', '模拟 M3 5 束\n峰位 RMSD', '模拟 M3 5 束\n|峰位误差| 中位']
M = ['gate', 'A11', 'A12', 'XieMLP']; lab = {'gate': '门控支', 'A11': 'A11', 'A12': 'A12（A11 + 峰位校正）', 'XieMLP': 'Xie MLP'}


def val(m, gi):
    if gi < 2:
        nm = ['单峰 15–60 nm', '窄线 4–10 nm'][gi]
        return J['fampk'][nm][m] if m != 'XieMLP' else XJ['fampk'][nm]['XieMLP']
    src = J if m != 'XieMLP' else XJ
    if gi == 2:
        return rm(np.array(src['real'][m]['dpk'])[:NB5])
    s = np.array(src['sim'][m]['dpk'], float)[:NB5]
    return rm(s) if gi == 3 else float(np.nanmedian(np.abs(s)))


x = np.arange(len(grp)); w = 0.2
for i, m in enumerate(M):
    v = [val(m, gi) for gi in range(len(grp))]
    a2.bar(x + (i - 1.5) * w, v, w, color=C[m], label=lab[m])
    for xi, vi in zip(x + (i - 1.5) * w, v):
        a2.text(xi, vi * 1.03, f'{vi:.2f}', ha='center', va='bottom', fontsize=7, rotation=90)
a2.set_xticks(x); a2.set_xticklabels(grp, fontsize=8.5); a2.set_ylabel('峰位误差（nm）'); a2.legend(fontsize=8.5, frameon=False, loc='upper left')
a2.set_title('峰位误差（越低越好）', fontsize=10.5); a2.grid(axis='y', color='#e4e3df', lw=0.6); a2.set_axisbelow(True)
for sp in ('top', 'right'):
    a1.spines[sp].set_visible(False); a2.spines[sp].set_visible(False)
fig.suptitle(f'Cui 2025 器件 · M3 读数：A12 = A11 输出往门控支峰位平移 q·d（q 小网在验证谱上训，λ = {J["lam"]:g}）对 A11', fontsize=12)
plt.tight_layout(); f = os.path.join(OUT, 'fig_a12_vs_a11.png'); fig.savefig(f, dpi=140); print(f)
