# -*- coding: utf-8 -*-
"""画 all_algos_cui_m3.py 的结果（46_Cui2025_真实R矩阵/results/all_algos_M3/）：
  fig_allrank_m3.png     生成谱族排名图（M3 读数），画法同 Uddin 的 plot_all_algos.fam_fig（直接调用它，标题里的器件名换成 Cui）
  fig_allrank_beams.png  测试光排名图：条 = 520–680 nm 5 束实测 |峰位误差| 平均，菱形 = 720 nm 那束（波段边缘），空心圈 = 同一批光模拟 M3 读数 200 次平均
  fig_all_algos_M3.png   两栏速览：谱族误差中位、测试光峰位 RMSD（实测 vs 模拟 M3）
  py -I -X utf8 plot_all_algos_cui_m3.py
"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text

ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import plot_all_algos as PA                                                       # 只用它的画法和配色，不跑它的 main
plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
OUT = os.path.join(ROOT, '46_Cui2025_真实R矩阵', 'results', 'all_algos_M3')
J0 = json.load(open(os.path.join(OUT, 'all_algos.json'), encoding='utf-8')); NM = J0['name']; NB5 = 5
INK, INK2, GRID = PA.INK, PA.INK2, PA.GRID


def cui_text(fig):
    """PA.fam_fig 的标题 / 图例写死了 Uddin，存图前换成 Cui。"""
    for t in fig.findobj(Text):
        s = t.get_text()
        if 'Uddin' in s:
            t.set_text(s.replace('Uddin 2024 器件模拟器', 'Cui 2025 器件模拟器').replace('Uddin 论文算法', 'Cui 论文算法').replace('Uddin', 'Cui'))


_save = plt.Figure.savefig
plt.Figure.savefig = lambda fig, *a, **k: (cui_text(fig), _save(fig, *a, **k))[1]
PA.MODE['m3'] = 'M3 读数（recon 档白噪声 + 逐 Vgs 帧增益漂移，a = 1）'

# ---------------- 谱族排名图：套 PA.fam_fig 的数据格式
J = dict(J0, fam={'m3': J0['fam']})
print('m3', [(m, round(v, 3), w) for m, v, w in PA.fam_fig(J, 'm3', OUT)])

# ---------------- 测试光排名图（Cui：720 nm 在波段边缘单列；参考 = 标称波长）
lt = np.array(J0['lam_test']); R, S = J0['real'], J0['sim']; ms = [m for m in NM if m in R]
a5 = {m: float(np.mean(np.abs(np.array(R[m]['dpk'])[:NB5]))) for m in ms}
a7 = {m: abs(R[m]['dpk'][-1]) for m in ms}
sm = {m: float(np.nanmean(np.abs(np.array(S[m]['dpk'], float)[:NB5]))) for m in ms if m in S}
order = sorted(ms, key=a5.get); y = np.arange(len(order))
fg, ax = plt.subplots(figsize=(11, 0.42 * len(order) + 1.6))
for i, m in enumerate(order):
    ax.barh(i, max(a5[m], 1e-3), height=0.56, color=PA.COL[m], zorder=2)
    ax.scatter(max(a7[m], 0.012), i, marker='D', s=26, color=INK2, zorder=3)
    if m in sm:
        ax.scatter(sm[m], i, marker='o', s=40, facecolors='none', edgecolors=INK, lw=1.0, zorder=4)
    ax.text(1.005, i, f'{a5[m]:.2f} nm（模拟 M3 {sm.get(m, np.nan):.2f}）', transform=ax.get_yaxis_transform(), va='center', fontsize=8.5, color=INK)
ax.set_xscale('log'); ax.set_xlim(0.01, 200)
ax.set_yticks(y); ax.set_yticklabels([PA.lab(J0, m) for m in order], fontsize=8.5, color=INK); ax.invert_yaxis()
ax.grid(axis='x', color=GRID, lw=0.6, which='both'); ax.set_axisbelow(True)
ax.set_xlabel('|峰位误差|（nm，对数刻度，越小越好；0 画在 0.01 处）', fontsize=8, color=INK2)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
h = [Line2D([], [], marker='s', ls='', color=INK2, ms=8, label=f'条 = 实测 {", ".join(f"{v:.0f}" for v in lt[:NB5])} nm 5 束平均'),
     Line2D([], [], marker='D', ls='', color=INK2, ms=5, label='720 nm 那束（波段边缘）'),
     Line2D([], [], marker='o', ls='', mfc='none', mec=INK, ms=6, label=f'同一批光 · 模拟 M3 读数 {J0["nsim"]} 次平均（5 束）')]
ax.legend(handles=h, fontsize=8, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, handletextpad=0.3, columnspacing=1.2)
fg.suptitle('Cui 2025 作者的 6 束真实读数（参考 = 标称波长；6 束都正好在 R 的学习波长上，线性方法占便宜）：各算法峰位误差排名\n'
            '颜色同谱族图（蓝 = A11，橙 = 论文算法，紫 = 文献机器学习，赭 = 文献其他）；要训练 / 调参的方法都在 M3 读数上训练',
            fontsize=10, color=INK, x=0.02, ha='left')
fg.savefig(os.path.join(OUT, 'fig_allrank_beams.png'), dpi=150, bbox_inches='tight'); plt.close(fg)
print('beams', [(m, round(a5[m], 2), round(a7[m], 2), None if m not in sm else round(sm[m], 2)) for m in order])

# ---------------- 两栏速览
F = J0['fam']; fm = list(next(iter(F.values()))); med = {m: np.median([F[k][m] for k in F]) for m in fm}
rm = lambda a: float(np.sqrt(np.nanmean(np.square(np.asarray(a, float)))))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw=dict(width_ratios=[1, 1.25]))
o = sorted(fm, key=med.get); yy = np.arange(len(o))
a1.barh(yy, [med[m] for m in o], color=['tab:red' if m == 'A11' else 'tab:gray' for m in o])
for i, m in enumerate(o):
    a1.text(med[m] * 1.03, i, f'{med[m]:.3f}', va='center', fontsize=8)
a1.set_yticks(yy); a1.set_yticklabels([NM[m] for m in o], fontsize=9); a1.invert_yaxis()
a1.set_xlabel('各族误差中位（峰值归一均方根 / 真值极差）'); a1.set_title(f'生成谱族 · M3 读数（{len(F)} 族，越短越好）', fontsize=11)
o2 = sorted(R, key=lambda m: rm(np.array(R[m]['dpk'])[:NB5])); x = np.arange(len(o2))
vr = [rm(np.array(R[m]['dpk'])[:NB5]) for m in o2]; vs = [rm(np.array(S[m]['dpk'], float)[:NB5]) for m in o2]
vmed = [float(np.nanmedian(np.abs(np.array(S[m]['dpk'], float)[:NB5]))) for m in o2]
a2.bar(x - 0.2, vr, 0.4, color='k', label='作者实测 6 束（前 5 束）')
a2.bar(x + 0.2, vs, 0.4, color='tab:orange', label=f'同一批光的模拟 M3 读数 × {J0["nsim"]}')
a2.plot(x + 0.2, vmed, 'v', color='tab:brown', ms=6, label='模拟 M3 的 |峰位误差| 中位')
for i, v in enumerate(vr):
    a2.text(i - 0.2, v * 1.12, f'{v:.2f}', ha='center', fontsize=7, rotation=90)
a2.set_yscale('log'); a2.set_xticks(x); a2.set_xticklabels([NM[m] for m in o2], rotation=50, ha='right', fontsize=8)
a2.set_ylabel('峰位 RMSD（nm，对数）'); a2.legend(fontsize=8, loc='upper left')
a2.set_title('测试光（520–680 nm 单色光，标称波长为参考）', fontsize=11)
fig.suptitle('Cui 2025 器件 · 读数含 M3 帧增益漂移：全部算法对比（要训练 / 调参的方法都在 M3 读数上训练）', fontsize=12)
plt.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_all_algos_M3.png'), dpi=150)
print(OUT)
