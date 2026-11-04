# -*- coding: utf-8 -*-
"""读 a11_device.py 输出的 a11.json，画"各族误差排名图"：左 = 每族各方法误差（按 A11 排序），右 = 各方法中位误差 + 第一名族数。
  py -I -X utf8 plot_rank.py <results/a11 目录> [<目录2> ...]     每个目录、每个读数模式各出一张 fig_rank_<模式>.png
"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
INK, INK2, GRID = '#1f1f1e', '#6b6a65', '#e4e3df'
# 主方法：颜色 + 点形双重编码（浅绿对白底对比度 2.74，靠点形和图例补）；A11 两个支路是内部部件，灰色小空心点
MAIN = [('A11', 'A11', '#2a78d6', 'o', 70), ('paper', '论文算法', '#eb6834', 's', 50), ('A1', '经典平滑非负', '#1baf7a', '^', 55)]
SUB = [('gate', 'A11 门控支路', 'v'), ('net', 'A11 网络支路', 'D')]
MODE = {'ideal': '理想读数', 'real': '真实模式读数', 'realR': '真实模式 + R 标定误差读数',
        'recon': '真实模式 recon 档', 'reading': '真实模式 reading 档'}


def one(J, mo, out, tag):
    F = {k.replace('留出·', ''): v for k, v in J['fam'][mo].items() if not k.startswith('_')}
    main = [m for m in MAIN if m[0] in next(iter(F.values()))]; sub = [m for m in SUB if m[0] in next(iter(F.values()))]
    fam = sorted(F, key=lambda k: F[k]['A11']); y = np.arange(len(fam))
    fg = plt.figure(figsize=(11.5, 0.32 * len(fam) + 1.8))
    gs = fg.add_gridspec(2, 2, width_ratios=[3, 1.1], height_ratios=[1, 2.6], wspace=0.42, hspace=0.25)
    ax = fg.add_subplot(gs[:, 0]); bx = fg.add_subplot(gs[0, 1])
    for i, k in enumerate(fam):
        v = [F[k][m[0]] for m in main]
        ax.plot([min(v), max(v)], [i, i], color=GRID, lw=3, solid_capstyle='round', zorder=1)
    for m, nm, mk in sub:
        ax.scatter([F[k][m] for k in fam], y, s=18, marker=mk, facecolors='none', edgecolors='#a3a29c', lw=0.8, zorder=2, label=nm)
    for m, nm, c, mk, s in main[::-1]:
        ax.scatter([F[k][m] for k in fam], y, s=s, marker=mk, color=c, edgecolors='white', lw=1.0, zorder=3, label=nm)
    ax.set_yticks(y); ax.set_yticklabels(fam, fontsize=8, color=INK); ax.invert_yaxis()
    ax.set_xlim(0, None); ax.grid(axis='x', color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel('误差（峰值归一均方根 / 真值极差，越小越好）', fontsize=8, color=INK2)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    h, l = ax.get_legend_handles_labels(); o = [l.index(m[1]) for m in main] + [l.index(m[1]) for m in sub]
    ax.legend([h[i] for i in o], [l[i] for i in o], fontsize=8, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=5, frameon=False,
              handletextpad=0.3, columnspacing=1.2)

    # 右：中位误差横条 + 第一名族数（只比三种主方法）
    med = {m[0]: float(np.median([F[k][m[0]] for k in fam])) for m in main}
    win = {m[0]: sum(min(main, key=lambda q: F[k][q[0]])[0] == m[0] for k in fam) for m in main}
    order = sorted(main, key=lambda m: med[m[0]])
    for j, (m, nm, c, mk, s) in enumerate(order):
        bx.barh(j, med[m], height=0.62, color=c)
        bx.text(med[m] + max(med.values()) * 0.03, j, f'{med[m]:.3f}', va='center', fontsize=9, color=INK)
    bx.set_yticks(range(len(order))); bx.set_yticklabels([f'{m[1]}\n{win[m[0]]}/{len(fam)} 族第一' for m in order], fontsize=8, color=INK)
    bx.invert_yaxis(); bx.set_xlim(0, max(med.values()) * 1.3)
    bx.set_title('各族误差中位数（越短越好）', fontsize=9, color=INK); bx.tick_params(axis='x', labelsize=7); bx.tick_params(axis='y', length=0)
    for sp in ('top', 'right'):
        bx.spines[sp].set_visible(False)
    fg.suptitle(f'{J["device"]}{tag}：{len(fam)} 族测试谱，{MODE.get(mo, mo)}（每族 {J.get("nte", 40)} 条，按 A11 误差排序）', fontsize=10.5, color=INK)
    fg.savefig(os.path.join(out, f'fig_rank_{mo}.png'), dpi=150, bbox_inches='tight'); plt.close(fg)
    return med, win


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    for d in sys.argv[1:]:
        J = json.load(open(os.path.join(d, 'a11.json'), encoding='utf-8'))
        tag = '（训练含 R 标定误差）' if os.path.basename(os.path.normpath(d)).startswith('a11_R') else ''
        for mo in J['fam']:
            med, win = one(J, mo, d, tag)
            print(d, mo, {k: round(v, 3) for k, v in med.items()}, win)
