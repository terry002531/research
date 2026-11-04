# -*- coding: utf-8 -*-
"""读 all_algos_device.py 的 all_algos.json，画"所有算法排名图"：
  fig_allrank_<模式>.png  生成谱族：每行一个算法，按各族误差中位排序；灰点 = 每族误差，彩色竖条 = 中位，右侧 = 第一名族数
  fig_allrank_beams.png    作者 5 束真实读数：条 = 除 700 nm 外 4 束平均 |峰位误差|，菱形 = 700 nm 那束，空心圈 = 同一批光模拟真实模式
  py -I -X utf8 plot_all_algos.py <results/all_algos 目录>
"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
INK, INK2, GRID, DOT = '#1f1f1e', '#6b6a65', '#e4e3df', '#b9b8b2'
# 五类来源（dataviz 校验：亮度、彩度、色盲区分、正常视觉均 PASS；绿色对比度 2.74 靠每行文字标签补）
CAT = [('本项目 A11', '#2a78d6', ['A11', 'A11_R', 'A11p', 'A11_Rp']),('Uddin 论文算法', '#eb6834', ['paper']),
       ('41 号经典方法', '#1baf7a', ['A1']), ('文献·机器学习', '#8a5cd1', ['XieMLP', 'NSF', 'RF']),
       ('文献·正则化 / 拟合 / 查表', '#b88a1e', ['T0NN', 'T0', 'Ridge', 'LapNN', 'Hao', 'CS', 'Gauss1', 'Pinv', 'LUT'])]
COL = {m: c for _, c, ms in CAT for m in ms}
OURS = ('A11', 'A11_R', 'A11p', 'A11_Rp')
PREF = ('A11p', 'A11', 'A11_Rp', 'A11_R')             # 左图高亮哪个本项目版本：有峰位校正版就用它
# a11_peakfix.py 的结果（同一批读数，旧 A11 各族值与 all_algos.json 逐族相同）：目录, 原键, 新键, 显示名
PEAKFIX = [('a11_peakfix', 'A11', 'A11p', 'A12（A11 + 峰位校正）'), ('a11_peakfix_R', 'A11_R', 'A11_Rp', 'A11_R + 峰位校正')]


def add_peakfix(J, d):
    """把 A12（键名 A11p）/ A11_R + 峰位校正 的各族误差并进 J['fam']；先核对旧版本各族值与 all_algos.json 一致。"""
    for sub, src, key, name in PEAKFIX:
        p = os.path.join(os.path.dirname(os.path.abspath(d)), sub, 'peakfix.json')
        if not os.path.exists(p):
            continue
        P = json.load(open(p, encoding='utf-8')); J['name'][key] = name; J['src'][key] = '本项目'
        for mo, F in P['fam'].items():
            if mo not in J['fam']:
                continue
            for k, v in J['fam'][mo].items():
                q = F[k.replace('留出·', '')]
                assert abs(q['A11'] - v[src]) < 1e-12, (sub, mo, k)
                v[key] = q['A11p']
MODE = {'ideal': '理想读数（无噪声）', 'real': '真实模式读数（器件噪声）', 'realR': '真实模式 + R 标定误差读数'}


def legend(ax, ms):
    h = [Line2D([], [], marker='s', ls='', color=c, ms=8, label=nm) for nm, c, mm in CAT if set(mm) & set(ms)]
    ax.legend(handles=h, fontsize=8, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=5, frameon=False, handletextpad=0.3, columnspacing=1.2)


def lab(J, m):
    return f'{J["name"][m]}  ·  {J["src"][m]}' if m not in OURS + ('A1',) else J['name'][m]


ML = ('XieMLP', 'NSF', 'RF')
TRAD_C = '#b88a1e'


def fam_fig(J, mo, out):
    """同 plot_rank 的画法：左 = 每族一行（按 A11 误差排序），灰点 = 所有算法，标出 A11 / 本族最好 / 本族最差 / 传统算法里最好的；
    中 = 每族最好、最差是谁；右 = 各算法各族误差中位 + 第一名族数。"""
    F = J['fam'][mo]; fams = list(F); ms = [m for m in J['name'] if m in F[fams[0]]]
    a11 = next(m for m in PREF if m in ms)
    trad = [m for m in ms if m not in ML and m not in OURS]
    ext = [m for m in ms if m not in OURS]                 # 外部算法；本项目各版本只和它们比，互不抢第一
    nm = lambda m: J['name'][m]
    med = {m: float(np.median([F[k][m] for k in fams])) for m in ms}
    win = {m: sum(min(ext + [m if m in OURS else a11], key=lambda q: F[k][q]) == m for k in fams) for m in ms}   # 外部算法和高亮版本比
    ms_l = ext + [a11]                                     # 左图只画高亮的那个本项目版本
    fam = sorted(fams, key=lambda k: F[k][a11]); y = np.arange(len(fam))
    fg = plt.figure(figsize=(19, 0.34 * len(fam) + 2.2))
    gs = fg.add_gridspec(2, 4, width_ratios=[3, 3.1, 0.75, 1.5], height_ratios=[1.3, 1], wspace=0.05, hspace=0.3)
    ax = fg.add_subplot(gs[:, 0]); tx = fg.add_subplot(gs[:, 1], sharey=ax); bx = fg.add_subplot(gs[0, 3])
    allv = [F[k][m] for k in fams for m in ms_l]; lo, hi = min(allv) / 1.5, max(allv) * 1.5
    for i, k in enumerate(fam):
        v = {m: F[k][m] for m in ms_l}; b = min(v, key=v.get); w = max(v, key=v.get); t = min(trad, key=v.get)
        ax.plot([v[b], v[w]], [i, i], color=GRID, lw=3, solid_capstyle='round', zorder=1)
        ax.scatter([v[m] for m in ms_l], [i] * len(ms_l), s=10, color=DOT, lw=0, zorder=2)
        ax.scatter(v[w], i, s=40, marker='X', color=INK2, lw=0, zorder=3)
        ax.scatter(v[t], i, s=50, marker='s', color=TRAD_C, edgecolors='white', lw=0.8, zorder=4)
        ax.scatter(v[a11], i, s=70, marker='o', color=COL['A11'], edgecolors='white', lw=1.0, zorder=5)
        if b != a11:
            ax.scatter(v[b], i, s=120, marker='*', color=INK, lw=0, zorder=6)
        win_s = f'★ {nm(a11)} 本族第一' if b == a11 else f'★ {nm(b)} {v[b]:.3f}'
        tx.text(0.02, i, win_s, va='center', fontsize=7.8, color=COL['A11'] if b == a11 else INK)
        tx.text(0.37, i, f'■ {nm(t)} {v[t]:.3f}', va='center', fontsize=7.8, color=INK)
        tx.text(0.74, i, f'✕ {nm(w)}', va='center', fontsize=7.8, color=INK2)
    ax.set_xscale('log'); ax.set_xlim(lo, hi)
    ax.set_yticks(y); ax.set_yticklabels([k.replace('留出·', '') for k in fam], fontsize=8, color=INK); ax.invert_yaxis()
    ax.grid(axis='x', color=GRID, lw=0.6, which='major'); ax.set_axisbelow(True)
    ax.set_xlabel('误差（峰值归一均方根 / 真值极差，对数刻度，越小越好）', fontsize=8, color=INK2)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    h = [Line2D([], [], marker='o', ls='', color=COL['A11'], ms=8, label=nm(a11) if a11 != 'A11' else 'A11（本项目）'),
         Line2D([], [], marker='*', ls='', color=INK, ms=11, label=f'本族最好（不是 {nm(a11)} 时）'),
         Line2D([], [], marker='s', ls='', color=TRAD_C, ms=7, label='传统算法里本族最好的'),
         Line2D([], [], marker='X', ls='', color=INK2, ms=7, label='本族最差'),
         Line2D([], [], marker='o', ls='', color=DOT, ms=4, label='其余算法')]
    ax.legend(handles=h, fontsize=8, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=5, frameon=False, handletextpad=0.3, columnspacing=1.0)
    tx.axis('off'); tx.set_xlim(0, 1)
    tx.text(0.02, -1.0, '本族最好', fontsize=8.5, color=INK); tx.text(0.37, -1.0, '传统算法里最好', fontsize=8.5, color=INK); tx.text(0.74, -1.0, '本族最差', fontsize=8.5, color=INK)

    order = sorted(ms, key=med.get)
    for j, m in enumerate(order):
        c = COL['A11'] if m in OURS else ('#8a5cd1' if m in ML else TRAD_C)
        bx.barh(j, med[m], height=0.66, color=c)
        bx.text(med[m] * 1.08, j, f'{med[m]:.3f}  {win[m]} 族第一' + ('*' if m in OURS and len(set(OURS) & set(ms)) > 1 else ''), va='center', fontsize=7.5, color=INK)
    bx.set_xscale('log'); bx.set_xlim(min(med.values()) / 1.5, max(med.values()) * 12)
    bx.set_yticks(range(len(order))); bx.set_yticklabels([nm(m) for m in order], fontsize=7.5, color=INK); bx.invert_yaxis()
    bx.set_title('各算法在所有族的误差中位（越短越好）', fontsize=9, color=INK); bx.tick_params(axis='x', labelsize=7); bx.tick_params(axis='y', length=0)
    for sp in ('top', 'right'):
        bx.spines[sp].set_visible(False)
    hb = [Line2D([], [], marker='s', ls='', color=c, ms=7, label=l) for l, c in
          (('本项目 A11', COL['A11']), ('文献机器学习', '#8a5cd1'), ('传统算法（含 Uddin 论文算法、41 号）', TRAD_C))]
    bx.legend(handles=hb, fontsize=7.5, loc='upper left', bbox_to_anchor=(0, -0.12), frameon=False, ncol=1, handletextpad=0.3,
              title='* 本项目各版本只和外部算法比，互不抢第一；外部算法的族数\n   是和 ' + nm(a11) + ' 比的；左图只画 ' + nm(a11) if len(set(OURS) & set(ms)) > 1 else None,
              title_fontsize=7.5, alignment='left')
    fg.suptitle(f'Uddin 2024 器件模拟器 · {MODE.get(mo, mo)}：{len(fam)} 族测试谱上每族谁最好（每族 {J["nte"]} 条 × 4 次抽样；行按 {nm(a11)} 误差从小到大排）',
                fontsize=10.5, color=INK, x=0.02, ha='left')
    fg.savefig(os.path.join(out, f'fig_allrank_{mo}.png'), dpi=150, bbox_inches='tight'); plt.close(fg)
    return [(m, med[m], win[m]) for m in order]


def real_fig(J, out):
    lt = np.array(J['lam_test']); k7 = int(np.argmin(np.abs(lt - 700))); keep = [j for j in range(len(lt)) if j != k7]
    R, S = J['real'], J['sim']['real']; ms = [m for m in J['name'] if m in R]
    a4 = {m: float(np.mean(np.abs(np.array(R[m]['dpk'])[keep]))) for m in ms}
    a7 = {m: abs(R[m]['dpk'][k7]) for m in ms}
    sm = {m: float(np.mean(np.abs(np.array(S[m]['dpk'])))) for m in ms if m in S}
    order = sorted(ms, key=a4.get); y = np.arange(len(order))
    fg, ax = plt.subplots(figsize=(11, 0.42 * len(order) + 1.6))
    for i, m in enumerate(order):
        ax.barh(i, a4[m], height=0.56, color=COL[m], zorder=2)
        ax.scatter(a7[m], i, marker='D', s=26, color=INK2, zorder=3)
        if m in sm:
            ax.scatter(sm[m], i, marker='o', s=40, facecolors='none', edgecolors=INK, lw=1.0, zorder=4)
        ax.text(1.005, i, f'{a4[m]:.1f} nm（5 束平均 {np.mean(np.abs(R[m]["dpk"])):.1f}）', transform=ax.get_yaxis_transform(), va='center', fontsize=8.5, color=INK)
    ax.set_xscale('log'); ax.set_xlim(0.1, 400)
    ax.set_yticks(y); ax.set_yticklabels([lab(J, m) for m in order], fontsize=8.5, color=INK); ax.invert_yaxis()
    ax.grid(axis='x', color=GRID, lw=0.6, which='both'); ax.set_axisbelow(True)
    ax.set_xlabel('|峰位误差|（nm，对数刻度，越小越好）', fontsize=8, color=INK2)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    h = [Line2D([], [], marker='s', ls='', color=INK2, ms=8, label=f'条 = 除 700 nm 外 4 束（{", ".join(f"{v:.0f}" for v in lt[keep])} nm）平均'),
         Line2D([], [], marker='D', ls='', color=INK2, ms=5, label='700 nm 那束'),
         Line2D([], [], marker='o', ls='', mfc='none', mec=INK, ms=6, label='同一批光 · 模拟器真实模式 200 次平均（5 束）')]
    ax.legend(handles=h, fontsize=8, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, handletextpad=0.3, columnspacing=1.2)
    fg.suptitle('Uddin 2024 作者的 5 束真实读数（参考 = 商用光谱仪）：各算法峰位误差排名；颜色同谱族图（蓝 = A11，橙 = 论文算法，紫 = 文献机器学习，赭 = 文献其他）',
                fontsize=10, color=INK, x=0.02, ha='left')
    fg.savefig(os.path.join(out, 'fig_allrank_beams.png'), dpi=150, bbox_inches='tight'); plt.close(fg)
    return [(m, a4[m], a7[m], sm.get(m)) for m in order]


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    d = sys.argv[1]; J = json.load(open(os.path.join(d, 'all_algos.json'), encoding='utf-8')); add_peakfix(J, d)
    for mo in J['fam']:
        print(mo, [(m, round(v, 3), w) for m, v, w in fam_fig(J, mo, d)])
    print('real', [(m, round(a, 1), round(b, 1), None if c is None else round(c, 1)) for m, a, b, c in real_fig(J, d)])
