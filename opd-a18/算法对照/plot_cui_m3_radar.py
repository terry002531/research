# -*- coding: utf-8 -*-
"""Cui 2025 器件 · M3 读数：各重建算法的指标蛛网图（小多图：每个算法一张，蓝线 = A15），辐条和标尺同 algos_radar.py（Uddin 版），
只改两处：峰位 / 线宽的"内区"取 520–700 nm；"≤ 2 nm 连续段"的满分 = 单线扫描全程 504–716 nm（212 nm）。
读 cui_m3_metrics.py 三组的 metrics_{a15,a12,algos}.json；A11 两组各算了一次（a15 组 = 两支融合），辐条值核对一致。
  py -I -X utf8 plot_cui_m3_radar.py [--quick]
输出：46_Cui2025_真实R矩阵/results/metrics_M3(_quick)/{fig_algos_radar_cui_m3.png, radar_spokes.json}
"""
import os, sys, json, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
Q = '_quick' if '--quick' in sys.argv else ''
RES = os.path.join(ROOT, '46_Cui2025_真实R矩阵', 'results'); OUT = os.path.join(RES, 'metrics_M3' + Q)
INK, INK2, GRID, C1, CA = '#1f1f1e', '#6b6a65', '#e4e3df', '#d97a2b', '#2a78d6'
plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
JG = {g: json.load(open(os.path.join(OUT, f'metrics_{g}.json'), encoding='utf-8')) for g in ('a15', 'a12', 'algos')}
XJ = json.load(open(os.path.join(RES, 'all_algos_M3', 'all_algos.json'), encoding='utf-8'))
NAME = dict(XJ['name'], A15='A15（三支融合）', A12='A12', A11='A11'); SRC = dict(XJ['src'], A15='本项目', A12='本项目', A11='本项目')
LO, HI = JG['a15']['inner']; CEN = np.array(JG['a15']['M']['A15']['scan']['8.0']['cen']); SPAN = float(CEN[-1] - CEN[0])

# (组, 短名, 标尺 (差端, 好端), 对数, 格式)
SPK = [('峰位', '窄窗最好 20 nm', (10, 0.1), 1, '{:.2f}'), ('峰位', '内区平均', (10, 0.1), 1, '{:.2f}'), ('峰位', '内区 RMSD', (10, 0.1), 1, '{:.2f}'),
       ('峰位', '90 % 分位', (10, 0.1), 1, '{:.2f}'),
       ('双峰', '8 nm 等高 ≥50 %', (64, 2), 1, '{:.0f}'), ('双峰', '8 nm 等高 ≥90 %', (64, 2), 1, '{:.0f}'), ('双峰', '强度比 0.5', (64, 2), 1, '{:.0f}'),
       ('双峰', '16 nm 线', (64, 2), 1, '{:.0f}'), ('线宽', '8 nm 线重建半高宽', (32, 1), 1, '{:.1f}'),
       ('PSNR', '最好 10 %', (10, 40), 0, '{:.1f}'), ('PSNR', '中位', (10, 40), 0, '{:.1f}'), ('PSNR', '最差 10 %', (10, 40), 0, '{:.1f}'),
       ('抗噪', '20 % 噪声 MSE', (0.1, 0.001), 1, '{:.3f}'), ('抗噪', '所需最低信噪比', (40, -20), 0, '{:.0f}'), ('波段', '≤2 nm 连续段', (0, SPAN), 0, '{:.0f}')]


def spokes(d):
    """同 algos_radar.spokes，内区换成 LO–HI。"""
    S = d['scan']['8.0']; c = np.array(S['cen']); p = np.array(S['pk_med']); q = p[(c >= LO) & (c <= HI)]
    n = int(round(20 / (c[1] - c[0]))) + 1; win = min(p[i:i + n].mean() for i in range(len(c) - n + 1))
    fw = float(np.nanmedian(np.array(S['fwhm_rec'])[(c >= LO) & (c <= HI)]))
    R = d['res']; rs = [R['8|1']['res']['0.5'], R['8|1']['res']['0.9'], R['8|0.5']['res']['0.5'], R['16|1']['res']['0.5']]
    sp = d['full']['sample_pct']; ps = [-20 * np.log10(sp[k]) for k in ('10', '50', '90')]
    ks = sorted(d['noise'], key=float); snr = np.array([d['noise'][k]['snr'] for k in ks]); med = np.array([d['noise'][k]['med'] for k in ks])
    pkn = np.array([d['noise'][k]['pk']['窄线 4–10 nm'] for k in ks])
    e20 = np.exp(np.interp(np.log(5.0), np.log(snr), np.log(med)))
    ok = pkn <= 15; i = next((i for i in range(len(ks)) if ok[i:].all()), None)       # 从这一档起往上都 ≤ 15 nm
    if i is None:
        s15 = None
    elif i == 0:
        s15 = snr[0]
    else:
        f = (np.log(15) - np.log(pkn[i - 1])) / (np.log(pkn[i]) - np.log(pkn[i - 1])); s15 = np.exp(np.log(snr[i - 1]) + f * (np.log(snr[i]) - np.log(snr[i - 1])))
    b2 = S['band']['2.0']
    return [win, q.mean(), np.sqrt((q ** 2).mean()), np.percentile(q, 90)] + rs + [fw] + ps + [e20 ** 2, None if s15 is None else 20 * np.log10(s15), 0 if b2 is None else b2[1] - b2[0]]


def score(v, lim, lg):
    if v is None:
        return 0.0
    a, b = lim; f = (np.log(max(v, 1e-9)) - np.log(a)) / (np.log(b) - np.log(a)) if lg else (v - a) / (b - a)
    return float(np.clip(f, 0, 1.04))


V = {'A15': spokes(JG['a15']['M']['A15']), 'A12': spokes(JG['a12']['M']['A12']), 'A11': spokes(JG['a12']['M']['A11'])}
v11 = spokes(JG['a15']['M']['A11'])                                               # 同一份读数、两个进程各解一次 A11
dif = max(abs((a or 0) - (b or 0)) for a, b in zip(v11, V['A11'])); print(f'A11 两组辐条值最大差 {dif:.2e}')
V.update({m: spokes(d) for m, d in JG['algos']['M'].items() if 'full' in d})
SCR = {m: [score(v, s[2], s[3]) for v, s in zip(vs, SPK)] for m, vs in V.items()}
order = sorted([m for m in V if m != 'A15'], key=lambda m: -np.mean(SCR[m]))
N = len(SPK); th = np.pi / 2 - np.arange(N) * 2 * np.pi / N


def base(ax):
    ax.set_ylim(0, 1.04); ax.axis('off')
    for r in (0.5, 1.0):
        ax.plot(np.linspace(0, 2 * np.pi, 300), [r] * 300, color=GRID, lw=0.7, zorder=0)
    for t in th:
        ax.plot([t, t], [0, 1], color=GRID, lw=0.6, zorder=0)


def poly(ax, s, col, fill, lw, z):
    ax.plot(np.r_[th, th[0]], np.r_[s, s[0]], color=col, lw=lw, zorder=z)
    if fill:
        ax.fill(np.r_[th, th[0]], np.r_[s, s[0]], color=col, alpha=0.15, lw=0, zorder=z - 1)


nc = 5; nr = int(np.ceil((len(order) + 1) / nc))
fg = plt.figure(figsize=(4.1 * nc, 4.25 * nr + 0.9))
for k in range(len(order) + 1):
    ax = fg.add_subplot(nr, nc, k + 1, polar=True); base(ax)
    if k == 0:                                                                     # 图例格：A15 自己 + 辐条编号
        poly(ax, SCR['A15'], CA, True, 2, 4)
        for i, (t, s) in enumerate(zip(th, SPK)):
            ha = 'center' if abs(np.cos(t)) < 0.2 else ('left' if np.cos(t) > 0 else 'right')
            ax.text(t, 1.18, f'{i + 1} {s[0]}·{s[1]}', ha=ha, va='center', fontsize=6.6, color=INK)
        ax.set_title(f'A15（本项目）  平均分 {np.mean(SCR["A15"]):.2f}', fontsize=9.5, color=INK, pad=26)
        continue
    m = order[k - 1]; poly(ax, SCR['A15'], CA, False, 1.0, 3); poly(ax, SCR[m], C1, True, 1.8, 5)
    for i, t in enumerate(th):
        ax.text(t, 1.14, str(i + 1), ha='center', va='center', fontsize=6.5, color=INK2)
    ax.set_title(f'{NAME[m]}  平均分 {np.mean(SCR[m]):.2f}\n{SRC[m]}', fontsize=9, color=INK, pad=14, linespacing=1.3)
fg.text(0.01, 0.995, 'Cui 2025 器件 · 同一份 M3 读数（recon 档白噪声 + 逐 Vgs 帧增益漂移，a = 1，P0）下各重建算法的指标：橙色 = 该算法，蓝线 = A15；越靠外越好',
        fontsize=12.5, color=INK, va='top')
fg.text(0.01, 0.975, f'标尺（圆心 → 外圈）：峰位 10 → 0.1 nm（对数，内区 {LO:.0f}–{HI:.0f} nm）；双峰 64 → 2 nm（对数，达不到 64 nm 记 0）；线宽 32 → 1 nm（对数）；PSNR 10 → 40 dB；'
        f'MSE 0.1 → 0.001（对数）；所需信噪比 40 → −20 dB；≤ 2 nm 连续段 0 → {SPAN:.0f} nm（= 扫描全程）。按平均分排序（平均分只为排版，不作总分）。', fontsize=8.2, color=INK2, va='top')
fg.tight_layout(rect=(0, 0, 1, 0.955)); f = os.path.join(OUT, 'fig_algos_radar_cui_m3.png'); fg.savefig(f, dpi=150); print(f)

hdr = ['方法'] + [f'{i + 1}' for i in range(N)] + ['平均分']
print('| ' + ' | '.join(hdr) + ' |'); print('|' + '---|' * len(hdr))
for m in ['A15'] + order:
    print('| ' + NAME[m] + ' | ' + ' | '.join('—' if v is None else s[4].format(v) for v, s in zip(V[m], SPK)) + f' | {np.mean(SCR[m]):.2f} |')
json.dump({m: [None if v is None else float(v) for v in vs] for m, vs in V.items()}, open(os.path.join(OUT, 'radar_spokes.json'), 'w', encoding='utf-8'), ensure_ascii=False)
