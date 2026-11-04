# -*- coding: utf-8 -*-
"""Uddin 模拟器 · 真实模式读数：A18 与常规算法的指标蛛网图（小多图：每个算法一张，蓝线 = A18），辐条和标尺同 algos_radar.py。
用户 2026-11-03："A18 对比常规算法，输出蛛网图"（参考 Cui 版 A15 蛛网图的版式）。
A18 的指标由 a15_three.py --a16a 32 --a16a-ss --a16a-vscale --a16a-escale --test-seed 4242 --metrics 算出（指标段实际由 run_a18_metrics_only.py 单独重跑，11-03 15:22–16:51）；
常规算法读 all_algos/metrics/metrics.json，A11 读 a11_peakfix/metrics_A11/metrics.json（A12、A15 已按用户要求去掉）。A11_R 已搁置，不画。
种子 4242 是开发种子（A12 起反复看过，不是干净测试集），图的总标题和右下角都标明（A18-R1，2026-11-03）。
  py -I -X utf8 plot_a18_radar.py
输入：A18/results/a18_seed4242/metrics/metrics.json（只读）
输出：同目录 {fig_algos_radar_a18_v3_devseed.png, radar_spokes_a18_v3_devseed.json}；原 v1（fig_algos_radar_a18.png）、v2（_v2）不再写、不覆盖。
"""
import os, json, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(ROOT, '45_Uddin2024_真实R矩阵', 'results')
A18DIR = os.path.join(os.path.dirname(ROOT), 'A18', 'results', 'a18_seed4242', 'metrics')
OUTFIG, OUTJSON = 'fig_algos_radar_a18_v3_devseed.png', 'radar_spokes_a18_v3_devseed.json'      # 新文件名，不覆盖原 v1 / v2（A18-R1）
assert OUTFIG not in ('fig_algos_radar_a18.png', 'fig_algos_radar_a18_v2.png') and OUTJSON not in ('radar_spokes_a18.json', 'radar_spokes_a18_v2.json')
INK, INK2, GRID, C1, CA ='#1f1f1e', '#6b6a65', '#e4e3df', '#d97a2b', '#2a78d6'
plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
ld = lambda *p: json.load(open(os.path.join(*p), encoding='utf-8'))
J18 = ld(A18DIR, 'metrics.json'); J11 = ld(BASE, 'a11_peakfix', 'metrics_A11', 'metrics.json')            # A11 = 两支融合（峰位校正之前），run_a11_metrics.py 用同一协议算的；A12、A15 的图已按用户要求去掉
JA = ld(BASE, 'all_algos', 'metrics', 'metrics.json')
NAME = dict(JA['name'], A11='A11', A18='A18'); SRC = dict(JA['src'], A11='本项目', A18='本项目')

# (组, 短名, 标尺 (差端, 好端), 对数, 格式)
SPK = [('峰位', '窄窗最好 20 nm', (10, 0.1), 1, '{:.2f}'), ('峰位', '全波段平均', (10, 0.1), 1, '{:.2f}'), ('峰位', '全波段 RMSD', (10, 0.1), 1, '{:.2f}'),
       ('峰位', '90 % 分位', (10, 0.1), 1, '{:.2f}'),
       ('双峰', '8 nm 线·间隔 20 nm', (0, 1), 0, '{:.2f}'), ('双峰', '8 nm 线·大间隔可靠度', (0, 1), 0, '{:.2f}'), ('双峰', '强弱 2:1·间隔 24 nm', (0, 1), 0, '{:.2f}'),
       ('双峰', '16 nm 线·间隔 32 nm', (0, 1), 0, '{:.2f}'), ('线宽', '8 nm 线重建 / 真值', (4, 1), 2, '{:.2f}'),
       ('PSNR', '最好 10 %', (10, 40), 0, '{:.1f}'), ('PSNR', '中位', (10, 40), 0, '{:.1f}'), ('PSNR', '最差 10 %', (10, 40), 0, '{:.1f}'),
       ('抗噪', '20 % 噪声 MSE', (0.1, 0.001), 1, '{:.3f}'), ('抗噪', '所需最低信噪比', (40, -20), 0, '{:.0f}'), ('波段', '≤2 nm 连续段', (0, 292), 0, '{:.0f}')]


def spokes(d):
    """同 algos_radar.spokes（内区 520–780 nm）。"""
    S = d['scan']['8.0']; c = np.array(S['cen']); p = np.array(S['pk_med']); q = p[(c >= 520) & (c <= 780)]
    n = int(round(20 / (c[1] - c[0]))) + 1; win = min(p[i:i + n].mean() for i in range(len(c) - n + 1))
    fw = float(np.nanmedian(np.array(S['fwhm_rec'])[(c >= 520) & (c <= 780)]))
    R = d['res']; fr = lambda cfg, dl: R[cfg]['frac'][str(dl)] if str(dl) in R[cfg]['frac'] else R[cfg]['frac'][str(float(dl))]
    rs = [fr('8|1', 20), float(np.mean([fr('8|1', dl) for dl in (32, 40, 48, 64)])), fr('8|0.5', 24), fr('16|1', 32)]
    sp = d['full']['sample_pct']; ps = [-20 * np.log10(sp[k]) for k in ('10', '50', '90')]
    ks = sorted(d['noise'], key=float); snr = np.array([d['noise'][k]['snr'] for k in ks]); med = np.array([d['noise'][k]['med'] for k in ks])
    pkn = np.array([d['noise'][k]['pk']['窄线 4–10 nm'] for k in ks])
    e20 = np.exp(np.interp(np.log(5.0), np.log(snr), np.log(med)))
    ok = pkn <= 15; i = next((i for i in range(len(ks)) if ok[i:].all()), None)
    if i is None:
        s15 = None
    elif i == 0:
        s15 = snr[0]
    else:
        f = (np.log(15) - np.log(pkn[i - 1])) / (np.log(pkn[i]) - np.log(pkn[i - 1])); s15 = np.exp(np.log(snr[i - 1]) + f * (np.log(snr[i]) - np.log(snr[i - 1])))
    b2 = S['band']['2.0']
    return [win, q.mean(), np.sqrt((q ** 2).mean()), np.percentile(q, 90)] + rs + [fw / 8.0] + ps + [e20 ** 2, None if s15 is None else 20 * np.log10(s15), 0 if b2 is None else b2[1] - b2[0]]


def score(v, lim, lg):
    if v is None:
        return 0.0
    a, b = lim
    if lg == 2:
        return float(np.clip(1 - abs(np.log(max(v, 1e-9))) / np.log(a), 0, 1.04))
    f = (np.log(max(v, 1e-9)) - np.log(a)) / (np.log(b) - np.log(a)) if lg else (v - a) / (b - a)
    return float(np.clip(f, 0, 1.04))


V = {'A18': spokes(J18), 'A11': spokes(J11)}
V.update({m: spokes(d) for m, d in JA['M'].items() if 'full' in d and m != 'A11_R'})
SCR = {m: [score(v, s[2], s[3]) for v, s in zip(vs, SPK)] for m, vs in V.items()}
order = sorted([m for m in V if m != 'A18'], key=lambda m: -np.mean(SCR[m]))
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
    if k == 0:                                                                     # 图例格：A18 自己 + 辐条编号
        poly(ax, SCR['A18'], CA, True, 2, 4)
        for i, (t, s) in enumerate(zip(th, SPK)):
            ha = 'center' if abs(np.cos(t)) < 0.2 else ('left' if np.cos(t) > 0 else 'right')
            ax.text(t, 1.18, f'{i + 1} {s[0]}·{s[1]}', ha=ha, va='center', fontsize=6.6, color=INK)
        ax.set_title(f'A18（本项目）  平均分 {np.mean(SCR["A18"]):.2f}', fontsize=9.5, color=INK, pad=26)
        continue
    m = order[k - 1]; poly(ax, SCR['A18'], CA, False, 1.0, 3); poly(ax, SCR[m], C1, True, 1.8, 5)
    for i, t in enumerate(th):
        ax.text(t, 1.14, str(i + 1), ha='center', va='center', fontsize=6.5, color=INK2)
    ax.set_title(f'{NAME[m]}  平均分 {np.mean(SCR[m]):.2f}\n{SRC[m]}', fontsize=9, color=INK, pad=14, linespacing=1.3)
fg.text(0.99, 0.004, '开发种子 4242（A12 起反复看过，非干净测试集；最终数字须换新种子）', fontsize=11, color='#b3261e', va='bottom', ha='right', fontweight='bold')   # 右下角，避开左上长标题
fg.text(0.01, 0.995, 'Uddin 2024 器件（模拟器）· 同一份真实模式读数（P0，测试种子 4242 = 开发种子）下 A18 与各重建算法的指标：橙色 = 该算法，蓝线 = A18；越靠外越好', fontsize=12.5, color=INK, va='top')
fg.text(0.01, 0.975, '标尺（圆心 → 外圈）：峰位 10 → 0.1 nm（对数）；双峰 = 在给定间隔下"分开"的比例 0 → 100 %（真值本身 12 nm 起就能分开）；线宽 = 重建半高宽 / 真值，1 为满分，偏大偏小 4 倍记 0；PSNR 10 → 40 dB；'
        'MSE 0.1 → 0.001（对数）；所需信噪比 40 → −20 dB；≤ 2 nm 连续段 0 → 292 nm（= 扫描全程 504–796 nm）。按平均分排序（平均分只为排版，不作总分）。', fontsize=8.2, color=INK2, va='top')
fg.tight_layout(rect=(0, 0.02, 1, 0.955)); f = os.path.join(A18DIR, OUTFIG); fg.savefig(f, dpi=150); print(f)

hdr = ['方法'] + [f'{i + 1}' for i in range(N)] + ['平均分']
print('| ' + ' | '.join(hdr) + ' |'); print('|' + '---|' * len(hdr))
for m in ['A18'] + order:
    print('| ' + NAME[m] + ' | ' + ' | '.join('—' if v is None else s[4].format(v) for v, s in zip(V[m], SPK)) + f' | {np.mean(SCR[m]):.2f} |')
json.dump({m: [None if v is None else float(v) for v in vs] for m, vs in V.items()}, open(os.path.join(A18DIR, OUTJSON), 'w', encoding='utf-8'), ensure_ascii=False)
