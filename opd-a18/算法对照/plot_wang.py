# -*- coding: utf-8 -*-
"""画 Wang 2024 器件（47 号）· 真实器件模式读数上的全算法对比：A15、A12、A11、Xie MLP、Wang NSF（重训）和传统方法。
读 47_Wang2024_真实R矩阵/results/ 下 a15/、a12/、all_algos/ 的 json + raw.pkl（三个脚本用同一份存盘测试读数，只有重建算法不同），
以及 a11/a11.json 里论文 NSF 在实测读数上的结果（读数来源不同，只作量级参考）。
  fig_rank_wang.png    三栏排名：生成谱族形状误差各族中位 / Fig 3 测试光 4 组平均形状误差 / Fig 3 单峰组 |峰位误差| 中位
  fig_table_wang.png   误差表：行 = 26 族（+ 各族中位），列 = 各方法，格内 = 该族形状误差平均（40 条 × 4 次）
  fig_fig3_wang.png    Fig 3 的 25 条测试光重建（测试器件 0）：真值、A15、A12、Xie MLP、Wang NSF（重训）、传统里最好的
  控制台：A15 对 A12 / Xie MLP 的配对比较（同一条测试谱同一次读数），bootstrap 95 % 区间
  py -I -X utf8 plot_wang.py
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
RES = os.path.join(ROOT, '47_Wang2024_真实R矩阵', 'results')
TEST = os.environ.get('WTEST') == '1'                                             # 只测脚本：读 --quick / SMOKE 试跑的结果
DIRS = dict(a15='a15_quick', a12='a12_quick', all_algos='all_algos_smoke', a11='a11_quick') if TEST else {}
OUT = os.path.join(RES, 'compare_test' if TEST else 'compare'); os.makedirs(OUT, exist_ok=True)
ld = lambda d, f: json.load(open(os.path.join(RES, DIRS.get(d, d), f), encoding='utf-8'))
lp = lambda d: pickle.load(open(os.path.join(RES, DIRS.get(d, d), 'raw.pkl'), 'rb'))
J15, J12, JA, J11 = ld('a15', 'a15.json'), ld('a12', 'a12.json'), ld('all_algos', 'all_algos.json'), ld('a11', 'a11.json')
R15, R12, RA = lp('a15'), lp('a12'), lp('all_algos')
F3 = np.load(os.path.join(RES, 'testset_fig3_quick.npz' if TEST else 'testset_fig3.npz'), allow_pickle=True)
WL = 400.0 + 4.0 * np.arange(F3['truth'].shape[1]); BM = (WL >= 400) & (WL <= 900); assert BM.sum() == 126   # 41 号链 4 nm 网格（400–1100 nm，= ladder.WLg）
wl = WL[BM]; GRP = [str(g) for g in F3['group']]; NAMEF3 = [str(n) for n in F3['name']]

# 方法 → (来源 json, raw, 键)
SRC = dict(A15=(J15, R15, 'A15'), A12=(J12, R12, 'A12'), A11=(J12, R12, 'A11'))
TRAD = ['A1', 'T0NN', 'T0', 'Ridge', 'LapNN', 'Hao', 'CS', 'Gauss1', 'Pinv']
for m in ['XieMLP', 'NSF'] + TRAD + ['LUT', 'RF']:
    SRC[m] = (JA, RA, m)
SPEC = [m for m in SRC if m not in ('LUT', 'RF')]                                  # 出整条谱的方法
LAB = dict(A15='A15', A12='A12', A11='A11', XieMLP='Xie MLP', NSF='Wang NSF（重训）', **{m: JA['name'][m] for m in TRAD + ['LUT', 'RF']})
C15 = '#d6402a'
COL = dict(PA.COL, A15=C15, A12=PA.COL['A11'], A11='#7fb0ea')
fams = [k for k in J15['fam'] if k in JA['fam'] and k in J12['fam']]
assert TEST or len(fams) == len(J15['fam']) == len(JA['fam']) == 26, (len(fams), len(JA['fam']))
fe = lambda m, k: SRC[m][0]['fam'][k][SRC[m][2]]
f3s = lambda m: SRC[m][0]['fig3'][SRC[m][2]]['summary']
PAP = J11['fig3']['论文 NSF（实测读数）']['summary']

# ------------------------------------------------------------------ 0. 读数一致性核对
d_two = max(abs(J15['fam'][k]['two'] - J12['fam'][k]['A11']) for k in fams)
d_a1 = max(abs(JA['fam'][k]['A1'] - J11['fam']['device'][k]['A1']) for k in fams) if 'A1' in J11['fam']['device'][fams[0]] else float('nan')
d_f3 = float(np.abs(R15['f3err']['two'] - R12['f3err']['A11']).max())
print(f'核对：A15 脚本的两支融合 vs A12 脚本的 A11 各族最大差 {d_two:.2e}，Fig 3 逐样本最大差 {d_f3:.2e}；all_algos 的 A1 vs a11.json {d_a1:.2e}')
print(f'q 小网 λ：A12 {J12["lam"]}，A15 {J15["lam"]}')

# ------------------------------------------------------------------ 1. 三栏排名
med = {m: float(np.median([fe(m, k) for k in fams])) for m in SPEC}
a4 = {m: f3s(m)['_4组平均'] for m in SPEC}
pk = {m: f3s(m)['_峰位中位'] for m in SRC}
pr = {m: f3s(m)['_峰位RMS'] for m in SRC}
panels = [('生成谱族 26 族 · 形状误差各族中位', med, None),
          ('Fig 3 测试光 · 形状误差 4 组平均', a4, PAP['_4组平均']),
          ('Fig 3 单峰组（窄线 + 逐 nm 线）· |峰位误差| 中位（nm）', pk, PAP['_峰位中位'])]
fig, axs = plt.subplots(1, 3, figsize=(17, 6.4))
for ax, (tt, v, ref) in zip(axs, panels):
    ms = sorted(v, key=v.get)[::-1]
    for i, m in enumerate(ms):
        ax.barh(i, v[m], height=0.62, color=COL[m], zorder=2)
        ax.text(v[m], i, f' {v[m]:.3f}' if v is not pk else f' {v[m]:.2f}', va='center', fontsize=8, color=PA.INK)
    ax.set_yticks(range(len(ms))); ax.set_yticklabels([LAB[m] for m in ms], fontsize=8.8)
    for t, m in zip(ax.get_yticklabels(), ms):
        if m == 'A15':
            t.set_color(C15); t.set_fontweight('bold')
    if ref is not None:
        ax.axvline(ref, color=PA.INK2, ls='--', lw=1.0, zorder=3)
        ax.text(ref, -0.9, ' 虚线 = 论文 NSF · 实测读数（读数来源不同，仅参考）', fontsize=7.2, color=PA.INK2, va='center')
    ax.set_title(tt, fontsize=10, loc='left', color=PA.INK); ax.grid(axis='x', color=PA.GRID, lw=0.6, zorder=0)
    if v is pk:                                                                   # 峰位误差跨 3 个量级：对数轴
        ax.set_xscale('log'); ax.set_xlim(min(min(v.values()), ref) * 0.6, max(v.values()) * 2.5)
    else:
        ax.set_xlim(0, max(v.values()) * 1.18)
    ax.set_ylim(-1.4, len(ms) - 0.5); ax.tick_params(axis='x', labelsize=8)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
h = [Line2D([], [], color=c, lw=7, label=l) for l, c in
     (('A15（本项目）', C15), ('A12 / A11（本项目）', PA.COL['A11']), ('文献机器学习（Xie MLP、Wang NSF、随机森林）', PA.COL['XieMLP']),
      ('41 号经典方法', PA.COL['A1']), ('文献正则化 / 拟合 / 查表', PA.COL['T0']))]
fig.legend(handles=h, loc='lower center', ncol=5, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle('Wang 2024 器件（47 号）· 真实器件模式读数：各重建算法对比（越短越好）\n'
             '读数含温漂、增益、AR(1) 漂移、偏压偏移抖动、暗电流、散粒 + 读出噪声；要训练的方法都在训练器件读数上重训，测试器件、测试谱与训练不重叠；'
             '查表 / 随机森林只给峰位，只进第三栏',
             fontsize=10.5, x=0.01, ha='left', color=PA.INK)
plt.tight_layout(rect=(0, 0.04, 1, 0.93)); f1 = os.path.join(OUT, 'fig_rank_wang.png'); fig.savefig(f1, dpi=140, bbox_inches='tight'); plt.close(fig)

# ------------------------------------------------------------------ 2. 误差表
COLS = ['A15', 'A12', 'A11', 'XieMLP', 'NSF'] + sorted(TRAD, key=med.get)
rows = sorted(fams, key=lambda k: fe('A15', k))
V = np.array([[fe(m, k) for m in COLS] for k in rows]); V = np.vstack([V, np.median(V, 0)]); rl = rows + ['各族中位']
fig, ax = plt.subplots(figsize=(1.05 * len(COLS) + 2.6, 0.34 * len(rl) + 1.6))
ax.imshow(V, cmap='YlOrBr', norm=LogNorm(vmin=max(V.min(), 0.01), vmax=V.max()), aspect='auto')
for i in range(len(rl)):
    j0 = int(np.argmin(V[i]))
    for j in range(len(COLS)):
        v = V[i, j]
        ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=7.6, color='white' if v > 0.25 else PA.INK, fontweight='bold' if j == j0 else 'normal')
    ax.add_patch(plt.Rectangle((j0 - 0.5, i - 0.5), 1, 1, fill=False, ec=PA.INK, lw=1.6))
ax.axhline(len(rows) - 0.5, color=PA.INK, lw=1.2); ax.axvline(4.5, color=PA.INK, lw=1.0)
ax.set_xticks(range(len(COLS))); ax.set_xticklabels([LAB[m] for m in COLS], rotation=40, ha='right', fontsize=8.5)
ax.set_yticks(range(len(rl))); ax.set_yticklabels(rl, fontsize=8.5); ax.tick_params(length=0)
for t, m in zip(ax.get_xticklabels(), COLS):
    if m == 'A15':
        t.set_color(C15); t.set_fontweight('bold')
nb = {m: int(sum(np.argmin(V[i]) == j for i in range(len(rows)))) for j, m in enumerate(COLS)}
ax.set_title('Wang 2024 器件 · 真实器件模式读数：各族形状误差（每族 40 条 × 4 次平均；峰值归一均方根 / 真值极差，越小越好）\n'
             '框 = 该行最好；行按 A15 从小到大排；竖线左边是本项目和文献机器学习方法，右边是传统方法（按各族中位排）\n'
             '各方法拿第一的族数：' + '，'.join(f'{LAB[m]} {nb[m]}' for m in COLS if nb[m]), fontsize=9.5, loc='left')
for sp in ax.spines.values():
    sp.set_visible(False)
plt.tight_layout(); f2 = os.path.join(OUT, 'fig_table_wang.png'); fig.savefig(f2, dpi=140, bbox_inches='tight'); plt.close(fig)

# ------------------------------------------------------------------ 3. Fig 3 测试光重建
nrm = lambda x: x / x.max() if x.max() > 0 else x
ex = lambda x, s: float(np.sqrt(np.mean((nrm(np.clip(x[BM], 0, None)) - nrm(s[BM])) ** 2)) / np.ptp(nrm(s[BM])))   # = LD.band_nrmse（负值先截 0）
bt = min(TRAD, key=a4.get); SH = ['A15', 'A12', 'XieMLP', 'NSF', bt]
X3 = {m: SRC[m][1]['f3demo'][SRC[m][2]] for m in SH}                            # (NW, 25)，测试器件 0
chk = max(abs(ex(X3['A15'][:, k], F3['truth'][k]) - R15['f3err']['A15'][0, k]) for k in range(25))
nc = 5; nr = 5
fig, axs = plt.subplots(nr, nc, figsize=(3.4 * nc, 2.4 * nr), sharex=True)
for k, ax in enumerate(axs.flat):
    s = F3['truth'][k]
    ax.plot(wl, nrm(s[BM]), color=PA.INK, lw=2.2, zorder=6)
    ax.plot(wl, nrm(F3['paper'][k][BM]), color=PA.DOT, lw=1.0, ls='--', zorder=1)
    for z, m in enumerate(SH[::-1]):
        ax.plot(wl, nrm(X3[m][BM, k]), color=COL[m], lw=1.5 if m == 'A15' else 1.1, ls=(0, (4, 2)) if m == 'NSF' else '-', zorder=2 + z)
    ax.set_ylim(-0.25, 1.3); ax.set_xlim(400, 900); ax.grid(color=PA.GRID, lw=0.5); ax.tick_params(labelsize=7)
    ax.set_title(f'{GRP[k]} · {NAMEF3[k]}', fontsize=8.6, color=PA.INK, pad=14)
    ax.text(0.5, 1.02, ' · '.join(f'{LAB[m].split("（")[0]} {ex(X3[m][:, k], s):.3f}' for m in SH),
            transform=ax.transAxes, ha='center', va='bottom', fontsize=5.6, color=PA.INK2)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
for ax in axs[-1]:
    ax.set_xlabel('波长（nm）', fontsize=8)
h = [Line2D([], [], color=PA.INK, lw=2.2, label='真值'), Line2D([], [], color=C15, lw=1.5, label='A15'),
     Line2D([], [], color=COL['A12'], lw=1.1, label='A12'), Line2D([], [], color=COL['XieMLP'], lw=1.1, label='Xie MLP'),
     Line2D([], [], color=COL['NSF'], lw=1.1, ls=(0, (4, 2)), label='Wang NSF（重训）'), Line2D([], [], color=COL[bt], lw=1.1, label=f'{LAB[bt]}（传统里 4 组平均最好）'),
     Line2D([], [], color=PA.DOT, lw=1.0, ls='--', label='论文 NSF · 实测读数（读数来源不同）')]
fig.legend(handles=h, loc='upper left', bbox_to_anchor=(0.01, 0.975), ncol=7, fontsize=8.5, frameon=False)
fig.suptitle('Wang 2024 器件 · 论文 Fig 3 的 25 条测试光：真实器件模式读数（测试器件 0，读一次）的重建；曲线峰值归一\n'
             '格顶小字 = 这一条的形状误差（峰值归一均方根 / 真值极差）', fontsize=10.5, color=PA.INK, x=0.01, y=1.0, ha='left')
plt.tight_layout(rect=(0, 0, 1, 0.955)); f3 = os.path.join(OUT, 'fig_fig3_wang.png'); fig.savefig(f3, dpi=140, bbox_inches='tight'); plt.close(fig)
print(f1, f2, f3, sep='\n'); print(f'示例核对：A15 格顶误差 vs raw f3err 最大差 {chk:.1e}')

# ------------------------------------------------------------------ 4. 配对比较
rng = np.random.default_rng(0); NB = 4000


def paired(a, b):
    """a 对 b：生成谱族（每条谱先对 4 次读数平均）和 Fig 3（每只测试器件先对 4 组平均）；区间 = 在族内重抽谱 / 重抽器件的 bootstrap。"""
    A = {k: SRC[a][1]['famerr'][SRC[a][2]][k].mean(0) for k in fams}; B = {k: SRC[b][1]['famerr'][SRC[b][2]][k].mean(0) for k in fams}
    rel = np.array([A[k].mean() / B[k].mean() - 1 for k in fams])
    win = int((rel < -0.01).sum()); lose = int((rel > 0.01).sum())
    bs = []
    for _ in range(NB):
        r = []
        for k in fams:
            i = rng.integers(0, len(A[k]), len(A[k])); r.append(A[k][i].mean() / B[k][i].mean() - 1)
        bs.append(np.median(r))
    if len(SRC[a][1]['f3err'][SRC[a][2]]) != len(SRC[b][1]['f3err'][SRC[b][2]]):              # 只在 WTEST（SMOKE 子集）时会发生
        print(f'\n{LAB[a]} 对 {LAB[b]}：Fig 3 测试器件数不同，跳过'); return
    ga = np.array([[SRC[a][1]['f3err'][SRC[a][2]][:, np.array(GRP) == G].mean(1)] for G in PA_F3]).squeeze(1).mean(0)
    gb = np.array([[SRC[b][1]['f3err'][SRC[b][2]][:, np.array(GRP) == G].mean(1)] for G in PA_F3]).squeeze(1).mean(0)
    i = rng.integers(0, len(ga), (NB, len(ga))); b3 = ga[i].mean(1) / gb[i].mean(1) - 1
    sa = np.abs(SRC[a][1]['f3dpk'][SRC[a][2]]); sb = np.abs(SRC[b][1]['f3dpk'][SRC[b][2]]); m_ = ~np.isnan(sa[0])
    pa_, pb_ = np.median(sa[:, m_], 1), np.median(sb[:, m_], 1); bp = pa_[i].mean(1) - pb_[i].mean(1)
    print(f'\n{LAB[a]} 对 {LAB[b]}：')
    print(f'  生成谱族：各族相对差中位 {np.median(rel) * 100:+.1f} %（95 % 区间 {np.percentile(bs, 2.5) * 100:+.1f} ~ {np.percentile(bs, 97.5) * 100:+.1f} %），'
          f'好 > 1 % 的 {win} 族、差 > 1 % 的 {lose} 族；最好 {fams[int(np.argmin(rel))]} {rel.min() * 100:+.1f} %，最差 {fams[int(np.argmax(rel))]} {rel.max() * 100:+.1f} %')
    print(f'  Fig 3 4 组平均：{ga.mean():.4f} vs {gb.mean():.4f}，相对 {(ga.mean() / gb.mean() - 1) * 100:+.1f} %（95 % 区间 {np.percentile(b3, 2.5) * 100:+.1f} ~ {np.percentile(b3, 97.5) * 100:+.1f} %）')
    print(f'  Fig 3 单峰组 |峰位误差|（每只器件取中位再平均）：{pa_.mean():.2f} vs {pb_.mean():.2f} nm，差 {bp.mean():+.2f} nm（95 % 区间 {np.percentile(bp, 2.5):+.2f} ~ {np.percentile(bp, 97.5):+.2f}）')
    for k in ('窄线 4–10 nm', '单峰 15–60 nm'):
        print(f'  {k} |峰位误差| 中位：{SRC[a][0]["fampk"][k][SRC[a][2]]:.3f} vs {SRC[b][0]["fampk"][k][SRC[b][2]]:.3f} nm')


PA_F3 = ['LED', '双峰', '窄线', '逐 nm 线']
for a, b in (('A15', 'A12'), ('A15', 'XieMLP'), ('A12', 'XieMLP'), ('A15', 'NSF')):
    paired(a, b)

print('\n总表（形状误差各族中位 / Fig 3 4 组平均 / Fig 3 单峰 |峰位| 中位 / RMS）：')
for m in sorted(SRC, key=lambda m: med.get(m, 9)):
    print(f'  {LAB[m]:<24s} {med.get(m, float("nan")):8.4f} {a4.get(m, float("nan")):8.4f} {pk[m]:7.2f} {pr[m]:7.2f}')
print(f'  {"论文 NSF · 实测读数（参考）":<24s} {"—":>8s} {PAP["_4组平均"]:8.4f} {PAP["_峰位中位"]:7.2f} {PAP["_峰位RMS"]:7.2f}')
print('拿第一族数：', {LAB[m]: n for m, n in nb.items() if n})
