# -*- coding: utf-8 -*-
"""A15（a15_three.py --third cnet：三支融合 + 峰位校正）的模拟重建示例，和旧 A12 并排（只 Uddin，真实模式读数，全模拟）。
  用户 2026-11-02："把这个算法叫做 A15，对 A15 使用文章十二（Uddin 2024，learned sensing 012）的模型进行模拟重建"。
  重放 a11_peakfix.main 的读数（生成谱族：种子 SEED_SIM + 1；作者 5 束测试光：SEED_SIM + 50），两个算法吃同一份读数；
  读已训练的 qnet.pt，不重训。先核对重放出的逐族误差和 peakfix.json 一致，再画图。
  图 1：每族取 A15 误差在本族中位的那条，画 真值 / A15 / 旧 A12；按"A15 / 旧 A12"从大到小排（前面是 A15 占优最少的族）。
  图 2：作者 5 束测试光（商用光谱仪测的参考谱）的模拟读数，每束取 A15 形状误差居中的那次读数。
  py -I -X utf8 a15_examples.py [--test-seed 4242]
  输出 45_Uddin2024_真实R矩阵/results/a15_seed<种子>/{fig_a15_examples.png, fig_a15_beams.png, a15_examples.npz}
"""
import os, sys, json, time, argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument('--test-seed', type=int, default=4242); A_ = ap.parse_args()
sys.argv = [os.path.join(ROOT, 'a15_three.py'), '--third', 'cnet', '--test-seed', str(A_.test_seed)]
sys.path.insert(0, ROOT)
import numpy as np
import a15_three as T
import torch, torch.nn as tnn
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

PF, AD, P18, LD = T.PF, T.AD, T.P18, T.LD
WL, BM, DEV = AD.WL, AD.BM, PF.DEV
INK, INK2, GRID = '#1f1f1e', '#6b6a65', '#e4e3df'
C15, C12 = '#2a78d6', '#9a9890'
plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
RES = os.path.join(AD.DEVDIR, 'results'); SEED = A_.test_seed
OLD = 'a11_peakfix' if SEED == 4242 else f'a12x_orig_seed{SEED}'          # 旧 A12 在同一份读数上的结果


def qload(d):
    Q = torch.load(os.path.join(d, 'qnet.pt'), weights_only=False); mu, sd = Q['mu'], Q['sd']
    net = tnn.Sequential(tnn.Linear(len(mu), 32), tnn.Tanh(), tnn.Linear(32, 1)).double().to(DEV); net.load_state_dict(Q['state']); net.eval()
    return mu, sd, net


def main():
    t0 = time.time()
    d15, P, _ = T.build3(); q15 = qload(PF.OUT)
    d12, _, _ = T.AD_build0(); q12 = qload(os.path.join(RES, OLD))
    PF.log(f'两个算法建好 {time.time() - t0:.0f} s')

    def run(dec, Q, Y):                                                 # = a11_peakfix.main 里的 dec2
        o = dec(Y); Fx, xa, d = PF.featx(o); q = PF.q_of(Fx, *Q)
        return P18.unflat(PF.shift_np(xa, q * d), Y.shape[0])           # (nd, NW, M)

    J15 = json.load(open(os.path.join(PF.OUT, 'peakfix.json'), encoding='utf-8'))
    J12 = json.load(open(os.path.join(RES, OLD, 'peakfix.json'), encoding='utf-8'))
    sets = {k.replace('留出·', ''): v for k, v in AD.all_fams(AD.TEST_ST, AD.NTE, P, False).items()}
    g = np.random.default_rng(AD.SEED_SIM + 1); nd = LD.NDRAW; out = {}; dmax = 0.0
    for nm, S in sets.items():                                          # 族的顺序同 a11_peakfix，随机数逐族消耗一致
        Y = AD.meas(S.T, g, 'real'); M = S.shape[0]
        X15 = run(d15, q15, Y).transpose(0, 2, 1); X12 = run(d12, q12, Y).transpose(0, 2, 1)       # (nd, M, NW)
        e15 = np.array([LD.band_nrmse(X15[r].T, S.T) for r in range(nd)]); e12 = np.array([LD.band_nrmse(X12[r].T, S.T) for r in range(nd)])
        dmax = max(dmax, abs(e15.mean() - J15['fam']['real'][nm]['A11p']), abs(e12.mean() - J12['fam']['real'][nm]['A11p']))
        r, i = np.unravel_index(np.argsort(e15, axis=None)[e15.size // 2], e15.shape)
        out[nm] = dict(S=S[i], A15=X15[r, i], A12=X12[r, i], e15=float(e15[r, i]), e12=float(e12[r, i]),
                       m15=float(e15.mean()), m12=float(e12.mean()), win=float((e15 < e12).mean()))
    assert dmax < 1e-9, dmax
    PF.log(f'重放读数：A15、旧 A12 逐族误差与 peakfix.json 最大差 {dmax:.1e}；用时 {time.time() - t0:.0f} s')

    order = sorted(out, key=lambda k: -out[k]['m15'] / out[k]['m12'])
    PF.log(f'\n== A15 模拟重建（Uddin 真实模式读数，测试种子 {SEED}）：族平均形状误差，A15 赢旧 A12 的样本比例')
    for k in order:
        e = out[k]; PF.log(f'   {k:<18s} A15 {e["m15"]:.4f}  旧 A12 {e["m12"]:.4f}  {(e["m15"] / e["m12"] - 1) * 100:+6.1f} %  赢 {e["win"] * 100:3.0f} %')
    PF.log(f'   各族中位：A15 {np.median([out[k]["m15"] for k in out]):.4f}，旧 A12 {np.median([out[k]["m12"] for k in out]):.4f}；'
           f'全部样本里 A15 赢 {np.mean([out[k]["win"] for k in out]) * 100:.0f} %（各族等权）')

    nc = 6; nr = int(np.ceil(len(order) / nc)); wl = WL[BM]; nrm = lambda x: x[BM] / max(x[BM].max(), 1e-30)
    fg, axs = plt.subplots(nr, nc, figsize=(3.1 * nc, 2.35 * nr), squeeze=False)
    for j, k in enumerate(order):
        ax = axs[j // nc, j % nc]; e = out[k]
        ax.plot(wl, nrm(e['S']), color=INK, lw=2.2, label='真值')
        ax.plot(wl, nrm(e['A12']), color=C12, lw=1.2, ls='--', label='旧 A12')
        ax.plot(wl, nrm(e['A15']), color=C15, lw=1.6, label='A15')
        ax.set_title(f'{k}\n族平均 A15 {e["m15"]:.3f} / 旧 A12 {e["m12"]:.3f}（{(e["m15"] / e["m12"] - 1) * 100:+.0f} %）', fontsize=7.5, color=INK, loc='left')
        ax.text(0.98, 0.95, f'本条 A15 {e["e15"]:.3f}\n旧 A12 {e["e12"]:.3f}', transform=ax.transAxes, ha='right', va='top', fontsize=6.5, color=INK2)
        ax.tick_params(labelsize=6.5); ax.set_ylim(-0.08, 1.25); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    for j in range(len(order), nr * nc):
        axs[j // nc, j % nc].axis('off')
    axs[0, 0].legend(fontsize=7, frameon=False, loc='center left')
    fg.suptitle(f'A15 模拟重建：Uddin 2024 器件模型、真实模式读数，测试种子 {SEED}（每族取 A15 误差居中的那条；按"A15 / 旧 A12"从大到小排）',
                fontsize=10.5, color=INK, x=0.01, ha='left')
    fg.tight_layout(rect=(0, 0, 1, 0.98)); f1 = os.path.join(PF.OUT, 'fig_a15_examples.png'); fg.savefig(f1, dpi=140); PF.log(f1)

    # 作者 5 束测试光：同 a11_peakfix.main 第 3 节真实模式（种子 SEED_SIM + 50）
    import openpyxl
    fig = AD.PC.read_fig(openpyxl.load_workbook(AD.FW.XLSX, read_only=True, data_only=True), 'vis')
    refs = [fig[float(lt)] for lt in AD.LAM_TEST]
    truth = np.array([AD.to_grid(fl, ff) for _, _, fl, ff in refs]); ref_pk = np.array([AD.PC.peak(fl, ff) for _, _, fl, ff in refs])
    rng = np.random.default_rng(AD.SEED_SIM + 50); bm = []; dmax = 0.0
    PF.log(f'\n== 作者 5 束测试光的模拟读数（各 {AD.NSIM} 次）：平均 |峰位误差| nm ｜ 平均形状误差')
    for j, lt in enumerate(AD.LAM_TEST):
        fl, ff = refs[j][2], refs[j][3]; p = np.clip(ff, 0, None); p = p / np.trapezoid(p, fl) * AD.P_FIT[j]
        Yb = np.array([AD.DV.measure(fl, p, dark=False, warn=False, real=True, rng=rng) for _ in range(AD.NSIM)])[:, :, None] / AD.SCALE
        X15 = run(d15, q15, Yb)[:, :, 0]; X12 = run(d12, q12, Yb)[:, :, 0]
        p15 = PF.peaks(X15) - ref_pk[j]; p12 = PF.peaks(X12) - ref_pk[j]
        dmax = max(dmax, np.abs(p15 - J15['sim']['real']['A11p']['dpk'][j]).max(), np.abs(p12 - J12['sim']['real']['A11p']['dpk'][j]).max())
        e15 = np.array([AD.nrmse1(x, truth[j]) for x in X15]); e12 = np.array([AD.nrmse1(x, truth[j]) for x in X12]); r = np.argsort(e15)[len(e15) // 2]
        bm.append(dict(lt=float(lt), S=truth[j], A15=X15[r], A12=X12[r], pk=float(ref_pk[j]), dp15=float(np.abs(p15).mean()), dp12=float(np.abs(p12).mean()),
                       e15=float(e15.mean()), e12=float(e12.mean())))
        PF.log(f'   {lt:.0f} nm：A15 {bm[-1]["dp15"]:.2f} nm ｜ {bm[-1]["e15"]:.4f}；旧 A12 {bm[-1]["dp12"]:.2f} nm ｜ {bm[-1]["e12"]:.4f}')
    assert dmax < 1e-6, dmax
    PF.log(f'   重放与 peakfix.json 的峰位最大差 {dmax:.1e} nm')

    fg, axs = plt.subplots(1, len(bm), figsize=(3.2 * len(bm), 2.9), squeeze=False)
    for ax, b in zip(axs[0], bm):
        ax.plot(wl, nrm(b['S']), color=INK, lw=2.2, label='参考谱（商用光谱仪）')
        ax.plot(wl, nrm(b['A12']), color=C12, lw=1.2, ls='--', label='旧 A12')
        ax.plot(wl, nrm(b['A15']), color=C15, lw=1.6, label='A15')
        ax.set_xlim(b['pk'] - 70, b['pk'] + 70); ax.set_ylim(-0.08, 1.25)
        ax.set_title(f'{b["lt"]:.0f} nm 测试光\n平均 |峰位误差| A15 {b["dp15"]:.2f} / 旧 A12 {b["dp12"]:.2f} nm', fontsize=8, color=INK, loc='left')
        ax.tick_params(labelsize=7); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); ax.set_xlabel('波长 / nm', fontsize=7.5, color=INK2)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    axs[0, 0].legend(fontsize=6.5, frameon=False, loc='upper left')
    fg.suptitle(f'A15 模拟重建：Uddin 2024 作者 5 束测试光的模拟读数（真实模式，各 {AD.NSIM} 次，画形状误差居中的那次）', fontsize=10, color=INK, x=0.01, ha='left')
    fg.tight_layout(rect=(0, 0, 1, 0.93)); f2 = os.path.join(PF.OUT, 'fig_a15_beams.png'); fg.savefig(f2, dpi=140); PF.log(f2)

    np.savez(os.path.join(PF.OUT, 'a15_examples.npz'), order=order, **{f'{k}|{m}': out[k][m] for k in out for m in ('S', 'A15', 'A12')},
             **{f'beam{b["lt"]:.0f}|{m}': b[m] for b in bm for m in ('S', 'A15', 'A12')})
    open(os.path.join(PF.OUT, 'log_examples.txt'), 'w', encoding='utf-8').write('\n'.join(PF._L))
    PF.log(f'用时 {time.time() - t0:.0f} s')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    import warnings; warnings.filterwarnings('ignore')
    main()
