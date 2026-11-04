# -*- coding: utf-8 -*-
"""A12（= A11p：A11 + 最后一步峰位校正）的各族重建示例，和 Xie MLP 并排（只 Uddin，真实模式读数，全模拟）。
  重放 a11_peakfix.py 的真实模式读数（种子 SEED_SIM + 1），读已训练的 qnet.pt 和 all_algos 的 xie_samelib.pt，不重训。
  每族取 A12 误差在本族中位的那条（读数 × 谱），画 真值 / A12 / Xie MLP；另统计每族 A12 赢 Xie 的样本比例。
  py -I -X utf8 a12_examples.py      输出 45_Uddin2024_真实R矩阵/results/a11_peakfix/{fig_a12_examples.png, a12_examples.npz}
"""
import os, sys, json, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.argv = [os.path.join(ROOT, 'a11_peakfix.py')]
sys.path.insert(0, ROOT)
import a11_peakfix as PF
import torch, torch.nn as tnn
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

AD, P18, LD = PF.AD, PF.P18, PF.LD
WL, BM, DEV = AD.WL, AD.BM, PF.DEV
INK, INK2, GRID = '#1f1f1e', '#6b6a65', '#e4e3df'
C_A12, C_XIE = '#2a78d6', '#8a5cd1'
plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False


class XieMLP(tnn.Module):                               # 同 all_algos_device.py
    def __init__(s, n_, no, w=512):
        super().__init__(); s.i = tnn.Linear(n_, w); s.o = tnn.Linear(w, no)
        s.B = tnn.ModuleList([tnn.Sequential(tnn.Linear(w, w), tnn.ReLU(), tnn.Linear(w, w)) for _ in range(3)])

    def forward(s, y):
        z = torch.relu(s.i(y))
        for blk in s.B:
            z = torch.relu(z + blk(z))
        return s.o(z)


def main():
    t0 = time.time(); RES = os.path.join(AD.DEVDIR, 'results')
    decode, P, used = AD.build()
    Q = torch.load(os.path.join(PF.OUT, 'qnet.pt'), weights_only=False); mu, sd = Q['mu'], Q['sd']
    qnet = tnn.Sequential(tnn.Linear(len(mu), 32), tnn.Tanh(), tnn.Linear(32, 1)).double().to(DEV); qnet.load_state_dict(Q['state']); qnet.eval()
    xs = XieMLP(AD.N_RD, int(BM.sum())).to(DEV); xs.load_state_dict(torch.load(os.path.join(RES, 'all_algos', 'xie_samelib.pt'))); xs.eval()

    def xie(Yf):
        torch.backends.cuda.matmul.allow_tf32 = True                             # 同 all_algos_device.py；A11 那边原跑没开，用完关掉
        with torch.no_grad():
            y = torch.tensor(Yf, dtype=torch.float32, device=DEV); y = y / (y.abs().sum(1, keepdim=True) + 1e-30)
            Xb = torch.nn.functional.softplus(xs(y)).double().cpu().numpy()
        torch.backends.cuda.matmul.allow_tf32 = False
        X = np.zeros((len(Xb), len(WL))); X[:, BM] = Xb
        return X

    PK = json.load(open(os.path.join(PF.OUT, 'peakfix.json'), encoding='utf-8'))['fam']['real']
    XJ = json.load(open(os.path.join(RES, 'all_algos', 'all_algos.json'), encoding='utf-8'))
    XF = {k.replace('留出·', ''): v for k, v in XJ['fam']['real'].items()}
    sets = {k.replace('留出·', ''): v for k, v in AD.all_fams(AD.TEST_ST, AD.NTE, P, False).items()}
    g = np.random.default_rng(AD.SEED_SIM + 1); nd = LD.NDRAW; out = {}; dx = 0.0
    for nm, S in sets.items():                                                   # 族的顺序与 a11_peakfix 相同，随机数逐族消耗一致
        Y = AD.meas(S.T, g, 'real'); M = S.shape[0]
        o = decode(Y); Fx, xa, d = PF.featx(o); q = PF.q_of(Fx, mu, sd, qnet)
        Xa = PF.shift_np(xa, q * d).reshape(nd, M, -1)                           # flat 顺序 = (读数, 谱)
        Xx = xie(Y.transpose(0, 2, 1).reshape(nd * M, -1)).reshape(nd, M, -1)
        ea = np.array([LD.band_nrmse(Xa[r].T, S.T) for r in range(nd)]); ex = np.array([LD.band_nrmse(Xx[r].T, S.T) for r in range(nd)])
        assert abs(ea.mean() - PK[nm]['A11p']) < 1e-9 and abs(ex.mean() / XF[nm]['XieMLP'] - 1) < 0.01, (nm, ea.mean(), PK[nm]['A11p'], ex.mean(), XF[nm]['XieMLP'])
        dx = max(dx, abs(ex.mean() / XF[nm]['XieMLP'] - 1))                     # Xie 原跑用 TF32 + 不同分批，浮点差 < 1 %
        r, i = np.unravel_index(np.argsort(ea, axis=None)[ea.size // 2], ea.shape)
        out[nm] = dict(S=S[i], A12=Xa[r, i], Xie=Xx[r, i], ea=float(ea[r, i]), ex=float(ex[r, i]), win=float((ea < ex).mean()),
                       ma=float(ea.mean()), mx=float(ex.mean()))
    print(f'重放读数：A12 与 peakfix.json 逐族一致；Xie 与 all_algos.json 最大相对差 {dx * 100:.2f} %；用时 {time.time() - t0:.0f} s')

    ext = [m for m in XJ['name'] if m in XF[next(iter(XF))] and m not in ('A11', 'A11_R')]
    best = {k: min(ext, key=lambda m: XF[k][m]) for k in out}
    ratio = {k: out[k]['ma'] / XF[k][best[k]] for k in out}
    order = sorted(out, key=lambda k: -ratio[k])
    print('族  A12  外部最好(谁)  比值  A12 赢 Xie 的样本比例')
    for k in order:
        print(f'{k:<16s} {out[k]["ma"]:.4f}  {XF[k][best[k]]:.4f}({XJ["name"][best[k]]})  ×{ratio[k]:.2f}  {out[k]["win"] * 100:.0f} %')

    nc = 6; nr = int(np.ceil(len(order) / nc)); wl = WL[BM]
    fg, axs = plt.subplots(nr, nc, figsize=(3.1 * nc, 2.35 * nr), squeeze=False)
    for j, k in enumerate(order):
        ax = axs[j // nc, j % nc]; e = out[k]; nrm = lambda x: x[BM] / max(x[BM].max(), 1e-30)
        ax.plot(wl, nrm(e['S']), color=INK, lw=2.2, label='真值')
        ax.plot(wl, nrm(e['A12']), color=C_A12, lw=1.6, label='A12')
        ax.plot(wl, nrm(e['Xie']), color=C_XIE, lw=1.2, ls='--', label='Xie MLP')
        lag = ratio[k] > 1.0001
        ax.set_title(f'{k}{"（落后）" if lag else ""}\n族平均 A12 {e["ma"]:.3f} / 外部最好 {XF[k][best[k]]:.3f} {XJ["name"][best[k]]}',
                     fontsize=7.5, color=INK, loc='left')
        ax.text(0.98, 0.95, f'本条 A12 {e["ea"]:.3f}\nXie {e["ex"]:.3f}', transform=ax.transAxes, ha='right', va='top', fontsize=6.5, color=INK2)
        ax.tick_params(labelsize=6.5); ax.set_ylim(-0.08, 1.25); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    for j in range(len(order), nr * nc):
        axs[j // nc, j % nc].axis('off')
    axs[0, 0].legend(fontsize=7, frameon=False, loc='center left')
    fg.suptitle('A12 各族重建示例（Uddin 模拟器真实模式读数；每族取 A12 误差在本族中位的那条；按"A12 / 外部最好"从大到小排，前面的是落后族）',
                fontsize=10.5, color=INK, x=0.01, ha='left')
    fg.tight_layout(rect=(0, 0, 1, 0.98)); f = os.path.join(PF.OUT, 'fig_a12_examples.png'); fg.savefig(f, dpi=140); print(f)
    np.savez(os.path.join(PF.OUT, 'a12_examples.npz'), order=order, **{f'{k}|{m}': out[k][m] for k in out for m in ('S', 'A12', 'Xie')})


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    import warnings; warnings.filterwarnings('ignore')
    main()
