# -*- coding: utf-8 -*-
"""A12x：A12 整条链不动，只把网络支（全族网络）换成 Xie 2026b 结构的网络（只 Uddin，真实模式）。

用户 2026-11-02："试一下"（接上一问"Xie MLP 的不少指标都优于 A12，为什么"）。
  上一轮诊断（a12_fusion/branch_log.txt）：同一训练库、同一读数误差模型、同样步数下，Xie 结构（512 宽 × 3 残差块、L1 归一输入）
  的网络在峰位和宽谱族上都比 A12 的网络支（3 层 × 1024 GELU、L2 归一）好，融合和峰位校正补不回来。
做法：P17.infer_mlp（A12 里网络支和 scale_y 估亮度都走它）换成读 all_algos/xie_samelib.pt 的 Xie 网络，不重训网络；
  其余全部照 a11_peakfix.main 重新建：一致化协方差和 κ、融合小网、峰位校正 q 小网都在融合验证集（种子 VSEED / QSEED）上重定。
  门控支、零件网络、A1 不变。另在每次解码里加一列 'xie' = Xie 网络原始输出（读数最小二乘定幅度，同 infer_mlp），方便同种子对照。
测试种子：
  --test-seed 4242（默认）= a11_device / a11_peakfix 的测试集，和旧 A12、雷达图同一份，但 A12 的各步修改都看过它；
  --test-seed 6161 = 开发时没用过的新种子（派生 6162 / 6661 / 7061 / 10482 / 6494 和训练、验证、先验种子都不重），
     旧 A12 也在这个种子上重跑一遍（--net orig），三者同种子比。
  --metrics：建好后接着跑 a12_metrics.main（雷达图那 15 项指标的来源），输出到本次结果目录的 metrics/。
  py -I -X utf8 a12x_xienet.py [--net xie|orig] [--test-seed 4242] [--metrics]
输出：45_Uddin2024_真实R矩阵/results/a12x_<net>_seed<种子>/{peakfix.json, log.txt, qnet.pt, metrics/}
"""
import os, sys, json, argparse
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument('--net', choices=['xie', 'orig'], default='xie')
ap.add_argument('--test-seed', type=int, default=4242); ap.add_argument('--metrics', action='store_true')
ARG = ap.parse_args()
sys.argv = [os.path.join(ROOT, 'a11_peakfix.py')]
sys.path.insert(0, ROOT)
import a11_peakfix as PF
import torch, torch.nn as tnn

AD, P18, P17 = PF.AD, PF.P18, PF.AD.P17
WL, BM, DEV = AD.WL, AD.BM, PF.DEV; NWB = int(BM.sum())
TAG = f'a12x_{ARG.net}_seed{ARG.test_seed}'
PF.OUT = os.path.join(AD.DEVDIR, 'results', TAG); os.makedirs(PF.OUT, exist_ok=True)
AD.TEST_ST = ARG.test_seed
SAME = ARG.net == 'orig' and ARG.test_seed == 4242                    # 只有这种情况，旧 a11.json / all_algos.json 的数才和本次同一份


class XieMLP(tnn.Module):                                              # = all_algos_device.XieMLP
    def __init__(s, n_, no, w=512):
        super().__init__(); s.i = tnn.Linear(n_, w); s.o = tnn.Linear(w, no)
        s.B = tnn.ModuleList([tnn.Sequential(tnn.Linear(w, w), tnn.ReLU(), tnn.Linear(w, w)) for _ in range(3)])

    def forward(s, y):
        z = torch.relu(s.i(y))
        for blk in s.B:
            z = torch.relu(z + blk(z))
        return s.o(z)


XN = XieMLP(AD.N_RD, NWB).to(DEV)
XN.load_state_dict(torch.load(os.path.join(AD.DEVDIR, 'results', 'all_algos', 'xie_samelib.pt'))); XN.eval()


def xie_infer(Rn, Y):
    """Y (nd, n, M) → (nd, NW, M)：Xie 网络（L1 归一输入）出形状，幅度按读数最小二乘（同 algo17.infer_mlp）。"""
    nd, n, M = Y.shape; Yr = Y.transpose(0, 2, 1).reshape(nd * M, n)
    with torch.no_grad():
        y = torch.tensor(Yr, dtype=torch.float32, device=DEV)
        x = torch.nn.functional.softplus(XN(y / (y.abs().sum(1, keepdim=True) + 1e-30))).double().cpu().numpy()
    X = np.zeros((nd * M, Rn.shape[1])); X[:, BM] = x; p = X @ Rn.T
    a = np.maximum((p * Yr).sum(1) / np.maximum((p ** 2).sum(1), 1e-30), 0)
    return (X * a[:, None]).reshape(nd, M, -1).transpose(0, 2, 1)


if ARG.net == 'xie':
    P17.infer_mlp = lambda NT, Rn, Y: xie_infer(Rn, Y)                 # 网络支 + scale_y 都换成 Xie

_build0 = AD.build


def _build():
    decode, P, used = _build0()

    def dec(Y):
        o = decode(Y); o['xie'] = np.clip(xie_infer(AD.RN, Y), 0, None)       # 原始读数；形状误差、峰位都与幅度无关
        return o
    return dec, P, used


AD.build = _build
PF.METH.append('xie')
PF.NAME.update(xie='Xie 原始', cons='网络支一致化', A11='融合', A11p='A12x' if ARG.net == 'xie' else 'A12')


class _JS:
    """旧结果文件只在同一份读数时才拿来对照；否则当空的读（日志里那几列显示 nan，不冒充同种子的数）。"""
    def __getattr__(s, k):
        return getattr(json, k)

    def load(s, f):
        nm = os.path.basename(getattr(f, 'name', ''))
        d = json.load(f)
        if nm == 'a11.json' and not SAME:
            return dict(fam={}, sim={})
        if nm == 'all_algos.json' and ARG.test_seed != 4242:
            return dict(fam={}, fampk={}, sim=d.get('sim', {}))        # 5 束模拟读数的种子和测试种子无关，照用
        return d


PF.json = _JS()


def summary():
    """本次 A12x / A12 与 Xie 原始，和旧 A12（a11_peakfix/peakfix.json，种子 4242）对照。"""
    J = json.load(open(os.path.join(PF.OUT, 'peakfix.json'), encoding='utf-8')); L = []
    fam = {k: v for k, v in J['fam']['real'].items() if not k.startswith('_')}
    me = 'A11p'; lab = PF.NAME['A11p']
    cols = [('xie', 'Xie 原始'), ('cons', '网络支一致化'), ('gate', '门控支'), (me, lab)]
    old = None
    if ARG.test_seed == 4242:
        O = json.load(open(os.path.join(AD.DEVDIR, 'results', 'a11_peakfix', 'peakfix.json'), encoding='utf-8'))
        old = {k: v for k, v in O['fam']['real'].items() if not k.startswith('_')}
    L.append(f'\n== 汇总（真实模式，测试种子 {ARG.test_seed}）：各族形状误差' + ('；旧 A12 读 a11_peakfix/peakfix.json（同种子同读数）' if old else ''))
    L.append('   族                    ' + ''.join(f'{c[1]:>12s}' for c in cols) + ('      旧 A12' if old else ''))
    for k in fam:
        L.append(f'   {k:<20s}' + ''.join(f'{fam[k][c[0]]:12.4f}' for c in cols) + (f'{old[k]["A11p"]:12.4f}' if old else ''))
    L.append('   各族中位              ' + ''.join(f'{np.median([fam[k][c[0]] for k in fam]):12.4f}' for c in cols)
             + (f'{np.median([old[k]["A11p"] for k in old]):12.4f}' if old else ''))
    win = sum(fam[k][me] < fam[k]['xie'] - 1e-4 for k in fam); lose = sum(fam[k][me] > fam[k]['xie'] + 1e-4 for k in fam)
    L.append(f'   {lab} 对 Xie 原始：好 {win} 族、差 {lose} 族')
    if old:
        w2 = sum(fam[k][me] < old[k]['A11p'] - 1e-4 for k in fam); l2 = sum(fam[k][me] > old[k]['A11p'] + 1e-4 for k in fam)
        L.append(f'   {lab} 对旧 A12：好 {w2} 族、差 {l2} 族')
    for nm in PF.PKF:
        r = J['fampk']['real'][nm]
        L.append(f'   {nm} 峰位 |误差| 中位 nm：' + '，'.join(f'{c[1]} {r[c[0]]:.3f}' for c in cols)
                 + (f'，旧 A12 {O["fampk"]["real"][nm]["A11p"]:.3f}' if old else ''))
    s = J['sim']['real']
    L.append('   5 束模拟读数（真实模式各 200 次）平均 |峰位误差| nm：' + '，'.join(f'{c[1]} {np.abs(np.array(s[c[0]]["dpk"])).mean():.3f}' for c in cols))
    for x in L:
        PF.log(x)
    open(os.path.join(PF.OUT, 'log.txt'), 'w', encoding='utf-8').write('\n'.join(PF._L))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    import warnings; warnings.filterwarnings('ignore')
    PF.log(f'{TAG}：网络支 = {"Xie 结构（all_algos/xie_samelib.pt）" if ARG.net == "xie" else "原 A12 全族网络（mlp.pt）"}；测试种子 {AD.TEST_ST}')
    PF.main(); summary()
    if ARG.metrics:
        import a12_metrics as MT                                       # 它在导入时按 PF.OUT 定输出目录，PF.OUT 已改到本次目录
        MT.main()
