# -*- coding: utf-8 -*-
"""Wang 2024 器件（47 号）· 真实器件模式：A15（三支融合 + 峰位校正）重训、重测。

用户 2026-11-02："使用 A15 算法对比 A12、Xie MLP 还有常规算法，进行光谱重建。使用这个的数据进行重新的训练。"
A15 算法代码一行不抄（同 a15_cui_m3.py）：先按 wang 导入 a11_device，再导入 a15_three（--third cnet），
  直接用 a15_three.build3 和 a11_peakfix 的 featx / fit_q / q_of / shift_np。
在 Wang 真实器件模式上重训的：第三支互补网络 cnet（读数误差 = a11_device wang 的 meas_t，训练器件）、两个网络支各自的 κ、
  两支 / 三支融合小网、峰位校正 q 小网。门控支、零件网络、原网络支读 results/a11/ 的 net8.pt / mlp.pt（同 A11、A12）。
测试：wang_eval（共用测试读数）；两支融合（同一次重拟）应与 a11.json 的 A11 相同，用来核对读数一致。
  py -I -X utf8 a15_wang.py [--quick]
输出：47_Wang2024_真实R矩阵/results/a15/{a15.json, raw.pkl, log.txt, qnet.pt, cnet.pt}
"""
import os, sys, json, time, pickle
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
QUICK = '--quick' in sys.argv
sys.argv = [os.path.join(ROOT, 'a11_device.py'), 'wang'] + (['--quick'] if QUICK else [])
import a11_device as AD
sys.argv = [os.path.join(ROOT, 'a15_three.py'), '--third', 'cnet']
import a15_three as T3
import wang_eval as WE
import torch

PF, P18, LD = T3.PF, AD.P18, AD.LD
assert PF.AD is AD and AD.WANG
RES = os.path.join(AD.DEVDIR, 'results'); OUT = os.path.join(RES, 'a15' + ('_quick' if QUICK else '')); os.makedirs(OUT, exist_ok=True)
for d in ('a11_peakfix', T3.TAG, 'a15_nets'):                          # 两个模块 import 时建的空目录（Uddin 命名），没东西就删
    d = os.path.join(RES, d)
    if os.path.isdir(d) and not os.listdir(d):
        os.rmdir(d)
T3.NETDIR = PF.OUT = OUT
if QUICK:
    _tc = T3.train_cnet; T3.train_cnet = lambda *a, **k: _tc(*a, steps=300, **k)
METH = ['gate', 'cons', 't3c', 'A2', 'A11', 'A11p']                     # a15_three 的键：A2 = 两支融合（= A11），A11 = 三支融合，A11p = A15
KEY = dict(gate='gate', cons='cons', t3c='t3c', A2='two', A11='three', A11p='A15')
NAME = dict(gate='门控支', cons='原网络支', t3c='互补支', two='两支融合', three='三支融合', A15='A15')
log = PF.log


def main():
    t0 = time.time()
    decode, P, used = AD.build()                                        # = T3.build3
    log(f'A15（Wang 真实器件模式重训）建好 {time.time() - t0:.0f} s')

    # 1. 峰位校正 q（= a11_peakfix.main 第 1 步）
    vsets = AD.all_fams(P18.VSEED, AD.NVF, P, True); g = np.random.default_rng(PF.QSEED); nd = LD.NDRAW
    Fv, Xa, de, St = [], [], [], []
    for k, S in vsets.items():
        o = decode(AD.meas(S.T, g)); Fx, xa, d = PF.featx(o)
        Fv.append(Fx); Xa.append(xa); de.append(d); St.append(np.tile(S, (nd, 1)))
    Fv, Xa, de, St = (np.concatenate(v) for v in (Fv, Xa, de, St))
    rng = np.random.default_rng(5); idx = rng.permutation(len(Fv)); tr = (idx[:int(0.7 * len(idx))], idx[int(0.7 * len(idx)):])
    va = tr[1]; pkm = PF.single_peak(St[va]); pt = PF.peaks(St[va])
    e_old = float(np.mean([AD.nrmse1(Xa[i], St[i]) for i in va])); p_all = np.abs(PF.peaks(Xa[va]) - pt)[pkm]
    log(f'\nq 训练：验证谱 {len(vsets)} 族 {len(Fv)} 个样本，|d| ≤ {PF.DMAX:g} nm 的占 {(de != 0).mean() * 100:.0f} %；'
        f'30 % 留出部分三支融合：形状误差 {e_old:.4f}，单峰 |峰位误差| 平均 {p_all.mean():.3f}、中位 {np.median(p_all):.3f} nm')
    cand = {}
    for lam in PF.LAMS:
        mu, sd, net = PF.fit_q(Fv, Xa, de, St, lam, tr); q = PF.q_of(Fv[va], mu, sd, net); Xp = PF.shift_np(Xa[va], q * de[va])
        e = float(np.mean([AD.nrmse1(Xp[i], St[va][i]) for i in range(len(va))])); pa_ = np.abs(PF.peaks(Xp) - pt)[pkm]
        cand[lam] = (e, float(pa_.mean()), mu, sd, net)
        log(f'   λ = {lam:<6g} 留出：形状误差 {e:.4f}（{(e / e_old - 1) * 100:+.1f} %），单峰 |峰位误差| 平均 {pa_.mean():.3f}、中位 {np.median(pa_):.3f} nm；q 均值 {q.mean():.2f}')
    ok = [l for l in PF.LAMS if cand[l][0] <= e_old * 1.005] or [min(PF.LAMS, key=lambda l: cand[l][0])]
    lam = min(ok, key=lambda l: cand[l][1]); _, _, mu, sd, qnet = cand[lam]
    log(f'   选 λ = {lam:g}（形状不差于三支融合 0.5 % 以内、峰位最小）')
    torch.save(dict(state=qnet.state_dict(), mu=mu, sd=sd, lam=lam, dmax=PF.DMAX), os.path.join(OUT, 'qnet.pt'))

    def run(Y):
        o = decode(Y); Fx, xa, d = PF.featx(o); q = PF.q_of(Fx, mu, sd, qnet)
        o['A11p'] = P18.unflat(PF.shift_np(xa, q * d), Y.shape[0])
        return {KEY[m]: o[m] for m in METH}, {}

    # 2. 测试
    M = [KEY[m] for m in METH]
    sets, TR, F3 = WE.test_data(P, used)
    J, raw = WE.evaluate(run, sets, TR, F3, log); J['lam'] = lam
    A0 = json.load(open(os.path.join(AD.OUT, 'a11.json'), encoding='utf-8'))
    dif = max(abs(J['fam'][k]['two'] - A0['fam']['device'][k]['A11']) for k in sets)
    log(f'\n== 生成谱族 · 真实器件模式读数（每族 {AD.NTE} 条 × {nd} 次）  两支融合（本次重拟）与 a11.json 的 A11 最大差 {dif:.2e}')
    J['med'] = WE.table(J, M, NAME, log)
    log(f'\n用时 {time.time() - t0:.0f} s')
    json.dump(J, open(os.path.join(OUT, 'a15.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
    pickle.dump(raw, open(os.path.join(OUT, 'raw.pkl'), 'wb'))
    open(os.path.join(OUT, 'log.txt'), 'w', encoding='utf-8').write('\n'.join(PF._L))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    import warnings; warnings.filterwarnings('ignore')
    main()
