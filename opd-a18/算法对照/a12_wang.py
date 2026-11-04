# -*- coding: utf-8 -*-
"""Wang 2024 器件（47 号）· 真实器件模式：A12（= A11 + 峰位校正）重训、重测。

用户 2026-11-02："使用 A15 算法对比 A12、Xie MLP 还有常规算法，进行光谱重建。使用这个的数据进行重新的训练。"
A12 算法一行不改（已定版）：先按 wang 导入 a11_device（训练读数 = 真实器件模式 meas_t / meas 'train'，训练器件），
  再导入 a11_peakfix（它里面 import 的 a11_device 已在 sys.modules，拿到的就是 wang 版），直接用它的 featx / fit_q / q_of / shift_np。
  A11 的零件网络、全族网络读 results/a11/ 里 a11_device.py wang 训好的 net8.pt / mlp.pt；q 小网在融合验证集的谱上新抽一份训练口径读数来训，
  λ 规则同 a11_peakfix（形状不差于 A11 0.5 % 以内、单峰峰位最小）。
测试：wang_eval（共用测试读数）；本次重算的 A11 和 results/a11/a11.json 的 'device' 档逐族比对，核对读数一致。
  py -I -X utf8 a12_wang.py [--quick]
输出：47_Wang2024_真实R矩阵/results/a12/{a12.json, raw.pkl, log.txt, qnet.pt}
"""
import os, sys, json, time, pickle
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
QUICK = '--quick' in sys.argv
sys.argv = [os.path.join(ROOT, 'a11_device.py'), 'wang'] + (['--quick'] if QUICK else [])
import a11_device as AD                                                 # 先占住 sys.modules：a11_peakfix 里的 import 拿到的就是 wang 版
sys.argv = [os.path.join(ROOT, 'a11_peakfix.py')]
import a11_peakfix as PF
import wang_eval as WE
import torch

P18, LD = AD.P18, AD.LD
assert PF.AD is AD and AD.WANG
RES = os.path.join(AD.DEVDIR, 'results'); OUT = os.path.join(RES, 'a12' + ('_quick' if QUICK else '')); os.makedirs(OUT, exist_ok=True)
if os.path.isdir(PF.OUT) and not os.listdir(PF.OUT):                   # a11_peakfix import 时建的空目录（Uddin 命名）
    os.rmdir(PF.OUT)
PF.OUT = OUT
METH = ['gate', 'cons', 'A11', 'A12']
NAME = dict(gate='门控支', cons='网络支', A11='A11', A12='A12')
log = PF.log


def main():
    t0 = time.time()
    decode, P, used = AD.build()
    log(f'A11（Wang 真实器件模式训练）建好 {time.time() - t0:.0f} s')

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
        f'30 % 留出部分 A11：形状误差 {e_old:.4f}，单峰 |峰位误差| 平均 {p_all.mean():.3f}、中位 {np.median(p_all):.3f} nm')
    cand = {}
    for lam in PF.LAMS:
        mu, sd, net = PF.fit_q(Fv, Xa, de, St, lam, tr); q = PF.q_of(Fv[va], mu, sd, net); Xp = PF.shift_np(Xa[va], q * de[va])
        e = float(np.mean([AD.nrmse1(Xp[i], St[va][i]) for i in range(len(va))])); pa_ = np.abs(PF.peaks(Xp) - pt)[pkm]
        cand[lam] = (e, float(pa_.mean()), mu, sd, net)
        log(f'   λ = {lam:<6g} 留出：形状误差 {e:.4f}（{(e / e_old - 1) * 100:+.1f} %），单峰 |峰位误差| 平均 {pa_.mean():.3f}、中位 {np.median(pa_):.3f} nm；q 均值 {q.mean():.2f}')
    ok = [l for l in PF.LAMS if cand[l][0] <= e_old * 1.005] or [min(PF.LAMS, key=lambda l: cand[l][0])]
    lam = min(ok, key=lambda l: cand[l][1]); _, _, mu, sd, qnet = cand[lam]
    log(f'   选 λ = {lam:g}（形状不差于 A11 0.5 % 以内、峰位最小）')
    torch.save(dict(state=qnet.state_dict(), mu=mu, sd=sd, lam=lam, dmax=PF.DMAX), os.path.join(OUT, 'qnet.pt'))

    def run(Y):
        o = decode(Y); Fx, xa, d = PF.featx(o); q = PF.q_of(Fx, mu, sd, qnet)
        o['A12'] = P18.unflat(PF.shift_np(xa, q * d), Y.shape[0])
        return {m: o[m] for m in METH}, {}

    # 2. 测试
    sets, TR, F3 = WE.test_data(P, used)
    J, raw = WE.evaluate(run, sets, TR, F3, log); J['lam'] = lam
    A0 = json.load(open(os.path.join(AD.OUT, 'a11.json'), encoding='utf-8'))
    dif = max(abs(J['fam'][k]['A11'] - A0['fam']['device'][k]['A11']) for k in sets)
    log(f'\n== 生成谱族 · 真实器件模式读数（每族 {AD.NTE} 条 × {nd} 次）  重算 A11 与 a11.json 最大差 {dif:.2e}')
    J['med'] = WE.table(J, METH, NAME, log)
    nb = sum(J['fam'][k]['A12'] < J['fam'][k]['A11'] - 1e-4 for k in sets); nw = sum(J['fam'][k]['A12'] > J['fam'][k]['A11'] + 1e-4 for k in sets)
    log(f'   A12 对 A11：形状误差变好 {nb} 族、变差 {nw} 族（差 > 1e-4）')
    log(f'\n用时 {time.time() - t0:.0f} s')
    json.dump(J, open(os.path.join(OUT, 'a12.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
    pickle.dump(raw, open(os.path.join(OUT, 'raw.pkl'), 'wb'))
    open(os.path.join(OUT, 'log.txt'), 'w', encoding='utf-8').write('\n'.join(PF._L))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    import warnings; warnings.filterwarnings('ignore')
    main()
