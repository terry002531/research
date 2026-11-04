# -*- coding: utf-8 -*-
"""Cui 2025 器件 · M3：A15（三支融合 + 峰位校正）在 Cui 上重训、重测，和 A11 / A12 / Xie MLP 等对比。

用户 2026-11-02："使用 A15 算法，对模拟 M3 的读数进行训练，最后和 A11、A12 还有 Xie MLP 等其他算法进行重建对比"；
  "算法框架要求是通用的，不同器件重新训练就行了"。
所以 A15 算法代码一行不抄：先按 cui --m3 导入 a11_device，再导入 a15_three（它里面的 a11_peakfix 写的是 uddin，
  但 a11_device 已在 sys.modules，拿到的就是 Cui · M3 版），直接用 a15_three.build3 和 a11_peakfix 的 featx / fit_q / q_of / shift_np。
在 Cui · M3 上重训的：第三支互补网络（cnet，读数误差 = a11_device --m3 的 meas_t，含逐 Vgs 帧增益漂移、a 对数均匀）、
  两个网络支各自的一致化 κ、两支 / 三支融合小网、峰位校正 q 小网（λ 照 a11_peakfix 规则挑）。
  门控支、零件网络、原网络支读 a11_M3/ 里已用 M3 训好的 net8.pt / mlp.pt（同 A11、A12）。
测试：重放 a11_device.py cui --m3 的读数（同 a12_cui_m3.py）：
  a. 生成谱族 · M3 读数（a = 1），种子 SEED_SIM + 3   b. 作者 6 束真实读数   c. 同一批光的模拟 M3 读数 × NSIM，种子 SEED_SIM + 52
  两支融合（同一次重拟）应与 a11.json 的 A11 相同，用来核对读数重放；A12 读 a12_M3/a12.json，其他算法读 all_algos_M3/all_algos.json。
  py -I -X utf8 a15_cui_m3.py [--no-b]
输出：46_Cui2025_真实R矩阵/results/a15_M3/{a15.json, log.txt, qnet.pt, cnet.pt, demo.pkl}
--no-b（用户 2026-11-02 的消融 "A 加 C 不要 B"）：a15_three --no-b，C 仍读 a15_M3/cnet.pt；输出 a15_noB_M3/，
  键 AC（A+C 融合）、A15noB（加峰位校正）；--no-b --cnet0 = 完全不用 B（C 换不加权的 cnet0，Cui 上没有就训，存 a15_M3/cnet0.pt），
  输出 a15_cnet0_noB_M3/，键 AC0 / A15free，另对照 a15_noB_M3 的 A15noB；对照多读 a15_M3/a15.json 的 A15（同读数）；两支融合核对读数重放的那几处跳过。
--no-b --cnetA / --cnetS（用户 2026-11-03 "两个都跑"：让 A 起作用的两个试法，见 a15_three 文件头）：B 换 cnetA / cnetS（Cui 上训，存 a15_M3/），
  输出 a16_bwA_M3/、a16_bsA_M3/；对照多读 a16_M3 的 A16 和它的 B 单独（键 t3raw）。
--no-b --cnet0 --a16a K [--a16a-fams nonpeaky] [--a16a-fuse raw] [--a16a-nvf N]（用户 2026-11-03 定名 A16A：B 不动，只改门控支 A 的先验，见 a15_three 文件头）：
  输出 a16a_M3/（K ≠ 32 或消融带后缀），键同 A16、主列 A16A；对照多读 a16_M3 的 A16。
"""
import os, sys, json, time, pickle
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
NOB = '--no-b' in sys.argv[1:]; C0 = '--cnet0' in sys.argv[1:]            # --cnet0：第三支用不加权的 cnet0（只和 --no-b 一起用 = 完全不用 B）
WX = next((t for t in ('cnetA', 'cnetS') if f'--{t}' in sys.argv[1:]), None)   # 让 A 起作用的两个试法
assert NOB or not (C0 or WX)
_a = sys.argv[1:]; KA = int(_a[_a.index('--a16a') + 1]) if '--a16a' in _a else 0
XA = [x for i, x in enumerate(_a) if x.startswith('--a16a') or (i and _a[i - 1].startswith('--a16a'))]   # --a16a* 原样传给 a15_three
assert not KA or (C0 and NOB)
sys.argv = [os.path.join(ROOT, 'a11_device.py'), 'cui', '--m3']
import a11_device as AD                                                 # 先占住 sys.modules：后面 a11_peakfix 的 import 拿到的就是这个
sys.argv = [os.path.join(ROOT, 'a15_three.py'), '--third', WX or ('cnet0' if C0 else 'cnet')] + (['--no-b'] if NOB else []) + XA
import a15_three as T3
import torch

PF, P18, LD = T3.PF, AD.P18, AD.LD
assert PF.AD is AD and AD.ARG.device == 'cui' and AD.M3
RES = os.path.join(AD.DEVDIR, 'results')
OUT = os.path.join(RES, T3.TAG.rsplit('_seed', 1)[0] + '_M3') if KA else os.path.join(RES, {'cnetA': 'a16_bwA_M3', 'cnetS': 'a16_bsA_M3'}[WX] if WX else 'a16_M3' if C0 else 'a15_noB_M3' if NOB else 'a15_M3'); os.makedirs(OUT, exist_ok=True)
for d in ('a11_peakfix', T3.TAG, 'a15_nets'):                          # 两个模块 import 时建的空目录（Uddin 命名），没东西就删
    d = os.path.join(RES, d)
    if os.path.isdir(d) and not os.listdir(d):
        os.rmdir(d)
T3.NETDIR = os.path.join(RES, 'a15_M3'); PF.OUT = OUT                 # C 的权重总在 a15_M3/（cnet.pt、cnet0.pt）
if WX:
    METH = ['gate', 't3', 't3c', 'fix', 'A11', 'A11p', 'orc']
    MAIN = {'cnetA': 'A16bw', 'cnetS': 'A16bs'}[WX]
    KEY = dict(gate='gate', t3='Braw', t3c='Bc', fix='fix', A11='AB', A11p=MAIN, orc='orc')
    OTH = ['A16', 'B0', 'A15', 'XieMLP']
elif C0:                                                                   # 用户 2026-11-03：这一版叫 A16；fix / orc / 计时 见 a15_three 文件头
    METH = ['gate', 't3', 't3c', 'fix', 'A11', 'A11p', 'orc']
    KEY = dict(gate='gate', t3='t3raw', t3c='t3c0', fix='fix', A11='AC0', A11p='A16', orc='orc'); MAIN = 'A16'
    OTH = ['A15', 'A15noB', 'A15free', 'A11', 'A12', 'XieMLP']
    if KA:
        KEY['A11p'] = MAIN = 'A16A'; OTH = ['A16', 'A15', 'A12', 'XieMLP']
elif NOB:
    METH = ['gate', 't3c', 'A11', 'A11p']; KEY = dict(gate='gate', t3c='t3c', A11='AC', A11p='A15noB'); MAIN = 'A15noB'
    OTH = ['A15', 'A11', 'A12', 'XieMLP']
else:
    METH = ['gate', 'cons', 't3c', 'A2', 'A11', 'A11p']                 # a15_three 的键：A11 = 三支融合，A11p = A15
    KEY = dict(gate='gate', cons='cons', t3c='t3c', A2='two', A11='three', A11p='A15'); MAIN = 'A15'
    OTH = ['A11', 'A12', 'XieMLP']                                      # 读旧结果对照的
NAME = dict(gate='门控支', cons='原网络支', t3c='互补支', two='两支融合', three='三支融合', A15='A15', A11='A11', A12='A12', XieMLP='Xie MLP',
            AC='A+C融合', A15noB='A15·无B', t3c0='C一致化', AC0='A+C融合', A15free='上次A16', A16='A16', t3raw='C原始', fix='固定权重', orc='最优上限', A16A=T3.ALG,
            Braw='B原始', Bc='B一致化', AB='A+B融合', A16bw='A16·B按A加权', A16bs='A16·B看A', B0='A16的B')
PKF = PF.PKF; NB5 = 5
log = PF.log


def main():
    t0 = time.time()
    decode, P, used = AD.build()                                        # = T3.build3
    log(f'A15（Cui · M3 重训）建好 {time.time() - t0:.0f} s')

    # 1. 峰位校正 q（= a11_peakfix.main 第 1 步）：融合验证集的谱，新抽一份 M3 训练口径读数
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

    def dec2(Y):
        o = decode(Y); Fx, xa, d = PF.featx(o); q = PF.q_of(Fx, mu, sd, qnet)
        o['A11p'] = P18.unflat(PF.shift_np(xa, q * d), Y.shape[0]); o['q'] = q; o['d'] = d
        return {KEY[m]: o[m] for m in METH} | dict(q=q, d=d)

    M = [KEY[m] for m in METH]
    A0 = json.load(open(os.path.join(RES, 'a11_M3', 'a11.json'), encoding='utf-8'))
    A12 = json.load(open(os.path.join(RES, 'a12_M3', 'a12.json'), encoding='utf-8'))
    XJ = json.load(open(os.path.join(RES, 'all_algos_M3', 'all_algos.json'), encoding='utf-8'))
    A15J = json.load(open(os.path.join(RES, 'a15_M3', 'a15.json'), encoding='utf-8')) if NOB else None
    A16J = json.load(open(os.path.join(RES, 'a16_M3', 'a15.json'), encoding='utf-8')) if WX or KA else None
    NBJ = json.load(open(os.path.join(RES, 'a15_noB_M3', 'a15.json'), encoding='utf-8')) if C0 else None
    PRV = json.load(open(os.path.join(RES, 'a15_cnet0_noB_M3', 'a15.json'), encoding='utf-8')) if C0 else None   # 11-02 同配置那次（核对复现）
    src = lambda m: A16J if m in ('A16', 'B0') else A15J if m == 'A15' and NOB else NBJ if m == 'A15noB' else PRV if m == 'A15free' else A12 if m in ('A11', 'A12') else XJ                  # a12.json 里 A11 / A12 都有（含 fampk）；其他算法读 all_algos.json
    rk = lambda m: 't3raw' if m == 'B0' else m
    ref = lambda sec, nm, m: src(m)[sec][nm][rk(m)] if sec in ('fam', 'fampk') else src(m)[sec][rk(m)]['dpk']
    J = dict(lam=lam, fam={}, fampk={}, real={}, sim={}); demo = {}

    # 2a. 生成谱族 · M3 读数
    sets = {k.replace('留出·', ''): v for k, v in AD.all_fams(AD.TEST_ST, AD.NTE, P, False).items()}
    leak = P18.overlap(sets, used); assert not leak, leak
    E = lambda X, S: np.array([LD.band_nrmse(X[r], S.T) for r in range(nd)])
    g = np.random.default_rng(AD.SEED_SIM + 3)
    for nm, S in sets.items():
        o = dec2(AD.meas(S.T, g, 'm3'))
        J['fam'][nm] = {m: float(E(o[m], S).mean()) for m in M}
        demo[nm] = {m: o[m][0, :, 0] for m in M}
        if nm in PKF:
            tp = PF.peaks(S)
            J['fampk'][nm] = {m: float(np.median(np.abs(PF.peaks(P18.flat(o[m])) - np.tile(tp, nd)))) for m in M}
    A0f = {k.replace('留出·', ''): v for k, v in A0['fam']['m3'].items()}
    dif = max(abs(J['fam'][k]['two'] - A0f[k]['A11']) for k in sets) if not NOB else float('nan')
    log(f'\n== 生成谱族 · M3 读数（a = 1；每族 {AD.NTE} 条 × {nd} 次；形状误差 = 峰值归一均方根 / 真值极差）  两支融合（本次重拟）与 a11.json 的 A11 最大差 {dif:.2e}')
    log('   族                    ' + ''.join(f'{NAME[m]:>9s}' for m in M + OTH))
    for nm in sets:
        log(f'   {nm:<20s}' + ''.join(f'{J["fam"][nm][m]:9.4f}' for m in M) + ''.join(f'{ref("fam", nm, m):9.4f}' for m in OTH))
    med = {m: float(np.median([J['fam'][k][m] for k in sets])) for m in M} | {m: float(np.median([ref('fam', k, m) for k in sets])) for m in OTH}
    J['med'] = med
    log('   各族中位              ' + ''.join(f'{med[m]:9.4f}' for m in M + OTH))
    for m in OTH:
        w = sum(J['fam'][k][MAIN] < ref('fam', k, m) - 1e-4 for k in sets); l = sum(J['fam'][k][MAIN] > ref('fam', k, m) + 1e-4 for k in sets)
        log(f'   {NAME[MAIN]} 对 {NAME[m]}：形状误差好 {w} 族、差 {l} 族（差 > 1e-4）')
    for nm in PKF:
        log(f'   {nm} 峰位 |误差| 中位（nm）：' + '，'.join(f'{NAME[m]} {J["fampk"][nm][m]:.3f}' for m in M) + ''.join(f'，{NAME[m]} {ref("fampk", nm, m):.3f}' for m in OTH))
    log(f'   用时 {time.time() - t0:.0f} s')

    # 2b. 作者 6 束真实读数
    LT = AD.LAM_TEST.astype(float); rm = lambda a: float(np.sqrt(np.nanmean(np.square(a))))
    o = dec2((AD.Y_REAL / AD.SCALE)[None])
    for m in M:
        J['real'][m] = dict(dpk=list(PF.peaks(o[m][0].T) - LT))
    J['real']['q'] = list(o['q']); J['real']['d'] = list(o['d'])
    rchk = max(abs(x - y) for x, y in zip(A0['real']['A11']['dpk'], J['real']['two']['dpk'])) if not NOB else float('nan')
    log(f'\n== 作者 6 束真实读数（{", ".join(f"{v:.0f}" for v in LT)} nm，参考 = 标称波长）：峰位误差 nm   RMSD 只算前 {NB5} 束   （两支融合与 a11.json 的 A11 最大差 {rchk:.2e}）')
    for m in [m for m in M if m != 'orc'] + OTH:                         # orc 只在谱族有真值时才有意义
        dp = np.array(J['real'][m]['dpk'] if m in M else ref('real', None, m), float)
        log(f'   {NAME[m]:<8s}' + ' '.join(f'{v:+7.2f}' for v in dp) + f'   RMSD(5) {rm(dp[:NB5]):.2f}')
    log('   q：' + ' '.join(f'{v:.2f}' for v in o['q']) + '   d：' + ' '.join(f'{v:+.2f}' for v in o['d']))

    # 2c. 同一批光的模拟 M3 读数
    rng = np.random.default_rng(AD.SEED_SIM + 52); KB = AD.M3_B.shape[1]; J['sim'] = {m: dict(dpk=[]) for m in M}; qs = []
    for j, lt in enumerate(LT):
        I0 = AD.DV.measure([lt], [AD.P0], lines=True, warn=False).ravel()
        Yb = np.array([AD.U.T @ (np.tile(1 + AD.M3_B @ rng.standard_normal(KB), AD._nV) * I0 + AD.FW.REAL['recon'] * rng.standard_normal(I0.size))
                       for _ in range(AD.NSIM)])[:, :, None] / AD.SCALE
        o = dec2(Yb); qs.append(float(o['q'].mean()))
        for m in M:
            J['sim'][m]['dpk'].append(list(PF.peaks(o[m][:, :, 0]) - lt))
    schk = np.nanmax(np.abs(np.array(A0['sim']['m3']['A11']['dpk'], float) - np.array(J['sim']['two']['dpk']))) if not NOB else float('nan')
    log(f'\n== 同一批光的模拟 M3 读数（各 {AD.NSIM} 次）：每束 |峰位误差| 平均 nm；RMSD / 中位只算前 {NB5} 束   （两支融合与 a11.json 的 A11 最大差 {schk:.2e}）')
    for m in [m for m in M if m != 'orc'] + OTH:
        dp = np.array(J['sim'][m]['dpk'] if m in M else ref('sim', None, m), float)
        log(f'   {NAME[m]:<8s}' + ' '.join(f'{v:6.2f}' for v in np.nanmean(np.abs(dp), 1)) + f'   RMSD(5) {rm(dp[:NB5]):6.2f}   中位 {np.nanmedian(np.abs(dp[:NB5])):.3f}')
    log('   各束 q 均值：' + ' '.join(f'{v:.2f}' for v in qs))
    log(f'\n用时 {time.time() - t0:.0f} s')
    if (C0 or WX) and T3.TM['n']:
        log(f'推理计时（本次运行全部 decode，{T3.TM["n"]} 条读数）：门控支 {T3.TM["gate"] / T3.TM["n"] * 1e3:.2f} ms/条，C 支 {T3.TM["t3"] / T3.TM["n"] * 1e3:.3f} ms/条')
    json.dump(J, open(os.path.join(OUT, 'a15.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
    pickle.dump(demo, open(os.path.join(OUT, 'demo.pkl'), 'wb'))
    open(os.path.join(OUT, 'log.txt'), 'w', encoding='utf-8').write('\n'.join(PF._L))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    import warnings; warnings.filterwarnings('ignore')
    main()
