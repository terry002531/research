# -*- coding: utf-8 -*-
"""A12 两支诊断（只 Uddin，真实模式读数，重放 a11_peakfix 的测试读数 SEED_SIM + 1）。不训练任何东西。

用户 2026-11-02："门控支本来就不主打峰类，网络支又不主打平滑宽谱，两个是互补的，会不会这两支已经足够好了，还是最后选择的时候有问题？"
逐族比：
  网络支：原始 MLP 输出（一致化前）/ 一致化后 / Xie MLP（同训练库、同噪声模型、同步数，只是结构和输入归一不同）
  门控支：门控输出 / 本族自己那一类的解（非峰族才有）/ 本族那一类的后验权重 / 各类解与三个峰候选里事后最好的一个
  选择：逐条在 {门控, 一致化} 里挑好的、在 {门控, 一致化, Xie} 里挑好的 —— 和外部最好比，看"两支 + 完美选择"够不够
  py -I -X utf8 a12_branch_diag.py      输出 45_Uddin2024_真实R矩阵/results/a12_fusion/{branch_diag.json, branch_log.txt}
"""
import os, sys, json, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.argv = [os.path.join(ROOT, 'a11_peakfix.py')]
sys.path.insert(0, ROOT)
import a11_peakfix as PF
import torch
import a12_examples as EX

AD, P18, LD, DS = PF.AD, PF.P18, PF.LD, PF.AD.DS
WL, BM, DEV = AD.WL, AD.BM, PF.DEV
OUT = os.path.join(AD.DEVDIR, 'results', 'a12_fusion'); os.makedirs(OUT, exist_ok=True)
_L = []; _G = {}


def log(s=''):
    print(s, flush=True); _L.append(s)


_fp0 = DS.fuse_parts


def _fp(Rn, Xk, W, dP, cands, Y, GM, classes):
    """decode 里门控的各类解 Xk、后验 W、三个峰候选存下来。"""
    p = _fp0(Rn, Xk, W, dP, cands, Y, GM, classes); _G.update(Xk=Xk, W=W, C=p['C'], classes=classes)
    return p


DS.fuse_parts = _fp


def nr(X, S):
    """X (nd, NW, M)，S (M, NW) → (nd, M)。"""
    return np.array([LD.band_nrmse(X[r], S.T) for r in range(X.shape[0])])


def main():
    t0 = time.time(); RES = os.path.join(AD.DEVDIR, 'results')
    decode, P, used = AD.build()
    xs = EX.XieMLP(AD.N_RD, int(BM.sum())).to(DEV); xs.load_state_dict(torch.load(os.path.join(RES, 'all_algos', 'xie_samelib.pt'))); xs.eval()
    XJ = json.load(open(os.path.join(RES, 'all_algos', 'all_algos.json'), encoding='utf-8'))
    XF = {k.replace('留出·', ''): v for k, v in XJ['fam']['real'].items()}
    PK = json.load(open(os.path.join(PF.OUT, 'peakfix.json'), encoding='utf-8'))['fam']['real']
    ext = [m for m in XJ['name'] if m in XF[next(iter(XF))] and m not in ('A11', 'A11_R')]
    sets = {k.replace('留出·', ''): v for k, v in AD.all_fams(AD.TEST_ST, AD.NTE, P, False).items()}
    g = np.random.default_rng(AD.SEED_SIM + 1); nd = LD.NDRAW; R = {}
    for nm, S in sets.items():
        Y = AD.meas(S.T, g, 'real'); M = S.shape[0]; o = decode(Y)
        assert abs(nr(o['A11'], S).mean() - PK[nm]['A11']) < 1e-9
        Yf = Y.transpose(0, 2, 1).reshape(nd * M, -1)
        torch.backends.cuda.matmul.allow_tf32 = True
        with torch.no_grad():
            y = torch.tensor(Yf, dtype=torch.float32, device=DEV); xb = torch.nn.functional.softplus(xs(y / (y.abs().sum(1, keepdim=True) + 1e-30))).double().cpu().numpy()
        torch.backends.cuda.matmul.allow_tf32 = False
        Xx = np.zeros((nd * M, len(WL))); Xx[:, BM] = xb; Xx = P18.unflat(Xx, nd)
        e = dict(gate=nr(o['gate'], S), net=nr(o['net'], S), cons=nr(o['cons'], S), xie=nr(Xx, S), A11=nr(o['A11'], S))
        cls = [c.replace('留出·', '') for c in _G['classes']]; W = _G['W']                         # W (nd, M, K)
        ek = np.stack([nr(np.clip(_G['Xk'][i], 0, None), S) for i in range(len(cls))])            # 各类解 (K, nd, M)
        ec = np.stack([nr(c.astype(float), S) for c in _G['C']])                                  # 峰候选 ParamE / A8 / 峰类均值
        r = {m: float(v.mean()) for m, v in e.items()}
        r['pick2'] = float(np.minimum(e['gate'], e['cons']).mean()); r['pick3'] = float(np.minimum(np.minimum(e['gate'], e['cons']), e['xie']).mean())
        r['bestpart'] = float(np.minimum(ek.min(0), ec.min(0)).mean())
        if nm in cls:
            i = cls.index(nm); r['own'] = float(ek[i].mean()); r['own_w'] = float(W[:, :, i].mean()); r['own_top'] = float((W.argmax(2) == i).mean())
        top = np.bincount(W.argmax(2).ravel(), minlength=len(cls)); r['top_cls'] = cls[int(top.argmax())]; r['top_frac'] = float(top.max() / top.sum())
        r['ext'] = min(ext, key=lambda m: XF[nm][m]); r['ext_v'] = XF[nm][r['ext']]
        assert abs(r['xie'] / XF[nm]['XieMLP'] - 1) < 0.01, (nm, r['xie'], XF[nm]['XieMLP'])
        R[nm] = r
    log(f'重放核对：A11 与 peakfix.json 一致，Xie 与 all_algos.json 相差 < 1 %；{time.time() - t0:.0f} s')

    f = lambda v: f'{v:8.4f}' if isinstance(v, float) else f'{"—":>8s}'
    log('\n== 网络支（真实模式，族平均形状误差）      原始网络  一致化后    Xie   一致化相对原始')
    for k, r in R.items():
        log(f'   {k:<18s}' + f(r['net']) + f(r['cons']) + f(r['xie']) + f'   {(r["cons"] / r["net"] - 1) * 100:+6.1f} %')
    log('\n== 门控支                                门控输出  本族类的解 本族类后验 判为本族比例 事后最好零件   门控最常判成')
    for k, r in R.items():
        log(f'   {k:<18s}' + f(r['gate']) + f(r.get('own')) + f(r.get('own_w')) + f(r.get('own_top')) + f(r['bestpart']) + f'   {r["top_cls"]}（{r["top_frac"] * 100:.0f} %）')
    log('\n== 两支 + 完美逐条选择 够不够（和外部最好比）  A12    挑{门控,一致化}  挑{+Xie}   外部最好')
    for k, r in R.items():
        a12 = PK[k]['A11p']
        log(f'   {k:<18s}' + f(a12) + f(r['pick2']) + f(r['pick3']) + f(r['ext_v']) + f' {XJ["name"][r["ext"]]}' + ('   ← 完美选择也输' if r['pick2'] > r['ext_v'] else ''))
    md = lambda m: float(np.median([R[k][m] for k in R])); mn = lambda m: float(np.mean([R[k][m] for k in R]))
    log('\n   族中位 / 族均值：' + '，'.join(f'{m} {md(m):.4f} / {mn(m):.4f}' for m in ('gate', 'net', 'cons', 'xie', 'A11', 'pick2', 'pick3')))
    log(f'   完美挑 {{门控, 一致化}} 仍输外部最好：{sum(R[k]["pick2"] > R[k]["ext_v"] for k in R)} 族；加上 Xie 后仍输：{sum(R[k]["pick3"] > R[k]["ext_v"] for k in R)} 族')
    log(f'用时 {time.time() - t0:.0f} s')
    json.dump(R, open(os.path.join(OUT, 'branch_diag.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    open(os.path.join(OUT, 'branch_log.txt'), 'w', encoding='utf-8').write('\n'.join(_L))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    import warnings; warnings.filterwarnings('ignore')
    main()
