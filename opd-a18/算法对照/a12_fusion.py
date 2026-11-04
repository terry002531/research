# -*- coding: utf-8 -*-
"""A12 重做融合选择（只 Uddin，评估只看真实模式读数）。a11_device.py / a11_peakfix.py 不改，导入它们。

用户 2026-11-02："先重做融合选择"——最后一步 x = w·网络支 + (1−w)·门控支 的 w 怎么定。
问题：A12 在真实模式 26 族里有 16 族不如"固定只用其中一支"。

两步：
  collect  跑一遍 A11 解码，把每条样本的两支输出、融合特征、旧 w 存成 npz（放 scratchpad，不进项目）：
           F 集 = 新种子 FSEED 的谱（训练族，读数 train 模式），给新融合用；
           Q 集 = 融合验证集谱 + QSEED 读数（= a11_peakfix 训练 q 的那份），给峰位校正 q 用；
           T 集 = 测试谱、真实模式读数，重放 a11_peakfix（种子 SEED_SIM + 1），用来核对 A12 与 peakfix.json 一致。
  fit      离线：诊断上限（逐条挑一支、逐条最优 w、每族固定 w）→ 训练新的 w → 在 Q 集上重训 q → 测试集上和 A12 比。
  py -I -X utf8 a12_fusion.py collect|fit
"""
import os, sys, json, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
MODE = sys.argv[1] if len(sys.argv) > 1 else 'fit'
sys.argv = [os.path.join(ROOT, 'a11_peakfix.py')]
sys.path.insert(0, ROOT)
import a11_peakfix as PF
import torch

AD, P18, LD = PF.AD, PF.P18, PF.LD
WL, BM, DEV = AD.WL, AD.BM, PF.DEV
CACHE = os.path.join(os.environ.get('A12F_CACHE', r'C:\Users\Shi\AppData\Local\Temp\claude\D--OneDrive-Terry-01-Research-08-OPD-learned-sensing\da47454e-15e5-4b1d-b0aa-28f30ea62edf\scratchpad'), 'a12_fusion_cache')
OUT = os.path.join(AD.DEVDIR, 'results', 'a12_fusion'); os.makedirs(OUT, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
FSEED, NF = 7777, 160                                   # 新融合训练谱：每族 NF 条
_L = []


def log(s=''):
    print(s, flush=True); _L.append(s)


def run(decode, sets, g, mode):
    """sets {族: (M, NW)} → 每条样本的 Xc, Xg, f8, lw, 旧 w, St, 族号（flat 顺序 = (读数, 谱)）。"""
    nd = LD.NDRAW; D = {k: [] for k in ('Xc', 'Xg', 'f8', 'lw', 'w', 'St', 'fam')}
    for j, (k, S) in enumerate(sets.items()):
        o = decode(AD.meas(S.T, g, mode)); f8, lw = PF._F['last']
        D['Xc'].append(P18.flat(o['cons'])); D['Xg'].append(P18.flat(o['gate'])); D['f8'].append(f8); D['lw'].append(lw)
        D['w'].append(o['w'].reshape(-1)); D['St'].append(np.tile(S, (nd, 1))); D['fam'].append(np.full(nd * len(S), j))
        assert np.allclose(P18.flat(o['A11']), D['w'][-1][:, None] * D['Xc'][-1] + (1 - D['w'][-1][:, None]) * D['Xg'][-1])
    out = {k: np.concatenate(v) for k, v in D.items()}
    for k in ('Xc', 'Xg', 'St'):
        out[k] = out[k].astype(np.float32)
    return out


def collect():
    t0 = time.time(); decode, P, used = AD.build(); log(f'A11 建好 {time.time() - t0:.0f} s')
    test = {k.replace('留出·', ''): v for k, v in AD.all_fams(AD.TEST_ST, AD.NTE, P, False).items()}
    fset = AD.all_fams(FSEED, NF, P, True)
    leak = P18.overlap(test, dict(used, F集=np.concatenate(list(fset.values())))); assert not leak, leak
    names = list(test); assert [k.replace('留出·', '') for k in fset] == names, list(fset)
    T = run(decode, test, np.random.default_rng(AD.SEED_SIM + 1), 'real'); log(f'T 集 {len(T["w"])} 条 {time.time() - t0:.0f} s')
    Qs = AD.all_fams(P18.VSEED, AD.NVF, P, True); Qd = run(decode, Qs, np.random.default_rng(PF.QSEED), 'train'); log(f'Q 集 {len(Qd["w"])} 条')
    Fd = run(decode, fset, np.random.default_rng(FSEED + 1), 'train'); log(f'F 集 {len(Fd["w"])} 条 {time.time() - t0:.0f} s')
    for nm, d in (('T', T), ('Q', Qd), ('F', Fd)):
        np.savez(os.path.join(CACHE, f'{nm}.npz'), names=np.array(names), **d)
    log(f'存到 {CACHE}')


# ------------------------------------------------------------------ 离线部分
BMt = torch.tensor(BM, device=DEV); NB = int(BM.sum()); NBIN = 24
WG = np.linspace(0, 1, 21)                              # 逐条最优 w 的网格（诊断用）


def load(nm):
    z = np.load(os.path.join(CACHE, f'{nm}.npz'))
    return {k: z[k].astype(np.float64) if z[k].dtype == np.float32 else z[k] for k in z.files}


def nrmse(X, S):
    """逐行，= LD.band_nrmse：截负、峰值归一后均方根 / 真值极差（波段内）。"""
    x, s = np.clip(X[:, BM], 0, None), S[:, BM]
    xn = x / np.maximum(x.max(1, keepdims=True), 1e-30); sn = s / s.max(1, keepdims=True)
    return np.sqrt(((xn - sn) ** 2).mean(1)) / (sn.max(1) - sn.min(1))


def blend(D, w):
    return w[:, None] * D['Xc'] + (1 - w[:, None]) * D['Xg']


def bins(X):
    """波段内峰值归一后分 NBIN 段平均。"""
    x = np.clip(X[:, BM], 0, None); x = x / np.maximum(x.max(1, keepdims=True), 1e-30)
    return np.stack([c.mean(1) for c in np.array_split(x, NBIN, axis=1)], 1)


def feat(D, kind):
    F0 = np.c_[D['f8'], D['lw']]
    if kind == 'old':
        return F0
    bc, bg = bins(D['Xc']), bins(D['Xg'])
    return np.c_[F0, bc, bg, bc - bg]


def fit_w(Fv, D, kind, tr, va, hid, steps=4000, seed=5):
    """w = sigmoid(net(特征))；损失 = 融合后形状误差（同 algo18.fit_blend_net），按验证部分早停。"""
    torch.manual_seed(seed); T = lambda a: torch.tensor(a, dtype=torch.float64, device=DEV)
    mu, sd = Fv.mean(0), Fv.std(0) + 1e-9; f = T((Fv - mu) / sd)
    xc, xg, s = T(D['Xc'][:, BM]), T(D['Xg'][:, BM]), T(D['St'][:, BM])
    sn = s / s.amax(1, keepdim=True); rg = sn.amax(1) - sn.amin(1)
    L = [torch.nn.Linear(f.shape[1], hid[0]), torch.nn.Tanh()]
    for a, b in zip(hid[:-1], hid[1:]):
        L += [torch.nn.Linear(a, b), torch.nn.Tanh()]
    net = torch.nn.Sequential(*L, torch.nn.Linear(hid[-1], 1)).double().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), 3e-3, weight_decay=1e-2); itr, iva = T(tr).long(), T(va).long()

    def loss(i):
        w = torch.sigmoid(net(f[i])); x = (w * xc[i] + (1 - w) * xg[i]).clamp(min=0)
        xn = x / (x.amax(1, keepdim=True) + 1e-30)
        return (torch.sqrt(((xn - sn[i]) ** 2).mean(1)) / rg[i]).mean()
    best = (np.inf, None, 0)
    for it in range(steps):
        l = loss(itr); opt.zero_grad(); l.backward(); opt.step()
        if it % 50 == 0:
            with torch.no_grad():
                lv = float(loss(iva))
            if lv < best[0]:
                best = (lv, {k: v.clone() for k, v in net.state_dict().items()}, it)
    net.load_state_dict(best[1])

    def w_of(D2):
        with torch.no_grad():
            return torch.sigmoid(net(T((feat(D2, kind) - mu) / sd)))[:, 0].cpu().numpy()
    return w_of, best[2]


def featq(D, w):
    """= a11_peakfix.featx，用存下的两支与新的 w。"""
    Xa = blend(D, w); pa, pg = PF.peaks(Xa), PF.peaks(D['Xg']); d = pg - pa; de = np.where(np.abs(d) <= PF.DMAX, d, 0.0)
    Fx = np.c_[D['f8'], D['lw'], de, np.abs(de), (pa - WL[BM].mean()) / np.ptp(WL[BM]), w, PF.width(Xa), PF.width(D['Xg'])]
    return Fx, Xa, de


def fit_q(Q, w):
    """照 a11_peakfix.main 第 1 步：λ 扫 LAMS，形状不差于未校正 0.5 % 以内、单峰峰位最小。"""
    Fv, Xa, de = featq(Q, w); St = Q['St']
    idx = np.random.default_rng(5).permutation(len(Fv)); tr = (idx[:int(0.7 * len(idx))], idx[int(0.7 * len(idx)):]); va = tr[1]
    pkm = PF.single_peak(St[va]); pt = PF.peaks(St[va]); e0 = nrmse(Xa[va], St[va]).mean(); cand = {}
    for lam in PF.LAMS:
        mu, sd, net = PF.fit_q(Fv, Xa, de, St, lam, tr); q = PF.q_of(Fv[va], mu, sd, net); Xp = PF.shift_np(Xa[va], q * de[va])
        cand[lam] = (nrmse(Xp, St[va]).mean(), np.abs(PF.peaks(Xp) - pt)[pkm].mean(), mu, sd, net)
    ok = [l for l in PF.LAMS if cand[l][0] <= e0 * 1.005] or [min(PF.LAMS, key=lambda l: cand[l][0])]
    lam = min(ok, key=lambda l: cand[l][1]); _, _, mu, sd, net = cand[lam]

    def apply(D, w2):
        Fx, Xa2, d2 = featq(D, w2)
        return PF.shift_np(Xa2, PF.q_of(Fx, mu, sd, net) * d2)
    return apply, lam


def famtab(fam, E):
    """E {方法: 逐条误差} → {方法: (族数,) 族平均}。"""
    nf = fam.max() + 1
    return {m: np.array([e[fam == j].mean() for j in range(nf)]) for m, e in E.items()}


def fit():
    t0 = time.time(); Tt, Q, Fd = load('T'), load('Q'), load('F'); names = [str(n) for n in Tt['names']]; nf = len(names)
    PK = json.load(open(os.path.join(PF.OUT, 'peakfix.json'), encoding='utf-8'))
    XJ = json.load(open(os.path.join(AD.DEVDIR, 'results', 'all_algos', 'all_algos.json'), encoding='utf-8'))
    XF = {k.replace('留出·', ''): v for k, v in XJ['fam']['real'].items()}
    ext = [m for m in XJ['name'] if m in XF[names[0]] and m not in ('A11', 'A11_R')]
    log(f'读缓存：T {len(Tt["w"])}、Q {len(Q["w"])}、F {len(Fd["w"])} 条')

    # 0. 复现 A12（读 qnet.pt）
    qz = torch.load(os.path.join(PF.OUT, 'qnet.pt'), weights_only=False)
    qn = torch.nn.Sequential(torch.nn.Linear(len(qz['mu']), 32), torch.nn.Tanh(), torch.nn.Linear(32, 1)).double().to(DEV); qn.load_state_dict(qz['state']); qn.eval()
    Fx, Xa, de = featq(Tt, Tt['w']); X12 = PF.shift_np(Xa, PF.q_of(Fx, qz['mu'], qz['sd'], qn) * de)
    E = dict(gate=nrmse(Tt['Xg'], Tt['St']), cons=nrmse(Tt['Xc'], Tt['St']), A11=nrmse(Xa, Tt['St']), A12=nrmse(X12, Tt['St']))
    FT = famtab(Tt['fam'], E); d12 = max(abs(FT['A12'][j] - PK['fam']['real'][names[j]]['A11p']) for j in range(nf))
    log(f'复现 A12：与 peakfix.json 逐族最大差 {d12:.1e}'); assert d12 < 1e-6

    # 1. 诊断：上限
    def oracle(D):
        eg = np.stack([nrmse(blend(D, np.full(len(D['w']), w)), D['St']) for w in WG], 1)
        return eg, eg.min(1), np.minimum(nrmse(D['Xc'], D['St']), nrmse(D['Xg'], D['St']))
    for nm, D in (('T（真实模式测试）', Tt), ('F（新融合训练集）', Fd)):
        eg, eo, ep = oracle(D); e_old = nrmse(blend(D, D['w']), D['St'])
        ff = famtab(D['fam'], dict(old=e_old, pick=ep, ow=eo, **{f'w{w:.2f}': eg[:, i] for i, w in enumerate(WG)}))
        fixed = np.min(np.stack([ff[f'w{w:.2f}'] for w in WG]), 0)
        med = lambda a: float(np.median(a))
        log(f'\n== 上限诊断 · {nm}：族中位  旧 w 融合 {med(ff["old"]):.4f}｜每族最好固定 w {med(fixed):.4f}｜逐条挑一支 {med(ff["pick"]):.4f}｜逐条最优 w {med(ff["ow"]):.4f}')
        log(f'   旧 w 分布：<0.1 {np.mean(D["w"] < .1) * 100:.0f} %，0.1–0.9 {np.mean((D["w"] >= .1) & (D["w"] <= .9)) * 100:.0f} %，>0.9 {np.mean(D["w"] > .9) * 100:.0f} %；'
            f'逐条最优 w 分布：<0.1 {np.mean(WG[eg.argmin(1)] < .1) * 100:.0f} %，中间 {np.mean((WG[eg.argmin(1)] >= .1) & (WG[eg.argmin(1)] <= .9)) * 100:.0f} %，>0.9 {np.mean(WG[eg.argmin(1)] > .9) * 100:.0f} %')
        if D is Tt:
            UP = dict(fixed=fixed, pick=ff['pick'], ow=ff['ow'])

    # 1b. 不学习的规则选法（规则里要定的数都在 F 集上定，不看测试集）
    E['规则·选更吻合读数的一支'] = nrmse(blend(Tt, (Tt['f8'][:, 0] < Tt['f8'][:, 1]).astype(float)), Tt['St'])   # f8 前两列 = 两支对读数的 log χ²
    egF = np.stack([nrmse(blend(Fd, np.full(len(Fd['w']), w)), Fd['St']) for w in WG], 1)
    cF, cT = Fd['lw'].argmax(1), Tt['lw'].argmax(1)                                 # 门控判的类别（实测时可得）
    wc = {c: WG[egF[cF == c].mean(0).argmin()] for c in np.unique(cF)}
    E['规则·按门控类别定 w'] = nrmse(blend(Tt, np.array([wc.get(c, 1.0) for c in cT])), Tt['St'])
    wfam = np.array([WG[egF[Fd['fam'] == j].mean(0).argmin()] for j in range(nf)])
    E['参考·按真实族定 w（需知族）'] = nrmse(blend(Tt, wfam[Tt['fam']]), Tt['St'])
    log(f'\n规则：门控类别 → w：' + '，'.join(f'{c}:{v:.2f}' for c, v in wc.items()) + f'；按族 w：' + ' '.join(f'{v:.2f}' for v in wfam))

    # 2. 新 w：在 F 集上训练（按谱分 80 / 20，早停用 20 %）
    nfam_sp = {j: (Fd['fam'] == j).sum() // LD.NDRAW for j in range(nf)}
    spec = np.concatenate([np.tile(np.arange(nfam_sp[j]), LD.NDRAW) + 100000 * j for j in range(nf)])   # flat 顺序：族内 (读数, 谱)
    us = np.unique(spec); vs = set(np.random.default_rng(3).choice(us, len(us) // 5, replace=False))
    va = np.array([s in vs for s in spec]); tr, va = np.where(~va)[0], np.where(va)[0]
    VAR = {'w0：旧特征旧网（只换训练数据）': ('old', (32,)), 'w1：加两支形状特征': ('shape', (64, 64))}
    res, XO = {}, {'A12': X12}
    for vn, (kind, hid) in VAR.items():
        w_of, it = fit_w(feat(Fd, kind), Fd, kind, tr, va, hid)
        ef = nrmse(blend(Fd, w_of(Fd)), Fd['St'])[va].mean(); eo = nrmse(blend(Fd, Fd['w']), Fd['St'])[va].mean()
        wq, wt = w_of(Q), w_of(Tt); qa, lam = fit_q(Q, wq)
        XO[vn] = qa(Tt, wt); E[vn] = nrmse(blend(Tt, wt), Tt['St']); E[vn + ' + 峰位校正'] = nrmse(XO[vn], Tt['St'])
        res[vn] = dict(it=it, lam=lam, ef=ef, eo=eo)
        log(f'\n{vn}：F 留出 20 % 形状误差 {ef:.4f}（旧 w {eo:.4f}），早停于第 {it} 步；q 重训选 λ = {lam:g}；{time.time() - t0:.0f} s')

    # 3. 测试集逐族对比
    FT = famtab(Tt['fam'], E); FT.update(UP); ms = list(E) + ['fixed', 'pick', 'ow']
    nmc = {'fixed': '上限·每族固定w', 'pick': '上限·逐条挑支', 'ow': '上限·逐条最优w'}
    log(f'\n== 测试集真实模式逐族（每族 {AD.NTE} 条 × {LD.NDRAW} 次）')
    log('   族' + ' ' * 16 + ''.join(f'{nmc.get(m, m)[:10]:>12s}' for m in ms) + '      外部最好')
    for j, k in enumerate(names):
        b = min(ext, key=lambda m: XF[k][m])
        log(f'   {k:<18s}' + ''.join(f'{FT[m][j]:12.4f}' for m in ms) + f'  {XF[k][b]:.4f} {XJ["name"][b]}')
    med = {m: float(np.median(FT[m])) for m in ms}
    log('   各族中位          ' + ''.join(f'{med[m]:12.4f}' for m in ms))
    for m in [m for m in E if m not in ('gate', 'cons', 'A11')]:
        lose = sum(min(FT['gate'][j], FT['cons'][j]) < FT[m][j] for j in range(nf))
        win = sum(FT[m][j] < min(XF[k][e] for e in ext) for j, k in enumerate(names))
        bt = sum(FT[m][j] < FT['A12'][j] - 1e-4 for j in range(nf)); wo = sum(FT[m][j] > FT['A12'][j] + 1e-4 for j in range(nf))
        log(f'   {m}：族中位 {med[m]:.4f}；不如固定单支 {lose} 族；赢全部外部算法 {win} 族；比 A12 好 {bt} 族、差 {wo} 族')
    for pf in PF.PKF:
        j = names.index(pf); i = Tt['fam'] == j; tp = PF.peaks(Tt['St'][i])
        log(f'   {pf} 峰位 |误差| 中位 nm：' + '，'.join(f'{m} {np.median(np.abs(PF.peaks(X[i]) - tp)):.3f}' for m, X in XO.items()))
    json.dump(dict(names=names, fam={m: FT[m].tolist() for m in ms}, med=med, res=res), open(os.path.join(OUT, 'fusion.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    open(os.path.join(OUT, 'log.txt'), 'w', encoding='utf-8').write('\n'.join(_L))
    log(f'用时 {time.time() - t0:.0f} s')


def plot():
    """各族相对 A12 的误差变化（%）：重训的两版（含峰位校正）、按真实族定 w（参考）、逐条最优 w（上限）。"""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
    J = json.load(open(os.path.join(OUT, 'fusion.json'), encoding='utf-8')); F = {m: np.array(v) for m, v in J['fam'].items()}; nm = J['names']
    S = [('w0：旧特征旧网（只换训练数据） + 峰位校正', 'w0 重训（旧特征，4 倍数据）+ 峰位校正', '#2a78d6', 'o'),
         ('w1：加两支形状特征 + 峰位校正', 'w1 重训（加两支形状）+ 峰位校正', '#1baf7a', 's'),
         ('参考·按真实族定 w（需知族）', '参考：按真实谱族定 w（实测不可得）', '#9a9890', 'D'),
         ('ow', '上限：逐条最优 w', '#e2591f', '|')]
    r = {m: (F[m] / F['A12'] - 1) * 100 for m, *_ in S}; order = np.argsort(r['ow'])[::-1]
    fg, ax = plt.subplots(figsize=(8.2, 8.6)); y = np.arange(len(nm))
    ax.axvline(0, color='#1f1f1e', lw=1)
    for m, lb, c, mk in S:
        ax.scatter(r[m][order], y, color=c, marker=mk, s=60 if mk == '|' else 28, lw=2 if mk == '|' else 0.6, edgecolor='white' if mk != '|' else None, zorder=3,
                   label=f'{lb}（族均值 {(F[m].mean() / F["A12"].mean() - 1) * 100:+.1f} %）')
    ax.set_yticks(y); ax.set_yticklabels([nm[i] for i in order], fontsize=8.5); ax.invert_yaxis()
    ax.set_xlabel('相对 A12 的形状误差变化（%，负 = 更好）'); ax.grid(axis='x', color='#e4e3df', lw=0.6); ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc='lower left', bbox_to_anchor=(0, 1.0))
    ax.set_title('重做融合选择：真实模式测试集各族（40 条 × 4 次）', fontsize=10.5, loc='left', pad=78)
    fg.tight_layout(); f = os.path.join(OUT, 'fig_fusion.png'); fg.savefig(f, dpi=150); print(f)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    import warnings; warnings.filterwarnings('ignore')
    {'collect': collect, 'fit': fit, 'plot': plot}[MODE]()
