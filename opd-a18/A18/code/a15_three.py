# -*- coding: utf-8 -*-
"""A15：三支融合（门控支 + 原网络支 + 第三支网络），后面照旧接峰位校正（只 Uddin，评估只看真实模式读数）。

用户 2026-11-02："试三支融合"（接 A12x 的结果：只把网络支换成 Xie，峰位变好、峰类族形状变差，两个网络各擅长不同谱形）；
  随后："第三支……可以借鉴他的思路，但是你不能用一模一样的这个算法呀，要不然创新性不够"。
第三支（--third）：
  xie    原样 Xie 2026b 网络（all_algos/xie_samelib.pt，同训练库、同噪声、同步数），只作对照；
  cnet   互补网络支（本文的）：借 Xie 的两点——读数 L1 归一输入、残差 MLP 骨架；三处不同——
         ① 互补加权：训练时原网络支冻结，在同一份带噪读数上算它的形状误差 e_o，样本权重 1 + min(e_o / 批均值, 3)，
            让第三支专学原网络支做不好的谱（为融合设计，而不是再训一个独立的全能网络）；
         ② 导数匹配损失：峰值归一谱的误差 + 0.5·r·一阶差分误差（r = 训练库真谱 值平方均值 / 差分平方均值，把两项配到同一量级），
            保峰位和边缘；
         ③ 结构：残差块 pre-LayerNorm + GELU，4 块 × 512（工程差异，不算创新点）。
         训练库、读数误差模型（meas_t）、30000 步 × 2048、OneCycle 1e-3 都同原网络和 Xie。
  cnet0  消融：同 cnet 但去掉 ① 互补加权（看 ① 自己带来多少）。
其余做法：a11_device.build 的流程照抄，只在融合这一步加第三支——
  第三支读数最小二乘定幅度后，和原网络支一样做一致化：用它自己在融合验证集上的误差协方差，κ 单独挑（κ = 0 即不修正）。
  融合小网：输入 = 原 8 个特征 + 门控各类对数后验 + 6 个第三支特征（第三支一致化后的 log χ²、它减原网络支的 log χ²、
    它与原网络支 / 门控支的峰值归一差、它的粗糙度、它与原网络支峰位差 |Δp|/20 nm 截到 1），
    32 个 tanh 隐单元 → 3 个 logit → softmax 权重 (w_g, w_c, w_3)；损失、70 / 30 分、早停、种子都同 algo18.fit_blend_net。
    融合验证集上同时重训原两支小网，作对照（列"两支融合"）。
  scale_y（亮度未知时估亮度）照旧用原网络。门控支、零件网络、原网络都不变。
  峰位校正：a11_peakfix.main 原样重训 q（QSEED 读数），q 的特征多出上面 6 个，融合权重那一列用 1 − w_g。
测试种子：--test-seed 4242（默认，和旧 A12、A12x 同一份读数）/ 6161（开发时没用过；对照读 a12x_orig_seed6161、a12x_xie_seed6161）。
  py -I -X utf8 a15_three.py [--third xie|cnet|cnet0] [--test-seed 4242] [--metrics] [--train-only] [--no-b]
--no-b（用户 2026-11-02 要的消融 "A 加 C 不要 B"）：推理时完全不用原网络支 B——C 一致化后顶替 B 在 8 个融合特征里的位置，
  第三支附加特征（都涉及 B）不用，融合小网 2 路 softmax (w_g, w_3)，scale_y 改用 C；C 的权重仍读 cnet.pt（训练时用 B 的误差加权）。
  输出 a15_noB_seed<种子>，summary 里直接对照同种子同读数的 A15。
A16（用户 2026-11-03："就把这个新的算法叫做 A 16"）= --third cnet0 --no-b：门控支 A + 不加权的互补支 C（cnet0），训练和推理都不用 B；
  输出 a16_seed<种子>。贡献拆解（用户同日："查门控和融合的贡献"）在同一次运行里多出三列（只 A16 / --no-b 有）：
    fix  固定权重融合：w_g 取一个常数（融合验证集 70 % 训练部分上网格 0, 0.05, …, 1 挑），看"逐样本学权重"比"一个常数"多出多少；
    orc  逐样本最优权重（知道真值才算得出，上限，不是算法）：w_g ∈ {0, 0.1, …, 1} 里挑误差最小的，峰位校正前；
         真值靠包一层 AD.meas 记下刚生成读数对应的谱；作者光束 / 5 束模拟没有经过 AD.meas，orc 退回 A+C 融合，这两处不报 orc。
    计时：门控支（ParamE + 零件网络 + GaussX + fuse_parts）和 C 支每条读数的平均推理时间。
让 A 起作用的两个试法（用户 2026-11-03："两个都跑"；同日改名：C 以后叫 B，原 MLP 叫旧 B），都只配 --no-b，流程同 A16（融合 + fix / orc + 峰位校正）：
  cnetA  B 按 A 的弱点加权：训练库每条谱先算门控支 A 的形状误差 e_A（训练口径读数 ND 次的平均），
         样本权重 1 + min(e_A / 全库均值, 3)，读数仍每步重抽——A 已经做好的谱少学，容量让给 A 做不好的谱。输出 a16_bwA_seed<种子>。
  cnetS  B 看 A：B 的输入 = L1 归一读数 ⊕ A 的峰值归一重建谱（波段内），由 B 自己学哪里信 A。
         门控支太慢不能每步跑，所以训练库每条谱预先抽 ND 次读数、算好 A，B 在这份固定读数池上训练
         （按谱留 3 % 早停）；scale_y 用 cnet0。输出 a16_bsA_seed<种子>。
  两者共用 a15_nets/gatelib.npz（训练库 × ND 次读数的 A 输出和 e_A；有就读）。
A16A（用户 2026-11-03："这个算法叫做 A 十六 A……先不动 B，我们就只改 branch A"）= A16 加 --a16a K：B（cnet0）原样，只改门控支 A 的先验——
  先在融合验证集上按 B 原始输出的误差挑出 B 最弱的 NHARD = 8 个族（只用验证集，不看测试集），
  这些族的单个高斯类换成多高斯：训练库里该族的谱峰值归一后 k-means 分 K 簇，每簇一类高斯（≥ 20 条才建），族总先验仍 GP_EACH、簇间均分；
  原本没有自己类的峰状族（PEAKY）若进了这 8 个，也照样加簇。峰分支（ParamE / 零件网络 / χ² 挑）不动。融合小网、κ、q 照 A16 重拟。
  诊断（11-03，诊断用新种子、Uddin）：8 族上 A 几何平均 0.199 → K = 32 时 0.162，硬选类别不帮忙（0.203），损失主要在先验。
  融合修正（用户 11-03"先修融合"）：融合小网和 q 的输入里有"门控每一类的对数后验"，K 簇后从 22 维变 272 维而融合验证集只有约 4000 条，
  所以默认把同族各簇的后验加回族一级（维数同 A16）；--a16a-fuse raw = 上一版（不合并，输出带 _raw）。
  --a16a-nvf N：融合验证集每族 N 条谱（默认同 A16 的 40；≠ 40 时输出带 _v<N>）。
  --a16a-beta β（用户 2026-11-03："宁可他把别的组给认成这个难的组，我也不想他这个难的组给认错"）：难族各簇的 log 先验加 log β，
  少漏判、多误报（误报的样本留给融合用 B 纠正）；≠ 1 时输出带 _b<β>。
  --a16a-ss（用户 2026-11-03："继续按照这条路进行推进，来优化这个算法"）：门控支输出再接结构解法（a16_ss.py，形状字典 + 贝叶斯证据贪心选形状，
  闭式后验均值）：A′ = A·(1 − α Σ w_h) + α Σ w_h X_SS，w_h = A 给激光线 / 荧光灯 / 4–6 峰 / 阶梯 / 零件组合各簇的后验和（A 判族、数学先验重建）；
  参数都在诊断用调参集上选定，融合小网、κ、q 照常重拟。输出带 _ss。
  --a16a-vscale（用户 2026-11-03"先改融合缩放"）：融合验证集读数先按 C 估的峰值缩放（同测试 decode_ac 的 scale_y）再拟一致化 κ 和融合小网；
  原来验证集不缩放（等于用真幅度），C 在窄线族上把幅度估低（激光线 a ≈ 0.64），测试时门控支失效而融合仍信它。输出带 _vs。
  A17（用户 2026-11-03"最新算法命名A17"）= --a16a 32 --a16a-ss --a16a-vscale，输出到 06_重建算法/A17/results/a17_seed<种子>。
  --a16a-escale（用户 2026-11-03"做证据挑缩放"）：门控支读数逐条按 GaussX 边缘似然在 0.45–1.3 倍里挑缩放（a16_ss.escale），输出带 _es；A17 + --a16a-escale 即 A18（用户 2026-11-03 定名），输出到 06_重建算法/A18/results/a18_seed<种子>。
  --a16a-fams nonpeaky：消融，只给原本有类的非峰状族换多高斯，PEAKY 族不加类。输出 a16a_seed<种子>（K ≠ 32 或消融带后缀）。
定名（用户 2026-11-02：A12t 改名 A15）：--third cnet 这一版是 A15，输出目录 a15_seed<种子>；xie / cnet0 叫 A15·Xie第三支 / A15·无加权，只作对照和消融。
输出：45_Uddin2024_真实R矩阵/results/a15_seed<种子> 或 a15_<third>_seed<种子>/{peakfix.json, log.txt, qnet.pt, metrics/}；
  第三支权重 results/a15_nets/<third>.pt（有就读，没有就训）
"""
import os, sys, json, time, argparse
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument('--third', choices=['xie', 'cnet', 'cnet0', 'cnetA', 'cnetS'], default='cnet')
ap.add_argument('--nd-lib', type=int, default=2, help='cnetA / cnetS：训练库每条谱预算 A 的读数次数')
ap.add_argument('--test-seed', type=int, default=4242); ap.add_argument('--metrics', action='store_true')
ap.add_argument('--train-only', action='store_true', help='只训练第三支网络（cnet / cnet0）就退出')
ap.add_argument('--a16a', type=int, default=0, help='A16A：B 弱族的高斯先验换成 K 簇多高斯（0 = 不用）')
ap.add_argument('--a16a-fams', choices=['all', 'nonpeaky'], default='all', help='A16A 消融：nonpeaky = PEAKY 族不加类')
ap.add_argument('--a16a-fuse', choices=['pool', 'raw'], default='pool', help='A16A：簇后验合回族一级（pool）还是逐簇（raw，上一版）')
ap.add_argument('--a16a-beta', type=float, default=1.0, help='A16A：难族各簇先验乘 β（1 = 不放大）')
ap.add_argument('--a16a-nvf', type=int, default=0, help='A16A：融合验证集每族条数（0 = 同 A16）')
ap.add_argument('--a16a-ss', action='store_true', help='A16A：门控支再接结构解法（a16_ss.py；A 判族、形状字典重建）')
ap.add_argument('--a16a-vscale', action='store_true', help='A16A：融合验证集读数也按 C 峰值缩放（同测试）')
ap.add_argument('--a16a-escale', action='store_true', help='A17：门控支读数按 GaussX 证据逐条挑缩放（a16_ss.escale）')
ap.add_argument('--no-b', action='store_true', help='消融：去掉原网络支 B，只融合门控支 A + 第三支 C（scale_y 也改用 C）')
ARG = ap.parse_args()
sys.argv = [os.path.join(ROOT, 'a11_peakfix.py')]
sys.path.insert(0, ROOT)
import a11_peakfix as PF
import torch, torch.nn as tnn

AD, P18, P17, LD = PF.AD, PF.P18, PF.AD.P17, PF.LD
WL, BM, DEV = AD.WL, AD.BM, PF.DEV; NWB = int(BM.sum())
ALG = {'cnet': 'A15', 'xie': 'A15·Xie第三支', 'cnet0': 'A15·无加权'}.get(ARG.third, '')   # 用户 2026-11-02：A12t 改名 A15（对照、消融版带后缀）
NOB = ARG.no_b                                                         # 用户 2026-11-02："branch A 加 C 不要 B，对比 A 加 B 加 C"
WA = ARG.third in ('cnetA', 'cnetS'); STK = ARG.third == 'cnetS'       # 让 A 起作用的两个试法（文件头）
SMOKE = bool(os.environ.get('A16X_SMOKE'))                              # 试跑：训练库取 1/40、300 步、不存网络，验证集算完就退
assert NOB or not WA, 'cnetA / cnetS 只配 --no-b'
TAG = f'a15_seed{ARG.test_seed}' if ARG.third == 'cnet' else f'a15_{ARG.third}_seed{ARG.test_seed}'
if NOB:
    ALG = ALG + '·无B'; TAG = TAG.replace('_seed', '_noB_seed')
    if ARG.third == 'cnet0':                                            # 用户 2026-11-03：完全不用 B 的这一版叫 A16
        ALG, TAG = 'A16', f'a16_seed{ARG.test_seed}'
    if WA:
        ALG = {'cnetA': 'A16·B按A加权', 'cnetS': 'A16·B看A'}[ARG.third]
        TAG = f'a16_{"bsA" if STK else "bwA"}_seed{ARG.test_seed}'
if ARG.a16a:                                                          # 用户 2026-11-03：只改 A 的这一版叫 A16A
    assert NOB and ARG.third == 'cnet0', 'A16A 只配 --third cnet0 --no-b'
    sfx = (('' if ARG.a16a == 32 else f'_K{ARG.a16a}') + ('' if ARG.a16a_fams == 'all' else '_np') + ('' if ARG.a16a_fuse == 'pool' else '_raw')
           + (f'_v{ARG.a16a_nvf}' if ARG.a16a_nvf and ARG.a16a_nvf != 40 else '') + (f'_b{ARG.a16a_beta:g}' if ARG.a16a_beta != 1 else '') + ('_ss' if ARG.a16a_ss else '') + ('_vs' if ARG.a16a_vscale else '') + ('_es' if ARG.a16a_escale else ''))
    ALG, TAG = 'A16A' + sfx.replace('_', '·'), f'a16a{sfx}_seed{ARG.test_seed}'
A17 = ARG.a16a == 32 and sfx.startswith('_ss_vs') if ARG.a16a else False  # 用户 2026-11-03："最新算法命名A17"（= A16A + SS + 融合验证集按 C 缩放），结果放 06_重建算法/A17
A18 = A17 and sfx == '_ss_vs_es'                                       # 用户 2026-11-03："把 A 十七 ES 当做 A 十八算法"，结果放 06_重建算法/A18
if A18:
    ALG, TAG = 'A18', f'a18_seed{ARG.test_seed}'
elif A17:
    ALG, TAG = 'A17' + sfx[6:].replace('_', '·'), f'a17{sfx[6:]}_seed{ARG.test_seed}'
PF.OUT = (os.path.join(os.path.dirname(ROOT), 'A18' if A18 else 'A17', 'results', TAG) if A17 else os.path.join(AD.DEVDIR, 'results', TAG)); os.makedirs(PF.OUT, exist_ok=True)
NETDIR = os.path.join(AD.DEVDIR, 'results', 'a15_nets'); os.makedirs(NETDIR, exist_ok=True)
AD.TEST_ST = ARG.test_seed
LAB3 = {'xie': 'Xie 原样', 'cnet': '互补支', 'cnet0': '互补支·无加权', 'cnetA': 'B·按A加权', 'cnetS': 'B·看A'}[ARG.third]
log = PF.log
TM = dict(gate=0.0, t3=0.0, n=0)                                       # 推理计时（秒、读数条数）
_TR = {}
_meas0 = AD.meas


def _meas(S, *a, **k):
    """记下读数对应的真谱 (NW, n)，只给 orc（逐样本最优权重上限）用。"""
    Y = _meas0(S, *a, **k); _TR['Y'], _TR['S'] = Y, S
    return Y


AD.meas = _meas


class XieMLP(tnn.Module):                                              # = all_algos_device.XieMLP
    def __init__(s, n_, no, w=512):
        super().__init__(); s.i = tnn.Linear(n_, w); s.o = tnn.Linear(w, no)
        s.B = tnn.ModuleList([tnn.Sequential(tnn.Linear(w, w), tnn.ReLU(), tnn.Linear(w, w)) for _ in range(3)])

    def forward(s, y):
        z = torch.relu(s.i(y))
        for blk in s.B:
            z = torch.relu(z + blk(z))
        return s.o(z)


class CNet(tnn.Module):
    """互补网络支：残差 MLP，块 = pre-LayerNorm + GELU。"""
    def __init__(s, n_, no, w=512, nb=4):
        super().__init__(); s.i = tnn.Linear(n_, w); s.n = tnn.LayerNorm(w); s.o = tnn.Linear(w, no)
        s.B = tnn.ModuleList([tnn.Sequential(tnn.LayerNorm(w), tnn.Linear(w, w), tnn.GELU(), tnn.Linear(w, w)) for _ in range(nb)])

    def forward(s, y):
        z = s.i(y)
        for blk in s.B:
            z = z + blk(z)
        return s.o(torch.nn.functional.gelu(s.n(z)))


l1n = lambda y: y / (y.abs().sum(1, keepdim=True) + 1e-30)


nrm_t = lambda x: x / (x.amax(1, keepdim=True) + 1e-6)


def train_cnet(Rn, Slib, NN, comp, steps=30000, bs=2048, seed=11, lib=None, libx=None):
    """互补网络支训练。comp = 是否互补加权（原网络 NN 冻结，算同一读数上的误差当权重）。
    lib（cnetA / cnetS）= gatelib：cnetA 用其中的 e_A 作每条谱的固定权重；cnetS 在其中的固定读数池上训练，输入多 A 的重建谱。"""
    t0 = time.time(); torch.manual_seed(seed)
    S = torch.tensor(Slib, dtype=torch.float32, device=DEV); bt = torch.tensor(BM, device=DEV); N = len(S)
    sb = S[:, bt]; r = float((sb ** 2).mean() / (sb.diff(dim=1) ** 2).mean())
    net = CNet(AD.N_RD + (NWB if STK else 0), NWB).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
    sp = torch.nn.functional.softplus; nrm = nrm_t
    wA = None if lib is None else torch.tensor(1 + np.minimum(lib['eA'] / lib['eA'].mean(), 3), dtype=torch.float32, device=DEV)
    if STK:     # 读数池 = 训练库 × ND 次 + 扩充池（libx）× 1 次，按谱留 3 % 早停；每批一半取池（A 输入加扰动、10 % 置零），一半在线新读数（A 输入置零）
        f2 = lambda a: a.transpose(0, 2, 1).reshape(a.shape[0] * a.shape[2], -1)
        ps = [lib] + ([libx] if libx is not None else []); off = np.cumsum([0, N] + [p['G'].shape[2] for p in ps[1:]])
        Yp = torch.tensor(np.concatenate([f2(p['Y']) for p in ps]), dtype=torch.float32, device=DEV)
        Gp = nrm_t(torch.tensor(np.concatenate([f2(p['G']) for p in ps]), dtype=torch.float32, device=DEV).clamp(min=0))
        sid = torch.tensor(np.concatenate([o + np.arange(p['G'].shape[0] * p['G'].shape[2]) % p['G'].shape[2] for o, p in zip(off, ps)]), device=DEV)
        SA = torch.cat([S] + [torch.tensor(p['S'], dtype=torch.float32, device=DEV) for p in ps[1:]])
        hs = np.random.default_rng(seed).random(len(SA)) < 0.03; hold = torch.tensor(hs, device=DEV)[sid]
        itr, iva = torch.nonzero(~hold).ravel(), torch.nonzero(hold).ravel(); best = (np.inf, None)
        otr = torch.nonzero(~torch.tensor(hs[:N], device=DEV)).ravel(); hb = bs // 2
        lin = torch.linspace(-1, 1, NWB, device=DEV)
    what = {'cnetA': 'B·按A加权（权重 1 + min(e_A/均值, 3)）',
            'cnetS': f'B·看A（读数池 {len(SA) if STK else 0} 谱 / {len(Yp) if STK else 0} 条读数，半批在线新读数，A 输入加扰动，按谱留 3 % 早停）'}.get(
        ARG.third, 'cnet（互补加权）' if comp else 'cnet0（无互补加权）')
    log(f'  训练 {what}：训练库 {N} 条，{steps} 步 × {bs}；导数项系数 0.5·r，r = {r:.1f}')

    def err(xn, st):
        return ((xn - st) ** 2).mean(1) + 0.5 * r * ((xn.diff(dim=1) - st.diff(dim=1)) ** 2).mean(1)
    for it in range(steps):
        if STK:
            if it % 50 == 0:
                gb = np.random.default_rng(seed * 100000 + it)
            j = itr[torch.randint(0, len(itr), (bs - hb,), device=DEV)]; nj = len(j)
            ga = Gp[j] * (1 + 0.05 * torch.randn(nj, 1, device=DEV) * lin) + 0.02 * torch.randn(nj, NWB, device=DEV)
            ga = nrm(ga.clamp(min=0)) * (torch.rand(nj, 1, device=DEV) > 0.1)
            so = SA[otr[torch.randint(0, len(otr), (hb,), device=DEV)]]; y = AD.H.meas_t(torch, Rn, so, gb, DEV)
            xin = torch.cat([torch.cat([l1n(Yp[j]), ga], 1), torch.cat([l1n(y), torch.zeros(hb, NWB, device=DEV)], 1)])
            st = torch.cat([SA[sid[j]][:, bt], so[:, bt]]); xn = nrm(sp(net(xin)))
        else:
            if it % 50 == 0:
                gb = np.random.default_rng(seed * 100000 + it)
            ii = torch.randint(0, N, (bs,), device=DEV); s = S[ii]; st = s[:, bt]
            y = AD.H.meas_t(torch, Rn, s, gb, DEV)
            xn = nrm(sp(net(l1n(y))))
        e = err(xn, st)
        if comp:
            with torch.no_grad():
                eo = ((nrm(sp(NN['net'](y / (y.norm(dim=1, keepdim=True) + 1e-30)))) - st) ** 2).mean(1)
                wg = 1 + (eo / eo.mean()).clamp(max=3)
            loss = (wg * e).sum() / wg.sum()
        elif ARG.third == 'cnetA':
            wg = wA[ii]; loss = (wg * e).sum() / wg.sum()
        else:
            loss = e.mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if STK and (it % 1000 == 0 or it == steps - 1):
            with torch.no_grad():
                lv = float(torch.cat([err(nrm(sp(net(torch.cat([l1n(Yp[k]), Gp[k]], 1)))), SA[sid[k]][:, bt]) for k in iva.split(8192)]).mean())
            if lv < best[0]:
                best = (lv, it, {k: v.clone() for k, v in net.state_dict().items()})
        if it % 5000 == 0 or it == steps - 1:
            log(f'    step {it} loss {loss.item():.5f}' + (f'  留出 {lv:.5f}' if STK else '') + f'  {time.time() - t0:.0f} s')
    if STK:
        net.load_state_dict(best[2]); log(f'    早停：取 step {best[1]}（留出 {best[0]:.5f}）')
    return net.eval()


XN = None; XN0 = None                                                # XN0 = cnet0（cnetA / cnetS 预算 A 时、cnetS 的 scale_y 用）


def load_third(Rn, Slib, NN):
    global XN, XN0
    if ARG.third == 'xie':
        XN = XieMLP(AD.N_RD, NWB).to(DEV); XN.load_state_dict(torch.load(os.path.join(AD.DEVDIR, 'results', 'all_algos', 'xie_samelib.pt')))
    else:
        f = os.path.join(NETDIR, f'{ARG.third}.pt')
        if os.path.exists(f):
            XN = CNet(AD.N_RD + (NWB if STK else 0), NWB).to(DEV); XN.load_state_dict(torch.load(f)); log(f'  第三支：读入 {os.path.relpath(f, AD.DEVDIR)}')
        elif WA:
            XN = None                                                  # 要先有门控支，build3 里再训
        else:
            XN = train_cnet(Rn, Slib, NN, comp=ARG.third == 'cnet'); torch.save(XN.state_dict(), f)
    if WA:
        f0 = os.path.join(NETDIR, 'cnet0.pt'); assert os.path.exists(f0), '先跑 A16（cnet0）'
        XN0 = CNet(AD.N_RD, NWB).to(DEV); XN0.load_state_dict(torch.load(f0)); XN0.eval()
    if XN is not None:
        XN.eval()


def t3_infer(Rn, Y, G=None, net=None):
    """Y (nd, n, M) → (nd, NW, M)：第三支（L1 归一输入）出形状，幅度按读数最小二乘（同 algo17.infer_mlp）。
    cnetS 时 G = 门控支输出 (nd, NW, M)，波段内峰值归一后接在读数后面；net 给定时用它（scale_y 用 cnet0）。"""
    nd, n, M = Y.shape; Yr = Y.transpose(0, 2, 1).reshape(nd * M, n); net = net or XN
    with torch.no_grad():
        y = l1n(torch.tensor(Yr, dtype=torch.float32, device=DEV))
        if net is XN and STK:
            g = torch.tensor(G[:, BM, :].transpose(0, 2, 1).reshape(nd * M, -1), dtype=torch.float32, device=DEV).clamp(min=0)
            y = torch.cat([y, nrm_t(g)], 1)
        x = torch.nn.functional.softplus(net(y)).double().cpu().numpy()
    X = np.zeros((nd * M, Rn.shape[1])); X[:, BM] = x; p = X @ Rn.T
    a = np.maximum((p * Yr).sum(1) / np.maximum((p ** 2).sum(1), 1e-30), 0)
    return (X * a[:, None]).reshape(nd, M, -1).transpose(0, 2, 1)


def feats_x(Rn, Y, Xc, Xg, Xx, GM):
    """第三支的 6 个附加特征 (nd·M, 6)。"""
    cc = np.log(P18.T.chi2_all(Rn, Y, Xc, GM).reshape(-1) + 1e-3); cx = np.log(P18.T.chi2_all(Rn, Y, Xx, GM).reshape(-1) + 1e-3)
    nz = lambda X: (lambda f: f / np.maximum(f.max(1, keepdims=True), 1e-30))(P18.flat(X))
    nc, ng, nx = nz(Xc), nz(Xg), nz(Xx)
    d = lambda a, b: np.log(np.sqrt(((a - b) ** 2).mean(1)) + 1e-4)
    rough = np.log(np.abs(np.diff(nx, 2, axis=1)).mean(1) + 1e-5)
    dp = np.minimum(np.abs(PF.peaks(P18.flat(Xx)) - PF.peaks(P18.flat(Xc))) / 20.0, 1.0)
    return np.stack([cx, cx - cc, d(nx, nc), d(nx, ng), rough, dp], 1)


def fit_blend3(Fv, Xs, St, b, steps=3000, seed=5):
    """多支 softmax 权重的小网；Xs = [Xg, Xc, X3]（--no-b 时 [Xg, X3]），各 (N, NW)。训练方式同 algo18.fit_blend_net。"""
    t = torch
    t.manual_seed(seed); rng = np.random.default_rng(seed); N = len(Fv); idx = rng.permutation(N); tr, va = idx[:int(0.7 * N)], idx[int(0.7 * N):]
    mu, sd = Fv.mean(0), Fv.std(0) + 1e-9
    f = t.tensor((Fv - mu) / sd); xs = t.stack([t.tensor(X[:, b]) for X in Xs], 1); s = t.tensor(St[:, b])     # xs (N, 3, NWB)
    sn = s / s.amax(1, keepdim=True); rg = sn.amax(1) - sn.amin(1)
    net = t.nn.Sequential(t.nn.Linear(f.shape[1], 32), t.nn.Tanh(), t.nn.Linear(32, len(Xs))).double()
    opt = t.optim.AdamW(net.parameters(), 3e-3, weight_decay=1e-2)

    def loss(i):
        w = t.softmax(net(f[i]), 1); x = (w[:, :, None] * xs[i]).sum(1)
        xn = x / (x.amax(1, keepdim=True) + 1e-30)
        return (t.sqrt(((xn - sn[i]) ** 2).mean(1)) / rg[i]).mean()
    best = (np.inf, None)
    for it in range(steps):
        l = loss(tr); opt.zero_grad(); l.backward(); opt.step()
        if it % 50 == 0:
            with t.no_grad():
                lv = float(loss(va))
            if lv < best[0]:
                best = (lv, {k: v.clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1])
    return mu, sd, net, va


def blend3_w(Fx, mu, sd, net):
    with torch.no_grad():
        return torch.softmax(net(torch.tensor((Fx - mu) / sd)), 1).numpy()


def mix3(W, Xg, Xc, X3):
    nd = Xg.shape[0]
    return P18.unflat(W[:, 0:1] * P18.flat(Xg) + W[:, 1:2] * P18.flat(Xc) + W[:, 2:3] * P18.flat(X3), nd)


def gate_lib(Rn, Slib, gate_of, chunk=2000, name='gatelib', nd=None, seed=20261103):
    """训练库 × ND 次训练口径读数：scale_y（cnet0）后的读数 Y (ND, n, N)、门控支输出 G (ND, NWB, N)、A 的形状误差 e_A (N,)。
    存 NETDIR/<name>.npz，有就读（和测试种子无关，几个种子共用）。cnetS 的扩充池用 name='gatelibX'、nd=1。"""
    nd = nd or ARG.nd_lib
    f = os.path.join(NETDIR, f'{name}_smoke.npz' if SMOKE else f'{name}.npz')
    if os.path.exists(f):
        z = np.load(f)
        if z['G'].shape[0] == nd and z['G'].shape[2] == len(Slib):
            log(f'  门控支训练库输出：读入 {os.path.relpath(f, AD.DEVDIR)}'); return dict(z)
    t0 = time.time(); g = np.random.default_rng(seed); Ys, Gs = [], []
    for i in range(0, len(Slib), chunk):
        S = Slib[i:i + chunk]; Y = _meas0(S.T, g, 'train', nd=nd)
        a = t3_infer(Rn, Y, net=XN0)[:, BM, :].max(1); Y = Y / np.maximum(a, 1e-30)[:, None, :]
        Ys.append(Y); Gs.append(gate_of(Y)[0][:, BM, :].astype(np.float32))
        if i == 0 or (i // chunk) % 5 == 4:
            log(f'    门控支跑训练库 {i + len(S)}/{len(Slib)}  {time.time() - t0:.0f} s')
    Y, G = np.concatenate(Ys, 2), np.concatenate(Gs, 2)
    Gn = G / np.maximum(G.max(1, keepdims=True), 1e-30); St = Slib[:, BM] / Slib[:, BM].max(1, keepdims=True)
    eA = np.sqrt(((Gn - St.T[None]) ** 2).mean(1)).mean(0)              # 峰值归一均方根，ND 次平均
    np.savez(f, Y=Y, G=G, eA=eA)
    log(f'  门控支跑完{name} {len(Slib)} 条 × {nd} 次：{time.time() - t0:.0f} s；e_A 均值 {eA.mean():.4f}、中位 {np.median(eA):.4f}、最大 {eA.max():.4f}')
    return dict(Y=Y, G=G, eA=eA)


NHARD = 8


def gf_a16a(Rn, lib, famg, vsets, Yv, E):
    """A16A 的门控先验：融合验证集上 B（cnet0 原始输出）误差最大的 NHARD 个族，单高斯换成 K 簇多高斯（文件头）。"""
    from sklearn.cluster import KMeans
    eB = {k: float(E(np.clip(t3_infer(Rn, Yv[k]), 0, None), vsets[k]).mean()) for k in vsets if k in lib and k != '同分布五类'}
    cand = [k for k in eB if ARG.a16a_fams == 'all' or k in famg]
    hard = sorted(cand, key=eB.get, reverse=True)[:NHARD]
    log('A16A：B 在融合验证集上最弱的 ' + str(NHARD) + ' 族（换多高斯）：' + '，'.join(f'{k} {eB[k]:.3f}' for k in hard))
    ex, pw = {}, {}
    for k in famg:
        if k not in hard:
            ex[k] = lib[k]; pw[k] = P17.GP_EACH
    for k in hard:
        S = lib[k]; Sn = S[:, BM] / np.maximum(S[:, BM].max(1, keepdims=True), 1e-30)
        lab = KMeans(ARG.a16a, n_init=3, random_state=0).fit_predict(Sn); ok = [j for j in range(ARG.a16a) if (lab == j).sum() >= 20]
        for j in ok:
            ex[f'{k}#{j}'] = S[lab == j]; pw[f'{k}#{j}'] = P17.GP_EACH / len(ok)
    GF = AD.Q.GaussX(Rn, AD.BAND, ex, pw)
    log(f'   门控类别 {len(GF.classes)} 个（A16 是 {5 + len(famg)} 个）')
    if ARG.a16a_beta != 1:                                             # 难族先验放大（文件头）；直接加在 log 先验上，五类的 1 − Σ 不会变负
        GF.logpi[[i for i, c in enumerate(GF.classes) if '#' in c]] += float(np.log(ARG.a16a_beta)); log(f'   难族各簇先验 × β = {ARG.a16a_beta:g}')
    if ARG.a16a_fuse == 'pool':                                        # 融合 / q 的类后验特征：同族各簇加回一类（文件头）
        grp = list(dict.fromkeys(c.split('#')[0] for c in GF.classes)); Pm = np.zeros((len(GF.classes), len(grp)))
        for i, c in enumerate(GF.classes):
            Pm[i, grp.index(c.split('#')[0])] = 1
        _POOL['P'] = Pm; log(f'   融合 / q 的类后验特征合回 {len(grp)} 维')
    return GF


_POOL = {}
_feats_pf = P18.feats                                                  # a11_peakfix 已包过一层（存 PF._F['last'] 给 q）


def _feats_pool(*a):
    f8, lw = _feats_pf(*a)
    if 'P' in _POOL:
        lw = np.log(np.exp(lw) @ _POOL['P'] + 1e-6); PF._F['last'] = (f8, lw)
    return f8, lw


P18.feats = _feats_pool


def prep():
    """训练库、零件网络、原网络（= a11_device.build 前半段，网络读 results/a11/ 里已训练的）。"""
    Rn = AD.RN; D, H = AD.D, AD.H
    P = H.Parts(Rn, AD.BAND)
    lib = AD.all_fams(P17.TRSEED, AD.NTR, P, True)
    Slib = np.concatenate([lib[k] for k in lib] + [D.gen5(len(lib['同分布五类']) * 3, P17.TRSEED + 7, AD.BAND)[0]])
    f8n, fmn = os.path.join(AD.OUT, 'net8.pt'), os.path.join(AD.OUT, 'mlp.pt')
    assert os.path.exists(f8n) and os.path.exists(fmn), '先跑 a11_device.py uddin 训练 net8.pt / mlp.pt'
    NT8 = H.train_net(P, Rn, steps=1); NT8['net'].load_state_dict(torch.load(f8n)); c = torch.load(f8n + '.musd'); NT8['mu'], NT8['sd'] = c['mu'], c['sd']
    NN = P17.train_mlp(Rn, Slib[:10], AD.BAND, steps=1); NN['net'].load_state_dict(torch.load(fmn)); NN['net'].eval()
    load_third(Rn, Slib, NN)
    return P, lib, Slib, NT8, NN


def build3():
    """= a11_device.build，融合一步换成三支。"""
    global XN
    t0 = time.time(); Rn = AD.RN; BAND = AD.BAND; D, H, A, S7, F, Q, DS = AD.D, AD.H, AD.A, AD.S7, AD.F, AD.Q, AD.DS
    P, lib, Slib, NT8, NN = prep()
    sv, _, se = P18.SEEDS[0]; g = np.random.default_rng(se)
    Sva = D.gen5(60, sv, BAND)[0]; Yva = AD.meas(Sva.T, g)
    vsets = AD.all_fams(P18.VSEED, ARG.a16a_nvf or AD.NVF, P, True); Yv = {k: AD.meas(S.T, g) for k, S in vsets.items()}
    nd = LD.NDRAW; E = lambda X, S: np.array([LD.band_nrmse(X[r], S.T) for r in range(nd)])
    lam = min(LD.NN_LAMS, key=lambda l: E(A.dec_snn(Rn, Yva, l), Sva).mean())
    GM0 = D.Gauss5(Rn, BAND); sig = min(A.SIG_E, key=lambda s: E(GM0.solve(Yva, s)[0], Sva).mean())
    GM = S7.GaussMR(Rn, BAND, F.C_TRUE); PM = D.ParamE(Rn, BAND)
    famg = [k for k in lib if k not in P17.PEAKY and k != '同分布五类']
    if ARG.a16a:
        GF = gf_a16a(Rn, lib, famg, vsets, Yv, E)
    else:
        GF = Q.GaussX(Rn, BAND, {k: lib[k] for k in famg}, {k: P17.GP_EACH for k in famg})
    ipk = [GF.classes.index(c) for c in D.PARAM_CLS]
    GSS = None
    if ARG.a16a_ss:
        import a16_ss; GSS = a16_ss.GateSS(Rn, WL, BAND, GM, GF.classes, dev=DEV)
        log(f'A16A·SS：结构解法并入门控支，族 → 类型 {[(f, t_) for f, (_, t_) in GSS.idx.items()]}，α = {a16_ss.ALPHA}')
    log(f'λ（A1）= {lam:g}；σ_e = {sig}；训练库 {len(lib)} 族、{len(Slib)} 条；第三支 = {LAB3}；网络就绪 {time.time() - t0:.0f} s')

    def gate_of(Y):
        c = None
        if ARG.a16a_escale:                                            # 证据挑缩放：门控支在 c·Y 上算，输出再除回 c（同 C 的幅度）
            import a16_ss; c = a16_ss.escale(GF, Y)[:, None, :]; Y = Y * c
        Xp = PM.solve(Y, sig)[0]; X8 = H.solve_pick(P, Y, H.infer(NT8, Y))[0]
        Xf, dPf, Wf = GF.solve_u(Y); gp = DS.fuse_parts(Rn, Xf, Wf, dPf, [Xp, X8], Y, GF, GF.classes)
        g = np.clip(DS.gate_mean(gp), 0, None)
        if GSS is not None:                                            # --a16a-ss：A 判族、结构解法重建（a16_ss.py）
            g = GSS.apply(Y, g, Wf, np.clip(t3_infer(Rn, Y), 0, None))
        if c is not None:
            g = g / c
        return g, Wf

    Sx = None
    if STK:                                                            # cnetS 扩充池：另 4 个种子各生成一份训练库（每条谱 1 次读数），压固定池过拟合
        Sx = np.concatenate([np.concatenate(list(AD.all_fams(P17.TRSEED + 1000 * k, AD.NTR, P, True).values())
                                            + [D.gen5(len(lib['同分布五类']) * 3, P17.TRSEED + 1000 * k + 7, BAND)[0]]) for k in range(1, 5)])
    if WA and XN is None:                                              # cnetA / cnetS：训练库上先算 A，再训 B
        Sl = Slib[::40] if SMOKE else Slib; Sxl = None if Sx is None else (Sx[::160] if SMOKE else Sx)
        lx = None if Sx is None else dict(gate_lib(Rn, Sxl, gate_of, name='gatelibX', nd=1, seed=20261104), S=Sxl)
        XN = train_cnet(Rn, Sl, NN, comp=False, steps=300 if SMOKE else 30000, lib=gate_lib(Rn, Sl, gate_of), libx=lx); XN.eval()
        if not SMOKE:
            torch.save(XN.state_dict(), os.path.join(NETDIR, f'{ARG.third}.pt'))

    def base(Y):
        t_ = time.time()
        gt, Wf = gate_of(Y)
        o = dict(gate=gt, Wf=Wf); t1 = time.time()
        o['t3'] = np.clip(t3_infer(Rn, Y, gt), 0, None)
        TM['gate'] += t1 - t_; TM['t3'] += time.time() - t1; TM['n'] += Y.shape[0] * Y.shape[2]
        if not NOB:
            o['net'] = np.clip(P17.infer_mlp(NN, Rn, Y), 0, None)
        return o

    if ARG.a16a_vscale:                                               # 验证集读数也按 C 峰值缩放（文件头），和 decode_ac 一致
        Yv = {k: Y / np.maximum(t3_infer(Rn, Y, net=XN0 if STK else XN)[:, BM, :].max(1), 1e-30)[:, None, :] for k, Y in Yv.items()}
        log('A16A：融合验证集读数按 C 峰值缩放（同测试）')
    vo = {k: base(Y) for k, Y in Yv.items()}
    if SMOKE:
        log('试跑：融合验证集 B 原始 平均形状误差 ' + '，'.join(f'{k} {E(o["t3"], vsets[k]).mean():.4f}' for k, o in list(vo.items())[:4])); sys.exit()
    Sv = {k: np.tile(S, (nd, 1)) for k, S in vsets.items()}
    St_all = np.concatenate([Sv[k] for k in vsets]); Yf_all = np.concatenate([P18.flat(Yv[k]) for k in vsets])
    CS, kap = {}, {}
    for br in (('t3',) if NOB else ('net', 't3')):                    # 两个网络各自一致化：各自的误差协方差、各自挑 κ
        Xn_all = np.concatenate([P18.flat(vo[k][br]) for k in vsets])
        CS[br] = P18.Consist(Rn, BM, GM).fit_cov(Xn_all, St_all)
        ev = {kp: float(LD.band_nrmse(CS[br].apply(Xn_all, Yf_all, kp).T, St_all.T).mean()) for kp in P18.KAPPAS}; kap[br] = min(ev, key=ev.get)
        log(f'   一致化 {br}：κ = {kap[br]}（验证集形状误差 κ=0 {ev[0.0]:.4f} → {ev[kap[br]]:.4f}）')
    def used_():
        gp = np.random.default_rng(11); pool = np.concatenate([P.gen(1024 * 8, gp)[0] for _ in range(8)])
        return {'调参验证集': Sva, '融合验证集': np.concatenate(list(vsets.values())), '训练库': Slib, '零件网络训练池': pool,
                '高斯先验用谱': D.gen5(A.NTRAIN, 11, BAND)[0], **({} if Sx is None else {'B看A扩充池': Sx})}

    if NOB:                                                            # 消融 A + C：C 一致化后顶替 B 的位置（8 个特征照算，第三支附加特征全都涉及 B，不用）
        G, Xg, X3, fam = [], [], [], []
        for j, k in enumerate(vsets):
            o = vo[k]; o['t3c'] = P18.unflat(CS['t3'].apply(P18.flat(o['t3']), P18.flat(Yv[k]), kap['t3']), nd)
            f8, lw = P18.feats(Rn, Yv[k], o['t3c'], o['gate'], o['Wf'], GM, ipk)
            G.append(np.c_[f8, lw]); Xg.append(P18.flat(o['gate'])); X3.append(P18.flat(o['t3c'])); fam.append(np.full(len(Xg[-1]), j))
        G, Xg, X3, fam = (np.concatenate(v) for v in (G, Xg, X3, fam))
        mu2, sd2, net2, va = fit_blend3(G, [Xg, X3], St_all, BM)
        W = blend3_w(G, mu2, sd2, net2); Xb = W[:, 0:1] * Xg + W[:, 1:2] * X3
        ev = lambda X: np.array([AD.nrmse1(X[i], St_all[i]) for i in va])
        e = {nm: ev(X) for nm, X in (('门控', Xg), (f'{LAB3}一致化', X3), ('A+C 融合', Xb))}
        log('融合验证集留出 30 %（早停用过它，偏乐观）平均形状误差：' + '，'.join(f'{k} {v.mean():.4f}' for k, v in e.items()))
        log('   各族（留出部分）            门控  第三支  A+C融合    w_g   w_3')
        for j, k in enumerate(vsets):
            m = fam[va] == j
            log(f'   {k.replace("留出·", ""):<20s}' + ''.join(f'{e[n][m].mean():8.4f}' for n in e) + ''.join(f'{W[va][m][:, i].mean():6.2f}' for i in range(2)))
        trn = np.setdiff1d(np.arange(len(G)), va); WG = np.linspace(0, 1, 21)
        ef = [float(LD.band_nrmse((w * Xg[trn] + (1 - w) * X3[trn]).T, St_all[trn].T).mean()) for w in WG]; wfix = float(WG[int(np.argmin(ef))])
        log(f'κ（第三支 {kap["t3"]}）；A+C 融合小网训练完；固定权重 w_g = {wfix:.2f}（训练部分误差 {min(ef):.4f}，w_g = 0 时 {ef[0]:.4f}）；建模共 {time.time() - t0:.0f} s')
        WO = np.linspace(0, 1, 11)

        def decode_ac(Y):
            Y0 = Y; a = t3_infer(Rn, Y, net=XN0 if STK else XN)[:, BM, :].max(1); Y = Y / np.maximum(a, 1e-30)[:, None, :]      # scale_y 改用 C（cnetS 用 cnet0）
            o = base(Y); nd_ = Y.shape[0]
            o['t3c'] = P18.unflat(CS['t3'].apply(P18.flat(o['t3']), P18.flat(Y), kap['t3']), nd_)
            f8, lw = P18.feats(Rn, Y, o['t3c'], o['gate'], o['Wf'], GM, ipk)                     # 同时存进 PF._F['last'] 给 q
            W = blend3_w(np.c_[f8, lw], mu2, sd2, net2)
            o['A11'] = P18.unflat(W[:, 0:1] * P18.flat(o['gate']) + W[:, 1:2] * P18.flat(o['t3c']), nd_)
            o['fix'] = wfix * o['gate'] + (1 - wfix) * o['t3c']
            o['orc'] = o['A11'].copy()
            if _TR.get('Y') is Y0:                                       # 有真值：逐样本在 w_g 网格里挑最小误差（上限）
                S = _TR['S']; C = [w * o['gate'] + (1 - w) * o['t3c'] for w in WO]
                Ev = np.array([[LD.band_nrmse(X[r], S) for r in range(nd_)] for X in C]); b = Ev.argmin(0)      # (nd, n)
                o['orc'] = np.stack([np.stack([C[b[r, i]][r, :, i] for i in range(S.shape[1])], 1) for r in range(nd_)])
            o.pop('Wf'); o['w'] = (1 - W[:, 0]).reshape(nd_, -1)
            return o
        return decode_ac, P, used_()
    G2, G3, Xg, Xc, X3, fam = [], [], [], [], [], []
    for j, k in enumerate(vsets):
        o = vo[k]; o['cons'] = P18.unflat(CS['net'].apply(P18.flat(o['net']), P18.flat(Yv[k]), kap['net']), nd)
        o['t3c'] = P18.unflat(CS['t3'].apply(P18.flat(o['t3']), P18.flat(Yv[k]), kap['t3']), nd)
        f8, lw = P18.feats(Rn, Yv[k], o['cons'], o['gate'], o['Wf'], GM, ipk); fx = feats_x(Rn, Yv[k], o['cons'], o['gate'], o['t3c'], GM)
        G2.append(np.c_[f8, lw]); G3.append(np.c_[f8, fx, lw])
        Xg.append(P18.flat(o['gate'])); Xc.append(P18.flat(o['cons'])); X3.append(P18.flat(o['t3c'])); fam.append(np.full(len(Xg[-1]), j))
    G2, G3, Xg, Xc, X3, fam = (np.concatenate(v) for v in (G2, G3, Xg, Xc, X3, fam))
    BN = P18.fit_blend_net(G2, Xc, Xg, St_all, BM)
    mu3, sd3, net3, va = fit_blend3(G3, [Xg, Xc, X3], St_all, BM)
    # 验证集留出 30 %（两种小网都按它早停，所以只是参考）：两支融合 vs 三支融合 vs 各单支
    w2 = P18.blend_w_net(G2, *BN); W = blend3_w(G3, mu3, sd3, net3)
    Xb2 = w2[:, None] * Xc + (1 - w2[:, None]) * Xg; Xb3 = W[:, 0:1] * Xg + W[:, 1:2] * Xc + W[:, 2:3] * X3
    ev = lambda X: np.array([AD.nrmse1(X[i], St_all[i]) for i in va])
    e = {nm: ev(X) for nm, X in (('门控', Xg), ('原网络一致化', Xc), (f'{LAB3}一致化', X3), ('两支融合', Xb2), ('三支融合', Xb3))}
    log('融合验证集留出 30 %（早停用过它，偏乐观）平均形状误差：' + '，'.join(f'{k} {v.mean():.4f}' for k, v in e.items()))
    log('   各族（留出部分）            门控  原网络  第三支  两支融合  三支融合    w_g   w_c   w_3')
    for j, k in enumerate(vsets):
        m = fam[va] == j
        log(f'   {k.replace("留出·", ""):<20s}' + ''.join(f'{e[n][m].mean():8.4f}' for n in e) + ''.join(f'{W[va][m][:, i].mean():6.2f}' for i in range(3)))
    log(f'κ（原网络 {kap["net"]}、第三支 {kap["t3"]}）；两支、三支融合小网训练完；建模共 {time.time() - t0:.0f} s')

    def scale_y(Y):
        a = P17.infer_mlp(NN, Rn, Y)[:, BM, :].max(1)
        return Y / np.maximum(a, 1e-30)[:, None, :]

    def decode(Y):
        Y = scale_y(Y); o = base(Y); nd_ = Y.shape[0]
        o['cons'] = P18.unflat(CS['net'].apply(P18.flat(o['net']), P18.flat(Y), kap['net']), nd_)
        o['t3c'] = P18.unflat(CS['t3'].apply(P18.flat(o['t3']), P18.flat(Y), kap['t3']), nd_)
        f8, lw = P18.feats(Rn, Y, o['cons'], o['gate'], o['Wf'], GM, ipk); fx = feats_x(Rn, Y, o['cons'], o['gate'], o['t3c'], GM)
        PF._F['last'] = (np.c_[f8, fx], lw)                            # 峰位校正 q 的特征也带上第三支特征
        w2 = P18.blend_w_net(np.c_[f8, lw], *BN); W = blend3_w(np.c_[f8, fx, lw], mu3, sd3, net3)
        o['A2'] = P18.unflat(w2[:, None] * P18.flat(o['cons']) + (1 - w2[:, None]) * P18.flat(o['gate']), nd_)
        o['A11'] = mix3(W, o['gate'], o['cons'], o['t3c'])
        o.pop('Wf'); o['w'] = (1 - W[:, 0]).reshape(nd_, -1)
        return o

    return decode, P, used_()


AD_build0 = AD.build                                                # 旧 A12 的 build（a15_examples.py 重放对照用）
AD.build = build3
PF.METH[:] = ['gate', 't3', 't3c', 'fix', 'A11', 'A11p', 'orc'] if NOB else ['gate', 'cons', 't3', 't3c', 'A2', 'A11', 'A11p']
PF.NAME.update(fix='固定权重', orc='逐样本最优', cons='原网络支', t3=f'{LAB3}原始', t3c=f'{LAB3}一致化', A2='两支融合', A11='A+C融合' if NOB else '三支融合', A11p=ALG)


class _JS:
    """旧结果文件只在同一份读数时才拿来对照；否则当空的读。"""
    def __getattr__(s, k):
        return getattr(json, k)

    def load(s, f):
        nm = os.path.basename(getattr(f, 'name', ''))
        d = json.load(f)
        if nm == 'a11.json':
            return dict(fam={}, sim={})
        if nm == 'all_algos.json' and ARG.test_seed != 4242:
            return dict(fam={}, fampk={}, sim=d.get('sim', {}))
        return d


PF.json = _JS()


def summary():
    """A15 和 旧 A12 / A12x / 原样 Xie 三支（同种子同读数，有就比）逐族比。"""
    J = json.load(open(os.path.join(PF.OUT, 'peakfix.json'), encoding='utf-8')); L = []
    fam = {k: v for k, v in J['fam']['real'].items() if not k.startswith('_')}
    R = os.path.join(AD.DEVDIR, 'results'); s_ = ARG.test_seed
    refs = [('旧 A12', 'a11_peakfix' if s_ == 4242 else f'a12x_orig_seed{s_}'), ('A12x', f'a12x_xie_seed{s_}')]
    if ARG.third != 'xie':
        refs.append(('Xie 三支', f'a15_xie_seed{s_}'))
    if NOB:
        refs.append(('A15', f'a15_seed{s_}'))
        if ARG.third != 'cnet':
            refs.append(('A15·无B', f'a15_noB_seed{s_}'))
        if ARG.a16a:
            refs += [('A16', f'a16_seed{s_}')] + ([('A16A', f'a16a_seed{s_}')] if ALG != 'A16A' else [])
            if ALG.startswith('A17·') or ALG == 'A18':                                  # A17 变体对照 A17（结果在 06_重建算法/A17）
                refs.append(('A17', os.path.join(os.path.dirname(ROOT), 'A17', 'results', f'a17_seed{s_}')))
        if ALG == 'A16':
            refs.append(('上次A16', f'a15_cnet0_noB_seed{s_}'))                   # 11-02 那次同配置运行（核对复现）
        if WA:                                                         # 两个试法：对照 A16 和它的 B 单独（新 B 原始输出，同读数）
            refs += [('A16', f'a16_seed{s_}'), ('A16的B', f'a16_seed{s_}', 't3')]
    refs = [(r[0], json.load(open(os.path.join(R, r[1], 'peakfix.json'), encoding='utf-8')), r[2] if len(r) > 2 else 'A11p') for r in refs
            if os.path.exists(os.path.join(R, r[1], 'peakfix.json'))]
    cols = ([('gate', '门控支'), ('t3', 'B原始' if WA else 'C原始'), ('t3c', 'B一致化' if WA else 'C一致化'), ('fix', '固定权重'),
             ('A11', 'A+B融合' if WA else 'A+C融合'), ('A11p', ALG), ('orc', '最优(上限)')] if NOB else
            [('gate', '门控支'), ('cons', '原网络支'), ('t3c', '第三支'), ('A2', '两支融合'), ('A11', '三支融合'), ('A11p', ALG)])
    L.append(f'\n== 汇总（真实模式，测试种子 {s_}，第三支 = {LAB3}）：各族形状误差；对照（同种子同读数）：' + '、'.join(r[0] for r in refs))
    L.append('   族                    ' + ''.join(f'{c[1]:>10s}' for c in cols) + ''.join(f'{nm:>10s}' for nm, _, _ in refs) + f'   {ALG} 对旧 A12')
    old = refs[0][1]['fam']['real']
    for k in fam:
        L.append(f'   {k:<20s}' + ''.join(f'{fam[k][c[0]]:10.4f}' for c in cols) + ''.join(f'{O["fam"]["real"][k][m_]:10.4f}' for _, O, m_ in refs)
                 + f'   {(fam[k]["A11p"] / old[k]["A11p"] - 1) * 100:+6.1f} %')
    med = lambda F, m: np.median([F[k][m] for k in fam])
    L.append('   各族中位              ' + ''.join(f'{med(fam, c[0]):10.4f}' for c in cols) + ''.join(f'{med(O["fam"]["real"], m_):10.4f}' for _, O, m_ in refs))
    gm = lambda F, m: float(np.exp(np.mean([np.log(F[k][m]) for k in fam])))
    L.append('   各族几何平均          ' + ''.join(f'{gm(fam, c[0]):10.4f}' for c in cols) + ''.join(f'{gm(O["fam"]["real"], m_):10.4f}' for _, O, m_ in refs))
    for nm, O, m_ in refs:
        ref = O['fam']['real']
        w = sum(fam[k]['A11p'] < ref[k][m_] - 1e-4 for k in fam); l = sum(fam[k]['A11p'] > ref[k][m_] + 1e-4 for k in fam)
        L.append(f'   {ALG} 对 {nm}：好 {w} 族、差 {l} 族')
    if not NOB:
        w = sum(fam[k]['A11'] < fam[k]['A2'] - 1e-4 for k in fam); l = sum(fam[k]['A11'] > fam[k]['A2'] + 1e-4 for k in fam)
        L.append(f'   三支融合 对 两支融合（峰位校正前，同一次重训）：好 {w} 族、差 {l} 族')
    for nm in PF.PKF:
        r = J['fampk']['real'][nm]
        L.append(f'   {nm} 峰位 |误差| 中位 nm：' + '，'.join(f'{c[1]} {r[c[0]]:.3f}' for c in cols)
                 + ''.join(f'，{n_} {O["fampk"]["real"][nm][m_]:.3f}' for n_, O, m_ in refs))
    s = J['sim']['real']; sa = lambda S, m: np.abs(np.array(S['sim']['real'][m]['dpk'])).mean()
    L.append('   5 束模拟读数平均 |峰位误差| nm：' + '，'.join(f'{c[1]} {np.abs(np.array(s[c[0]]["dpk"])).mean():.3f}' for c in cols if c[0] != 'orc')
             + ''.join(f'，{n_} {sa(O, m_):.3f}' for n_, O, m_ in refs))
    if NOB and TM['n']:
        L.append(f'   推理计时（本次运行全部 decode，{TM["n"]} 条读数）：门控支 {TM["gate"] / TM["n"] * 1e3:.2f} ms/条，C 支 {TM["t3"] / TM["n"] * 1e3:.3f} ms/条')
    for x in L:
        log(x)
    open(os.path.join(PF.OUT, 'log.txt'), 'w', encoding='utf-8').write('\n'.join(PF._L))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    import warnings; warnings.filterwarnings('ignore')
    if ARG.train_only:
        assert ARG.third != 'xie'
        prep(); sys.exit()
    log(f'{TAG}：' + (f'消融 A + C（门控 + {LAB3}一致化，去掉原网络支 B）' if NOB else f'三支融合（门控 + 原网络一致化 + {LAB3}一致化）') + f' + 峰位校正；测试种子 {AD.TEST_ST}')
    PF.main(); summary()
    if ARG.metrics:
        import a12_metrics as MT
        MT.ALG = ALG; MT.main()
