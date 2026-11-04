# -*- coding: utf-8 -*-
"""A18 单条读数延迟计时（A18-R1 第 7 条，2026-11-03 由 AI（Claude Code）写；写完未运行）。

为什么要它：A18 日志里的"门控支 92 ms/条"是整次运行全部 decode 的总时间 ÷ 17397 条读数，
  批量平摊、含融合验证集和理想模式读数、没排除同时在跑的其他任务，不是单条延迟。这里单独量 batch = 1 的延迟，并给 batch = 256 的平摊作对照。

读数：诊断集 all_fams(VSEED + 21, 30, jitter=False) 的真实模式读数，rng = default_rng(4321)，
  和 A18/diag/scale_ev.py 第 53、57–59 行完全相同的生成方式；不用测试集（测试种子 4242 / 6161 的谱和读数一条都不碰）。
  注意：这个诊断集的 CIE LED 族和测试集是同样 9 条实谱（jitter=False），但这里只计时、不报任何误差，不影响。
  从全部 26 族 × 30 条 × 4 次读数里固定随机挑：预热 20 条、单条计时 200 条、批量 256 条（三组互不重叠）。

模型：A18 设定（--third cnet0 --no-b --a16a 32 --a16a-ss --a16a-vscale --a16a-escale）。不训练任何网络：
  已存盘、直接读的：B（cnet0.pt）、零件网络（net8.pt）、旧 MLP（mlp.pt，只为 prep() 的接口）——都在 45_Uddin2024_真实R矩阵/results/ 下；
  峰位校正 q：读 A18/results/a18_seed6161/qnet.pt（6161 主运行存的；它在融合验证集的谱 + QSEED 9090 读数上训练，没看测试集）。只读不写。
  没存盘、必须现拟的：融合小网（net2）、一致化协方差和 κ、难族 k-means 簇、GaussX 先验。做法 = 直接调用 a15_three.build3()，
    它只用训练库和融合验证集（VSEED，非测试），和主运行同种子同流程；耗时约 440 s（已核实：A18/results/a18_seed6161/run_stdout.txt
    第 38 行"建模共 439 s"），其中大头是在融合验证集上跑一遍门控支给融合小网做特征，小网本身 3000 步只要几秒。
  没有选"随机初始化权重"那条路：build3 能在不碰测试集的前提下给出和主运行同一套权重，计时更贴近真实（虽然耗时和权重基本无关）。
  拿到 build3 返回的 decode_ac 后，用 inspect.getclosurevars 取出里面的各部件，按 decode_ac + a11_peakfix.dec2 的顺序逐步重放、逐段计时；
  重放结果先和 decode_ac + q 的原输出比对（8 条读数），不一致就中止，保证量的是同一个算法。

计时：每个部件前后 torch.cuda.synchronize()，time.perf_counter()。部件（顺序同 decode_ac）：
  B估亮度    t3_infer(Y) 取峰值缩放读数（decode_ac 第 1 行）
  escale     a16_ss.escale：11 个倍数各算两遍 GaussX 证据
  ParamE     PM.solve
  零件网络   H.infer + H.solve_pick
  GaussX     GF.solve_u（闭式解 + 类后验）
  fuse_parts DS.fuse_parts + gate_mean
  SS         GateSS.apply（内含一次 B 在 c·Y 上的推理，用于噪声协方差）+ 除回 c
  B+一致化   t3_infer + CS['t3'].apply
  融合小网   P18.feats（8 特征 + 族级类后验）+ blend3_w 前向 + 加权
  q          PF.featx + q_of + shift_np
  总计       整条流水线首尾（含各段间的同步开销）；"其他" = 总计 − 各段之和
  报中位数和 p90（ms）。另外 batch = 256 整批跑 3 遍（先预热 1 遍），每段时间 ÷ 256 取 3 遍中位，作吞吐对照。
  两者区别：单条延迟 = 一条读数从进到出的时间（实时测量时关心的）；批量平摊 = 一次喂 256 条、GPU 并行后摊到每条，通常小得多。

开头打印 GPU 型号、torch 版本、nvidia-smi 查到的 GPU 计算进程（本进程以外的进程会标出来，它们会拖慢计时），结束再查一次。
输出：06_重建算法/A18/results/latency_R1/{latency.json, log.txt}（新目录；已有 latency.json 就停，不覆盖）。

预计运行时长（推算）：build3 约 7.5 min（上面已核实的 439 s）+ 诊断集读数几秒 + 单条 220 次 × 约 0.1–0.5 s（推测：批量平摊是 92 ms/条，
  单条没有批并行，可能更慢）≈ 0.5–2 min + 批量 4 遍 × 256 条 × 约 0.1 s ≈ 1.5–2 min；合计约 10–12 min。
预计显存（推测，没实测过）：和 A18 主流程同量级，约 2–4 GB；内存约同主流程（训练库 87000 条 + 零件网络训练池）。脚本结束时打印 torch 峰值显存。
batch = 1 的路径从没跑过（主流程都是成批的）；若某部件不支持 M = 1，会在预热第一条就报错。

  cd 06_重建算法/算法对照
  py -I -X utf8 a18_latency.py
"""
import os, sys, json, time, inspect, hashlib, platform, subprocess, warnings

ALGDIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(ALGDIR), 'A18', 'results', 'latency_R1')
assert not os.path.exists(os.path.join(OUT, 'latency.json')), f'{OUT} 已有 latency.json，不覆盖；要重跑请换新目录名'
sys.path.insert(0, ALGDIR)
sys.argv = ['a15_three.py', '--third', 'cnet0', '--no-b', '--a16a', '32', '--a16a-ss', '--a16a-vscale', '--a16a-escale', '--test-seed', '6161']
sys.stdout.reconfigure(encoding='utf-8'); warnings.filterwarnings('ignore')
import numpy as np
import torch
import a15_three as M
import a16_ss

assert M.ALG == 'A18', M.ALG
PF, AD, P18, BM = M.PF, M.AD, M.P18, M.BM
NWARM, NSINGLE, NBATCH, NREP = 20, 200, 256, 3
QNET = os.path.join(os.path.dirname(ALGDIR), 'A18', 'results', 'a18_seed6161', 'qnet.pt')
PARTS = ['B估亮度', 'escale', 'ParamE', '零件网络', 'GaussX', 'fuse_parts', 'SS', 'B+一致化', '融合小网', 'q']
_L = []


def log(s=''):
    print(s, flush=True); _L.append(s)


def gpu_procs():
    """nvidia-smi 查 GPU 计算进程；返回 (原始文本, 本进程以外的行)。"""
    try:
        r = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=30)
        rows = [x.strip() for x in r.stdout.splitlines() if x.strip()]
        return rows, [x for x in rows if x.split(',')[0].strip() != str(os.getpid())]
    except Exception as e:                                             # 没有 nvidia-smi 也照跑，只记下来
        return [f'nvidia-smi 失败：{e}'], []


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ------------------------------------------------------------------ 环境
os.makedirs(OUT, exist_ok=True)
t_start = time.time()
dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU（无 CUDA）'
log(f'A18 单条延迟计时（A18-R1）  {time.strftime("%Y-%m-%d %H:%M:%S")}')
log(f'GPU {dev_name}；torch {torch.__version__}（CUDA {torch.version.cuda}）；Python {platform.python_version()}；{platform.platform()}')
log(M.code_hashes() + f'，a18_latency.py {hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:12]}')
rows0, other0 = gpu_procs()
log('开始时 GPU 计算进程：' + ('；'.join(rows0) if rows0 else '无') + f'（本进程 pid {os.getpid()}）')
if other0:
    log('   警告：有其他进程在用 GPU，计时会偏大：' + '；'.join(other0))

# ------------------------------------------------------------------ 建模（只用训练库和融合验证集）
t0 = time.time()
decode_ac, P, _ = AD.build()                                            # = a15_three.build3（NOB 分支返回 decode_ac）
t_build = time.time() - t0
log(f'build3 完成 {t_build:.0f} s（融合小网、κ、一致化协方差现拟；B / 零件网络读存盘）')
cv = inspect.getclosurevars(decode_ac).nonlocals
base = cv['base']; cb = inspect.getclosurevars(base).nonlocals; gate_of = cb['gate_of']; cg = inspect.getclosurevars(gate_of).nonlocals
Rn, CS, kap, GM, ipk, mu2, sd2, net2 = (cv[k] for k in ('Rn', 'CS', 'kap', 'GM', 'ipk', 'mu2', 'sd2', 'net2'))
GF, PM, sig, H, Pp, NT8, DS, GSS = (cg[k] for k in ('GF', 'PM', 'sig', 'H', 'P', 'NT8', 'DS', 'GSS'))
assert GSS is not None and M.ARG.a16a_escale and not M.STK
qd = torch.load(QNET, weights_only=False)                               # 本地自己存的文件（含 numpy 的 mu / sd）
assert abs(float(qd['dmax']) - PF.DMAX) < 1e-12
qmu, qsd = qd['mu'], qd['sd']
qnet = torch.nn.Sequential(torch.nn.Linear(len(qmu), 32), torch.nn.Tanh(), torch.nn.Linear(32, 1)).double().to(PF.DEV)
qnet.load_state_dict(qd['state']); qnet.eval()
log(f'q：读 {os.path.relpath(QNET, os.path.dirname(ALGDIR))}（λ = {qd["lam"]:g}，特征 {len(qmu)} 维），只读')


def pipeline(Y, rec):
    """Y (1, n, B) 原始幅度读数 → A18 输出 (1, NW, B)；= decode_ac + a11_peakfix.dec2 的 q（逐行对照 a15_three.py decode_ac / gate_of / base）。
    rec：部件名 → 累计秒。"""
    nd_ = Y.shape[0]
    sync(); t = ts = time.perf_counter()

    def lap(k):
        nonlocal t
        sync(); t1 = time.perf_counter(); rec[k] = rec.get(k, 0.0) + t1 - t; t = t1
    a = M.t3_infer(Rn, Y, net=M.XN)[:, BM, :].max(1); Ys = Y / np.maximum(a, 1e-30)[:, None, :]; lap('B估亮度')
    c = a16_ss.escale(GF, Ys)[:, None, :]; Yc = Ys * c; lap('escale')
    Xp = PM.solve(Yc, sig)[0]; lap('ParamE')
    X8 = H.solve_pick(Pp, Yc, H.infer(NT8, Yc))[0]; lap('零件网络')
    Xf, dPf, Wf = GF.solve_u(Yc); lap('GaussX')
    gp = DS.fuse_parts(Rn, Xf, Wf, dPf, [Xp, X8], Yc, GF, GF.classes); g = np.clip(DS.gate_mean(gp), 0, None); lap('fuse_parts')
    g = GSS.apply(Yc, g, Wf, np.clip(M.t3_infer(Rn, Yc), 0, None)); g = g / c; lap('SS')
    t3 = np.clip(M.t3_infer(Rn, Ys, g), 0, None)
    t3c = P18.unflat(CS['t3'].apply(P18.flat(t3), P18.flat(Ys), kap['t3']), nd_); lap('B+一致化')
    f8, lw = P18.feats(Rn, Ys, t3c, g, Wf, GM, ipk)                    # 同时存进 PF._F['last'] 给 q（同 decode_ac）
    W = M.blend3_w(np.c_[f8, lw], mu2, sd2, net2)
    A11 = P18.unflat(W[:, 0:1] * P18.flat(g) + W[:, 1:2] * P18.flat(t3c), nd_); lap('融合小网')
    o = dict(A11=A11, gate=g, w=(1 - W[:, 0]).reshape(nd_, -1))
    Fx, xa, d = PF.featx(o); q = PF.q_of(Fx, qmu, qsd, qnet); X = P18.unflat(PF.shift_np(xa, q * d), nd_); lap('q')
    rec['总计'] = rec.get('总计', 0.0) + t - ts
    return X


def reference(Y):
    """原实现：decode_ac + dec2 的 q（a11_peakfix.py dec2）。"""
    o = decode_ac(Y); Fx, xa, d = PF.featx(o); q = PF.q_of(Fx, qmu, qsd, qnet)
    return P18.unflat(PF.shift_np(xa, q * d), Y.shape[0])


# ------------------------------------------------------------------ 诊断集读数（同 A18/diag/scale_ev.py 第 53、57–59 行）
S_ = {k.replace('留出·', ''): v for k, v in AD.all_fams(P18.VSEED + 21, 30, P, False).items()}
gr = np.random.default_rng(4321)
Yall = np.concatenate([np.asarray(AD.meas(S.T, gr, 'real')).transpose(0, 2, 1).reshape(-1, Rn.shape[0]) for S in S_.values()])   # (26·30·nd, n)
perm = np.random.default_rng(8675309).permutation(len(Yall))          # 只用来抽读数下标；不和种子表（HANDOFF_A18-R1.md）里任何种子撞
assert len(perm) >= NWARM + NSINGLE + NBATCH, len(perm)
iw, isg, ib = perm[:NWARM], perm[NWARM:NWARM + NSINGLE], perm[NWARM + NSINGLE:NWARM + NSINGLE + NBATCH]
one = lambda i: Yall[i][None, :, None]                                  # (1, n, 1)
log(f'读数：诊断集 VSEED+21（{len(S_)} 族 × 30 条，真实模式，rng 4321），共 {len(Yall)} 条；预热 {NWARM}、单条 {NSINGLE}、批量 {NBATCH}（互不重叠）')

# 重放与原实现一致性（8 条成批）
Y8 = Yall[ib[:8]].T[None]
d_ref = float(np.abs(pipeline(Y8, {}) - reference(Y8)).max()); scl = float(np.abs(reference(Y8)).max())
log(f'重放 vs decode_ac + q（8 条）：最大差 {d_ref:.2e}（输出最大值 {scl:.3g}）')
assert d_ref <= 1e-6 * max(scl, 1e-30), '重放和原实现不一致，停止计时'

# ------------------------------------------------------------------ 单条（batch = 1）
for i in iw:
    pipeline(one(i), {})
recs = []
for i in isg:
    r = {}; pipeline(one(i), r); recs.append(r)
keys = PARTS + ['总计']
ms = {k: np.array([r[k] for r in recs]) * 1e3 for k in keys}
ms['其他'] = ms['总计'] - sum(ms[k] for k in PARTS)
# batch = 1 与成批输出是否一致（只报告：GPU 上分块大小不同可能有浮点级差别）
d_b1 = float(max(np.abs(pipeline(Yall[j][None, :, None], {})[0, :, 0] - pipeline(Y8, {})[0, :, jj]).max() for jj, j in enumerate(ib[:8])))

# ------------------------------------------------------------------ 批量（batch = 256）吞吐
Yb = Yall[ib].T[None]                                                   # (1, n, 256)
pipeline(Yb, {})
brec = []
for _ in range(NREP):
    r = {}; pipeline(Yb, r); brec.append(r)
bms = {k: float(np.median([r[k] for r in brec])) * 1e3 / NBATCH for k in keys}
bms['其他'] = bms['总计'] - sum(bms[k] for k in PARTS)

# ------------------------------------------------------------------ 报告
log(f'\n== 单条延迟（batch = 1，{NSINGLE} 条，预热 {NWARM} 条后）与批量平摊（batch = {NBATCH}，{NREP} 遍中位 ÷ {NBATCH}），单位 ms')
log(f'   {"部件":<12s}{"单条中位":>10s}{"单条 p90":>10s}{"批量平摊":>10s}')
for k in keys + ['其他']:
    log(f'   {k:<12s}{np.median(ms[k]):10.3f}{np.percentile(ms[k], 90):10.3f}{bms[k]:10.3f}')
log(f'   batch = 1 与成批（8 条）输出最大差 {d_b1:.2e}（只报告）')
log('   说明：单条 = 一条读数从进到出；批量平摊 = 一次喂 256 条摊到每条。A18 日志的 92 ms/条是整次运行批量平摊（含验证集、理想模式、未排除并发），和这两者都不同口径。')
rows1, other1 = gpu_procs()
peak = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0
log(f'结束时 GPU 计算进程：' + ('；'.join(rows1) if rows1 else '无') + ('' if not other1 else '  ← 有其他进程，计时可能偏大'))
log(f'torch 峰值显存 {peak:.2f} GB；总用时 {time.time() - t_start:.0f} s（其中 build3 {t_build:.0f} s）')

J = dict(task='A18-R1 第 7 条', time=time.strftime('%Y-%m-%d %H:%M:%S'), gpu=dev_name, torch=torch.__version__, cuda=torch.version.cuda,
         gpu_procs_start=rows0, gpu_procs_end=rows1, other_procs=bool(other0 or other1), code=M.code_hashes(),
         readings='all_fams(VSEED+21, 30, jitter=False)，real，rng 4321（同 A18/diag/scale_ev.py:53、57–59）', n_warm=NWARM, n_single=NSINGLE,
         n_batch=NBATCH, n_rep=NREP, qnet=QNET, build_s=t_build, replay_maxdiff=d_ref, b1_vs_batch_maxdiff=d_b1, peak_vram_gb=peak,
         single_ms={k: dict(median=float(np.median(ms[k])), p90=float(np.percentile(ms[k], 90))) for k in keys + ['其他']},
         batch_ms_per_reading=bms, single_raw_ms={k: ms[k].tolist() for k in keys})
json.dump(J, open(os.path.join(OUT, 'latency.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
open(os.path.join(OUT, 'log.txt'), 'w', encoding='utf-8').write('\n'.join(_L))
