# -*- coding: utf-8 -*-
"""只跑 A18 的指标段（a12_metrics.main()），主流程须已完成（PF.OUT 下有 qnet.pt、peakfix.json，只读不改）。
来源：11-03 15:22 scratchpad 里的同名脚本（原样副本在 A18/code/4242指标段/run_a18_metrics_only.py），A18-R1（2026-11-03）正式放进 算法对照/ 并改了三处：
  1. 输出不写原 metrics/，改写 A18/results/a18_seed<种子>/metrics_rerun_R1/（目录已存在且有 metrics.json 就停，不覆盖）；
  2. 每个光强档 k 记 cholesky 兜底的元素数 / 调用次数（a12_metrics 里按档清零），结束时打印全程累计；
  3. 日志头写代码 SHA256 前 12 位；种子 4242 时日志和图标题标明"开发种子"。
数值流程和原脚本相同（同一 a15_three 配置、同一 a12_metrics.main）。
  py -I -X utf8 run_a18_metrics_only.py [--test-seed 4242]
"""
import os, sys, warnings
ALGDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ALGDIR)
seed = int(sys.argv[sys.argv.index('--test-seed') + 1]) if '--test-seed' in sys.argv else 4242
sys.argv = ['a15_three.py', '--third', 'cnet0', '--no-b', '--a16a', '32', '--a16a-ss', '--a16a-vscale', '--a16a-escale', '--test-seed', str(seed)]
sys.stdout.reconfigure(encoding='utf-8'); warnings.filterwarnings('ignore')
import a15_three as T
import a12_metrics as MT
import a16_ss
assert T.ALG == 'A18', T.ALG
for f in ('qnet.pt', 'peakfix.json'):
    assert os.path.exists(os.path.join(T.PF.OUT, f)), f'主流程输出缺 {f}：{T.PF.OUT}'
OUT = os.path.join(T.PF.OUT, 'metrics_rerun_R1')
assert not os.path.exists(os.path.join(OUT, 'metrics.json')), f'{OUT} 已有 metrics.json，不覆盖；要重跑请换新目录名'
os.makedirs(OUT, exist_ok=True)
MT.OUT = OUT; MT.ALG = T.ALG
DEV_SEED = seed == 4242
if DEV_SEED:
    MT.NOTE = '测试种子 4242 = 开发种子（A12 起反复看过，非干净测试集）'
MT.log(T.code_hashes() + f'，a12_metrics.py {__import__("hashlib").sha256(open(os.path.join(ALGDIR, "a12_metrics.py"), "rb").read()).hexdigest()[:12]}')
MT.log(f'A18 指标段重跑（A18-R1）：测试种子 {seed}' + ('（开发种子，A12 起反复看过，非干净测试集）' if DEV_SEED else '') + f'；读 {T.PF.OUT} 的 qnet.pt / peakfix.json，输出 {OUT}')
MT.main()
MT.log(f'指标段结束，{T.chol_total()}')
open(os.path.join(OUT, 'log.txt'), 'w', encoding='utf-8').write('\n'.join(MT._L))
