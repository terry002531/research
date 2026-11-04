# -*- coding: utf-8 -*-
"""只跑 A18（测试种子 4242）的指标段：主流程已完成（qnet.pt、peakfix.json 在盘上），等价于 a15_three.py 末尾的 MT.main()。"""
import sys, warnings
ALGDIR = r'D:\OneDrive\Terry\01_Research\09_OPD端到端设计_光谱重建\06_重建算法\算法对照'
sys.path.insert(0, ALGDIR)
sys.argv = ['a15_three.py', '--third', 'cnet0', '--no-b', '--a16a', '32', '--a16a-ss', '--a16a-vscale', '--a16a-escale', '--test-seed', '4242']
sys.stdout.reconfigure(encoding='utf-8'); warnings.filterwarnings('ignore')
import a15_three as T
import a12_metrics as MT
import a16_ss
assert T.ALG == 'A18', T.ALG
MT.ALG = T.ALG
MT.main()
print('cholesky 兜底触发元素数：', a16_ss.NBAD[0], flush=True)
