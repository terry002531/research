"""Write the authorized handoff from verified diagnostics; no fitting or generation."""
from pathlib import Path
import json,csv
O=Path(__file__).resolve().parent; W=O.parents[2]
def j(n):return json.loads((O/n).read_text(encoding='utf-8-sig'))
def c(n):
    with (O/n).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
S=j('summary.json');V=j('verification.json');H=j('preflight_hashes.json');E=j('process_exit.json');B=j('B1_31451_same_rule_reanalysis.json');F=c('family_metrics.csv');old=c('B1_31451_same_rule_reanalysis.csv')
assert V['passed'] and E['exit_code']==0 and not E['timed_out']
T=('B 稳定胜','A 稳定胜','边界·未校正显著','跨零')
oldby={r['family']:r for r in old};newby={r['family']:r for r in F}
changed=[dict(family=r['family'],tier_old=oldby[r['family']]['tier'],tier_new=r['tier']) for r in F if oldby[r['family']]['tier']!=r['tier']]
allchanges=[dict(family=r['family'],tier_old=oldby[r['family']]['tier'],tier_new=r['tier'],changed=oldby[r['family']]['tier']!=r['tier']) for r in F]
with (O/'tier_changes_from_original_B1.csv').open('w',newline='',encoding='utf-8-sig') as f:
    wr=csv.DictWriter(f,fieldnames=list(allchanges[0]));wr.writeheader();wr.writerows(allchanges)
conclusions='；'.join(f"**{t} {len(S['tiers'].get(t,[]))} 族**："+('、'.join(S['tiers'].get(t,[])) or '无') for t in T)
change_text='；'.join(f"{r['family']}（{r['tier_old']} → {r['tier_new']}）" for r in changed) or '无'
report=(O/'报告.md').read_text(encoding='utf-8')
family_table=report.split('## 逐族\n\n',1)[1].split('\n\n## ',1)[0].rstrip()
lines=['# A18-B3：新种子诊断交接','',f'**推算，按预定的 B−A 成对差值区间分档**：{conclusions}。与旧 B1（31451）按同口径重算相比，换档 {len(changed)} 族：{change_text}。','',
'作者：AI（Codex）。全部结果为仿真，使用 Uddin 2024 公开实测 R 与项目既定噪声模型，不是 Terry 器件实测。',
'证据等级：**已核实**＝本次实际输出、文件或代码检查；**推算**＝由逐读数误差计算的均值、区间、比值及分档；**推测**＝未验证解释。本交接不提出改法或下一步方案。','',
'## 几何平均','',
'**推算**：每族先平均 30 条谱 × 4 次读数的 NRMSE，再对 26 个族均值取几何平均。NRMSE 无量纲，数值越小越好。B 为原始 cnet0；A 为当前 A18 门控支（含 SS、escale）；最终为融合后再加 q。','',
'| B 原始 | B 一致化 | A | 融合 q 前 | 最终 |','|---|---|---|---|---|',
'| '+' | '.join(f"{S['geometric_means'][k]:.6f}" for k in ('B_raw','B_consistent','A','fusion_pre_q','final'))+' |','',
'## 逐族','',
'下表直接逐字复制本次 `报告.md` 的逐族表。均值和统计量为**推算**；稳定分档仅由校正后的区间是否跨零决定，不按均值大小替代判据。B−A < 0 倾向 B；> 0 倾向 A。“边界”表示未校正 95% 区间不跨零，但校正区间跨零。','',family_table,'',
'## 与旧 B1 的分档变化','',
'**推算**：比较对象仅为原 B1 的 31451。旧 B1 的逐读数 CSV 按本次相同规则重算：每谱保留 4 次配对读数，先取 B 原始−A 的谱内均值，再做 20000 次 bootstrap（seed=1901122），26 族 Bonferroni，双侧 α=0.05/26。没有重新生成旧谱或读数，也没有采用后来 81063 报告中的 log 比值加 5% 门槛。旧重算结果保存为 `B1_31451_same_rule_reanalysis.csv/json`；全 26 族转档记录在 `tier_changes_from_original_B1.csv`。','',
'| 口径 | B 稳定胜 | A 稳定胜 | 边界·未校正显著 | 跨零 |','|---|---|---|---|---|',
'| B1：31451，同规则重算 | '+' | '.join(str(len(B['tiers'].get(t,[]))) for t in T)+' |',
f"| B3：{S['diagnostic']['seed']} | "+' | '.join(str(len(S['tiers'].get(t,[]))) for t in T)+' |','',
'| 换档族 | B1 旧档 | B3 新档 |','|---|---|---|']
lines += [f"| {r['family']} | {r['tier_old']} | {r['tier_new']} |" for r in changed]
if not changed:lines.append('| 无 | — | — |')
lines += ['','## B1 的三个 A 稳定胜族是否保留','', '**推算**，沿用校正后 B−A 区间判据：','',
'| 谱族 | B3 校正后区间（B−A） | B3 档 | 仍为 A 稳定胜 |','|---|---|---|---|']
for name in ('卤素灯 2000–3500 K','干涉波纹','LED+卤素'):
    r=newby[name];lines.append(f"| {name} | [{float(r['bonf_ci_low']):+.6f}, {float(r['bonf_ci_high']):+.6f}] | {r['tier']} | {'是' if r['tier']=='A 稳定胜' else '否'} |")
worse=S['fusion_worse_than_best_single']
lines += ['', '## 最终融合劣于较好单支的族','',
f'**推算**：共 {len(worse)} 族。按脚本既定口径，较好单支取 `min(A 族均值, B 一致化族均值)`。百分比为 `100×(最终/较好单支−1)`；本表按族均值比较，不据此声称差异显著。这个参照使用真值误差挑选较好单支，仅用于诊断。','',
'| 谱族 | 较好单支 NRMSE | 最终 NRMSE | 绝对增加 | 相对增加 % |','|---|---|---|---|---|']
for name in worse:
    r=newby[name];best=float(r['best_single']);final=float(r['final']);lines.append(f"| {name} | {best:.6f} | {final:.6f} | {final-best:+.6f} | {float(r['final_vs_best_single_percent']):+.2f} |")
d=S['diagnostic']
lines += ['', '## 运行、种子与查重','',
f"- **已核实**：诊断 seed={d['seed']}，噪声 seed={d['noise_seed']}；每族 30 条、4 次 real 读数，jitter=True，共 26 族、780 条谱、3120 条读数；bootstrap=20000。",
'- **已核实**：新种子与 R1 已用种子表的断言通过。`overlap.json` 中新诊断谱对训练/验证使用集、旧 VSEED+7/+13/+21 诊断集的重合字典均为空；逐谱比较口径为小数点后 9 位精确行比较。旧 +21 在本次查重中按文档强制 jitter=True。',
f"- **已核实**：计算计时 {S['seconds']:.1f} s（{S['seconds']/60:.2f} min）；启动器墙钟 {E['wall_seconds']:.1f} s。脚本计时从模型导入后开始，不含随后交接写作和独立核验。",
f"- **已核实**：torch 峰值分配显存 {S['peak_cuda_GB']:.6f} GiB（字段名 peak_cuda_GB，实际按 2^30 换算）；GPU 为 {S['architecture']['gpu']}。不代表整卡显存；启动前有桌面图形程序，未见其他 Python GPU 计算任务，见 gpu_before.txt。",
f"- **已核实**：Cholesky 兜底累计（元素数 / 有触发调用数）={S['chol_fallback'][0]} / {S['chol_fallback'][1]}。",
f"- **已核实**：q 的 λ={S['q_lambda']}，沿用原验证集选法；未以本次诊断结果调参。",
'- **已核实**：执行原命令，无原脚本修改。审计/导入、内存初始化适配、建模、q 拟合四个共用段与旧 B1 逐字一致，见 source_review.json。']
stderr=(O/'run_stderr.log').read_text(encoding='utf-8-sig',errors='replace')
lines += [f'- **已核实**：进程退出码 0，未超时；stderr 保留了 {stderr.count("RuntimeWarning:")} 条 RuntimeWarning 记录（exp overflow），未出现导致进程失败的异常，详见原始 run_stderr.log。','',
'## 哈希核对','',
'**已核实**：运行前后以下输入哈希均一致；前五项与任务文档指定值相同。脚本落盘后大小为 21781 字节。`a16_ss.py` 没有 B1 任务基准值，只登记本次实算值，不称与未知历史哈希一致。','',
'| 文件 | SHA256 | 核对 |','|---|---|---|',
f"| diagnose_b3_newseed.py | `{H['script']['sha256']}` | 指定值、大小匹配 |"]
for r in H['inputs']:lines.append(f"| {r['file']} | `{r['sha256']}` | 指定值匹配；运行前后相同 |")
lines += [f"| {H['a16_ss']['file']} | `{H['a16_ss']['sha256']}` | 实算；运行前后相同 |",f"| {H['b1']['file']} | `{H['b1']['sha256']}` | 存在；运行前后相同 |",'',
'## 文件、核验与范围','',
f"**已核实**：任务指定 11 项诊断交付全部存在，缺失项：无。独立 `verify_b3.py` 已通过 {V['checks_passed']} 项检查，从逐读数 CSV 重算族均值、谱级配对差值、bootstrap 区间、分档、几何平均及融合劣化量，并核对原表、JSON、报告、B1 对照与输入哈希。核验不重新导入重建算法或生成光谱。",
'**已核实**：原运行 `existing_file_changes.json` 为空对象；独立文件元数据核验也未发现受保护文件变化。运行结束后才新增本交接文件，避免干扰脚本的目录外变化断言。未修改已有代码、权重、诊断结果或文档；没有训练或重训 B，也没有读取 A18/results 或生成/读取 6161、4242 的测试谱、读数与结果。',
'新增项目文件限于 `A18/diag/B3_新种子_20261104/` 和 `A18/HANDOFF_A18-B3.md`；普通会话记录按用户 AGENTS.md 单独更新原项目 memory 文件。B3 目录另有原始 stdout/stderr、启动器状态、哈希记录、B1 同口径复算和独立核验辅助文件。','',
'## 局限','',
'- 仍是单一诊断种子、同一响应矩阵、同一生成机制与既定噪声模型；每族仅 30 条谱。bootstrap 描述本批生成谱的抽样波动，不证明跨器件或新光源稳定性。20000 次重采样在 Bonferroni 尾部分位点仍有 Monte Carlo 波动。',
'- 融合小网、κ、一致化协方差和 q 沿用 B1 适配在 VSEED=31438 上现拟，不是 6161 运行的逐位复现。调参入口从原 60 条适配为 VSEED 成员的 40 条，并与融合验证共用噪声随机流，会改变后续验证读数；A 的 σ_e 固定为 0.08。',
'- 未生成测试谱，因此没和 6161/4242 测试谱逐条查重。只报告已核验的训练/验证及旧诊断集重合情况，不声称完成测试集查重。',
'- 与旧 B1 的种子差异同时包含抽到的谱与读数差异，不能把差幅直接解释为去掉调参集偏乐观的因果效应。',
'- A 门控支仍使用 B 的亮度/噪声辅助量；本次 A/B 比较不代表把 B 完全移除后的独立 A。','']
target=W/'A18/HANDOFF_A18-B3.md'
assert not target.exists(),'Do not overwrite an existing handoff'
target.write_text('\n'.join(lines),encoding='utf-8')
assert family_table in target.read_text(encoding='utf-8')
print(json.dumps(dict(handoff=str(target),tier_counts={t:len(S['tiers'].get(t,[])) for t in T},tier_changes=changed,fusion_worse_count=len(worse)),ensure_ascii=False,indent=2))