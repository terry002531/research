"""Audit saved metrics and report the failed run. Does not rerun or repair it."""
from pathlib import Path
import csv,json,hashlib,math
import numpy as np
O=Path(__file__).resolve().parent;W=O.parents[2]
def j(n):return json.loads((O/n).read_text(encoding='utf-8-sig'))
def c(n):
    with (O/n).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def close(a,b):return bool(np.allclose(a,b,rtol=1e-7,atol=1e-9))
S=j('summary.json');F=c('family_metrics.csv');R=c('per_reading_metrics.csv');P=j('per_spectrum_deltas.json');H=j('preflight_hashes.json');E=j('process_exit.json');C=j('existing_file_changes.json');X=j('overlap.json');B=j('B1_31451_same_rule_reanalysis.json');old={r['family']:r for r in c('B1_31451_same_rule_reanalysis.csv')}
T=('B 稳定胜','A 稳定胜','边界·未校正显著','跨零');K=('B_raw','B_consistent','A','fusion_pre_q','final');checks={}
def ck(n,v):checks[n]=bool(v)
ck('26_unique_families',len(F)==26 and len({f['family'] for f in F})==26)
ck('3120_readings',len(R)==3120)
ck('finite_nonnegative_metrics',all(math.isfinite(float(r[k])) and float(r[k])>=0 for r in R for k in K))
for f in F:
    name=f['family'];rr=[r for r in R if r['family']==name];g={i:[] for i in range(30)}
    for r in rr:g[int(r['spectrum'])].append(r)
    ck(name+':structure',len(rr)==120 and all(len(v)==4 and {int(r['draw']) for r in v}=={0,1,2,3} for v in g.values()))
    for k in K:ck(name+':mean:'+k,close(np.mean([float(r[k]) for r in rr]),float(f[k])))
    d=np.array([np.mean([float(r['B_raw'])-float(r['A']) for r in g[i]]) for i in range(30)])
    rng=np.random.default_rng(1901122);boot=d[rng.integers(0,30,(20000,30))].mean(1)
    lo,hi=np.quantile(boot,[.025,.975]);bl,bh=np.quantile(boot,[.05/52,1-.05/52])
    tier='B 稳定胜' if bh<0 else 'A 稳定胜' if bl>0 else '边界·未校正显著' if hi<0 or lo>0 else '跨零'
    ck(name+':delta_and_intervals',close([d.mean(),lo,hi,bl,bh],[float(f[k]) for k in ('B_minus_A','paired_delta_ci_low','paired_delta_ci_high','bonf_ci_low','bonf_ci_high')]))
    ck(name+':corrected_tier',tier==f['tier'])
    ck(name+':spectrum_deltas',close(d,P[name]['d_B_minus_A']))
    for dest,key in [('eB','B_raw'),('eA','A'),('eBc','B_consistent'),('eF','final')]:ck(name+':spectrum:'+dest,close([np.mean([float(r[key]) for r in g[i]]) for i in range(30)],P[name][dest]))
    best=min(float(f['A']),float(f['B_consistent']))
    ck(name+':best_single_and_fusion',close([best,100*(float(f['final'])/best-1)],[float(f['best_single']),float(f['final_vs_best_single_percent'])]))
for k in K:ck('geometric:'+k,close(np.exp(np.mean(np.log([float(f[k]) for f in F]))),S['geometric_means'][k]))
for t in T:ck('tier_list:'+t,{f['family'] for f in F if f['tier']==t}==set(S['tiers'].get(t,[])))
ck('fusion_worse_list',{f['family'] for f in F if float(f['final_vs_best_single_percent'])>0}==set(S['fusion_worse_than_best_single']))
ck('reported_overlap_empty',X['diagnostic_vs_used']=={} and X['diagnostic_vs_old_diag']=={})
post=[]
for row in H['inputs']+[H['a16_ss'],H['b1']]:
    actual=hashlib.sha256((W/row['file']).read_bytes()).hexdigest();match=actual==row['sha256'];ck('hash:'+row['file'],match);post.append(dict(file=row['file'],sha256=actual,match=match))
ck('script_unmodified',hashlib.sha256((O/'diagnose_b3_newseed.py').read_bytes()).hexdigest()==H['script']['sha256'] and (O/'diagnose_b3_newseed.py').stat().st_size==21781)
required=['summary.json','family_metrics.csv','per_reading_metrics.csv','per_spectrum_deltas.json','compare_with_B1.csv','overlap.json','q_selection.json','architecture.json','run_validated.log','报告.md','existing_file_changes.json']
missing=[n for n in required if not (O/n).is_file()]
V=dict(full_run_passed=False,numerical_checks_passed=all(checks.values()),numerical_check_count=len(checks),failed_numerical_checks=[n for n,v in checks.items() if not v],exit_code=E['exit_code'],missing_deliverables=missing,original_outside_change_count=len(C),original_outside_changes=C,postflight_hashes=post,checks=checks,scope='Saved CSV/JSON arithmetic and bootstrap audit only. Does not override the failed run, assert outside files unchanged, recreate the missing report, rerun the algorithm, or access test spectra.')
(O/'partial_result_validation.json').write_text(json.dumps(V,ensure_ascii=False,indent=2),encoding='utf-8')
assert V['numerical_checks_passed'],V['failed_numerical_checks']
changes=[dict(family=f['family'],tier_old=old[f['family']]['tier'],tier_new=f['tier']) for f in F if old[f['family']]['tier']!=f['tier']]
with (O/'tier_changes_from_original_B1.csv').open('w',newline='',encoding='utf-8-sig') as out:
    wr=csv.DictWriter(out,fieldnames=['family','tier_old','tier_new']);wr.writeheader();wr.writerows(changes)
conclusion='；'.join(f"**{t} {len(S['tiers'].get(t,[]))} 族**："+('、'.join(S['tiers'].get(t,[])) or '无') for t in T)
changed_text='；'.join(f"{r['family']}（{r['tier_old']} → {r['tier_new']}）" for r in changes)
by={f['family']:f for f in F}
L=['# A18-B3：异常交接，结束保护断言未通过','',
f'**运行未通过完整核验**：26 族数值已保存，但第 330 行 `assert not changes, changes` 检测到目录外 8 处文件变化，进程退出码为 1，`报告.md` 未生成，`existing_file_changes.json` 非空。以下仅是已保存仿真输出的推算，不能标作完整成功交付：{conclusion}。与旧 B1（31451）同口径重算相比，换档 {len(changes)} 族：{changed_text}。','',
'作者：AI（Codex），2026-11-04。证据：**已核实**＝本次文件、原始运行记录或哈希；**推算**＝由已保存逐读数误差计算的统计量；**推测**＝未经归因核实的解释。全部为 Uddin 2024 实测 R 与既定噪声模型下的仿真，不是 Terry 器件实测。','',
'## 几何平均（已保存数值的推算）','',
'| B 原始 | B 一致化 | A | 融合 q 前 | 最终 |','|---|---|---|---|---|',
'| '+' | '.join(f"{S['geometric_means'][k]:.6f}" for k in K)+' |','',
'误差为无量纲 NRMSE；先算每族 30 条谱 × 4 次读数的算术均值，再对 26 个族均值取几何平均。B 为原始 cnet0，A 为 A18 门控支（SS、escale），最终为融合后再经 q。','',
'## 逐族（原报告缺失，交付缺项）','',
'**已核实**：原脚本在报告生成之前停止，因而无法按要求直接复制 `报告.md`。下面只为便于核查，按原脚本相同列和显示精度从已保存 CSV 展示；没有补造 `报告.md`，也不把完整交付标为通过。分档仅按校正后区间，边界档为仅未校正显著。','',
'| 谱族 | B | A | 最终 | B/A−1 % | 校正后区间（B−A） | 档 | 平均 w_A | 最终 vs 较好单支 % |','|---|---|---|---|---|---|---|---|---|']
for f in sorted(F,key=lambda r:float(r['B_vs_A_percent'])):L.append(f"| {f['family']} | {float(f['B_raw']):.4f} | {float(f['A']):.4f} | {float(f['final']):.4f} | {float(f['B_vs_A_percent']):+.1f} | [{float(f['bonf_ci_low']):+.4f}, {float(f['bonf_ci_high']):+.4f}] | {f['tier']} | {float(f['wA_mean']):.2f} | {float(f['final_vs_best_single_percent']):+.1f} |")
L+=['','## 与旧 B1 的比较（推算）','',
'旧 B1 只使用原 31451 的逐读数 CSV，按本次相同 B 原始−A 谱级差值、20000 次 bootstrap（seed=1901122）、26 族 Bonferroni（双侧 0.05/26）重算，未生成旧谱。不是后来 81063 的 log 比值加 5% 判据。旧重算为 B 稳定胜 10、A 稳定胜 3、边界 4、跨零 9；新种子为 10、3、6、7。','',
'| 换档族 | B1 旧档 | B3 新档 |','|---|---|---|']
L += [f"| {r['family']} | {r['tier_old']} | {r['tier_new']} |" for r in changes]
L+=['','## B1 的三个 A 稳定胜族（推算）','',
'| 谱族 | 新校正区间（B−A） | 新档 | 是否仍为 A 稳定胜 |','|---|---|---|---|']
for name in ('卤素灯 2000–3500 K','干涉波纹','LED+卤素'):
    f=by[name];L.append(f"| {name} | [{float(f['bonf_ci_low']):+.6f}, {float(f['bonf_ci_high']):+.6f}] | {f['tier']} | {'是' if f['tier']=='A 稳定胜' else '否'} |")
L+=['','## 融合劣于较好单支的族（推算）','',
f"共 {len(S['fusion_worse_than_best_single'])} 族。较好单支按原脚本定义取 min(A 族均值, B 一致化族均值)；百分比为 100×(最终/较好单支−1)。这是使用真值识别较好单支的诊断参照，不代表可部署路由，也未对这些差值作显著性判断。",'',
'| 谱族 | 较好单支 | 最终 | 绝对增加 | 相对增加 % |','|---|---|---|---|---|']
for name in S['fusion_worse_than_best_single']:
    f=by[name];best=float(f['best_single']);final=float(f['final']);L.append(f"| {name} | {best:.6f} | {final:.6f} | {final-best:+.6f} | {float(f['final_vs_best_single_percent']):+.2f} |")
L+=['','## 种子、查重、耗时与显存（已核实）','',
f"- 诊断 seed=31600，噪声 seed=8191；30 条/族，4 次 real 读数，jitter=True；26 族、780 条谱、3120 次读数。未启用备用种子。",
'- 种子断言通过；overlap.json 对训练/验证使用集和旧 +7/+13/+21 诊断集的重合字典均为空。按小数点后 9 位逐谱比较；本次旧 +21 查重依文档使用 jitter=True。',
f"- 原脚本计时 {S['seconds']:.1f} s（{S['seconds']/60:.2f} min），启动器墙钟 {E['wall_seconds']:.1f} s；未超时。",
f"- torch 峰值分配显存 {S['peak_cuda_GB']:.6f} GiB；字段名 peak_cuda_GB，代码按 2^30 换算。整卡显存含桌面程序另算；启动前未见其他 Python GPU 计算进程，见 gpu_before.txt。",
f"- Cholesky 兜底累计：元素数/有触发调用数 = {S['chol_fallback'][0]}/{S['chol_fallback'][1]}。q λ={S['q_lambda']}，只在原验证集选择。",
'- 审计/导入、内存初始化适配、建模与 q 拟合共用代码段已和旧 B1 逐字比较一致；见 source_review.json。','',
'## 哈希（已核实）','',
'原脚本未改，落盘 21781 字节；五项有指定基准的输入全部匹配，且运行前后相同。a16_ss.py 仅登记本次实算，不声称有未知历史基准。','',
'| 文件 | SHA256 | 核对 |','|---|---|---|',f"| diagnose_b3_newseed.py | `{H['script']['sha256']}` | 指定值匹配，运行后未改 |"]
for r in H['inputs']:L.append(f"| {r['file']} | `{r['sha256']}` | 指定值匹配；运行前后相同 |")
for r in (H['a16_ss'],H['b1']):L.append(f"| {r['file']} | `{r['sha256']}` | 本次实算；运行前后相同 |")
L+=['','## 失败位置与目录外变化','',
'**已核实**：原脚本 diagnose_b3_newseed.py 第 330 行 `assert not changes, changes` 失败；退出码 1，未超时。既有完整核验脚本也因 11 项交付缺失而退出 1，见 verification_failure.log。原始 existing_file_changes.json 保持不变；没有清空、回滚或排除这些变化来伪造通过。','',
'下列是原断言记录的 8 个路径；记录反映目录内其他位置发生变化，不直接证明写入主体。**推测**：B1/B2 的其他会话并发写入导致，但本轮没有核实归因，也未触碰或回滚这些文件。','',
'| 路径 | 变化 |','|---|---|']
for path,record in C.items():L.append(f"| {path} | "+('新增' if record['before'] is None else f"修改，{record['before']['size']} → {record['after']['size']} 字节")+' |')
L+=['','### 原始 stderr 最后 30 行','', '```text']+(O/'run_stderr.log').read_text(encoding='utf-8',errors='replace').splitlines()[-30:]+['```','', '### 原始 run_validated.log 最后 30 行','', '```text']+(O/'run_validated.log').read_text(encoding='utf-8',errors='replace').splitlines()[-30:]+['```','',
'## 交付与核验状态','',
f"- 指定 11 项文件中，存在 {len(required)-len(missing)} 项；缺失：{'、'.join(missing)}。existing_file_changes.json 存在但不是要求的空对象。这两项要求均未满足。",
f"- 保存结果的独立核算通过 {len(checks)} 项：逐读数均值、谱级配对差值、20000 次 bootstrap 校正区间、分档、几何平均、融合劣化、JSON/CSV 对齐与输入哈希。见 partial_result_validation.json：numerical_checks_passed=true，full_run_passed=false。这不覆盖或消除原运行错误。",
'- 按任务文档，已停止算法工作；未重跑、未更换种子、未修改原脚本或用本次结果调参。未训练或重训 B；未生成/读取 6161、4242 测试谱、读数或结果，未读取 A18/results 内容。',
'- 本轮新增 B3 诊断目录、此异常交接及核查辅助文件；按 AGENTS.md 单独续写同一普通项目会话记录。没有改写既有算法、权重或 B1/B2 文档。','',
'## 局限','',
'- 结束保护断言未通过，报告生成未执行；本轮只能作为带明确异常状态的已保存仿真输出核查材料。',
'- 仍只有一个诊断种子、同一器件响应矩阵和既定生成/噪声模型；每族 30 条谱。bootstrap 区间不代表跨器件或新光源稳定性；Bonferroni 尾部分位点存在 Monte Carlo 波动。',
'- 融合小网、κ、一致化协方差和 q 在 VSEED=31438 上按 B1 适配现拟，不是 6161 测试运行的逐位复现。调参入口由 60 条改为 VSEED 成员 40 条且共用噪声流，会改变后续融合验证噪声；σ_e 固定 0.08。',
'- 未生成测试谱，因而未与 6161/4242 测试谱逐条查重；只报告允许数据的查重结果。',
'- 两次种子差异同时包含谱形与噪声抽样差异，不能把差幅直接当成去除调参集偏差的因果效应；A 仍依赖 B 的亮度和噪声辅助量。','']
target=W/'A18/HANDOFF_A18-B3.md'
assert not target.exists(),'Existing handoff must not be overwritten'
target.write_text('\n'.join(L),encoding='utf-8')
print(json.dumps(dict(full_run_passed=False,numerical_checks_passed=True,numerical_check_count=len(checks),missing=missing,tiers={t:len(S['tiers'].get(t,[])) for t in T},tier_changes=changes,output=str(target)),ensure_ascii=False,indent=2))