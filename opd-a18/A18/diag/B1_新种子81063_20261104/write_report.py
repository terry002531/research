"""Write the 81063 evaluation report and append only its new handoff section."""
from pathlib import Path
import csv,json,hashlib,sys,datetime
import numpy as np

OUT=Path(__file__).resolve().parent; WORK=OUT.parents[2]; OLD=OUT.parent/'B1_诊断_20261104'
sys.path.insert(0,str(OUT))
from paired_stats import add_paired_statistics,summarize_statistics

def load(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def read_table(path):return list(csv.DictReader(path.open(encoding='utf-8-sig')))
def names(items):return '、'.join(items) if items else '无'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])

j=load('summary.json'); gm=j['geometric_means']; stats=j['statistics']
rows=read_table(OUT/'family_metrics.csv')
oldrows=read_table(OLD/'family_metrics.csv'); oldsamples=read_table(OLD/'per_reading_metrics.csv')
for sample in oldsamples:
    for key in ('A','B_raw','B_consistent'):sample[key]=float(sample[key])
oldrows=add_paired_statistics(oldrows,oldsamples)
with (OUT/'family_metrics_31451_reanalysis.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(oldrows[0])); w.writeheader(); w.writerows(oldrows)
oldstats=summarize_statistics(oldrows)
(OUT/'old_31451_reanalysis_summary.json').write_text(json.dumps(oldstats,ensure_ascii=False,indent=2),encoding='utf-8')

provenance=load('generation_provenance.json')
provenance['derived_seeds_from_verified_generator_source']=dict(
    main_shared_rng=81063,five_family_gen5=81064,CIE_jitter=81396,ood4=81563,parts_combo=81963)
(OUT/'generation_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')

ar=stats['A_stable_vs_B_raw']; ac=stats['A_stable_vs_B_consistent']
br=stats['B_stable_vs_B_raw']; uncertain=stats['uncertain_vs_B_raw']
rawA=[r['family'] for r in rows if float(r['A'])<float(r['B_raw'])]
rawB=[r['family'] for r in rows if float(r['B_raw'])<float(r['A'])]
boundary=[r['family'] for r in rows if float(r['A'])<float(r['B_raw']) and r['family'] not in ar]
degrade_raw=[r['family'] for r in rows if float(r['final'])>min(float(r['A']),float(r['B_raw']))]
degrade_cons=[r['family'] for r in rows if float(r['final'])>min(float(r['A']),float(r['B_consistent']))]
improve=j['final_improvement_over_old_B_raw_percent']
now=datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %z')
content=[]
add=content.append
marker='## 2026-11-04 新诊断种子 81063 重评（AI：Codex）'
add(marker)
add(f'作者：**AI（Codex）**。任务：A18-B1 评估重跑。报告生成时间：{now}。数值全部是带噪声的**仿真结果**，使用公开的 Uddin 响应矩阵，未验证 Terry 自己的 OPD。')
add('### 结论与判据')
add(f'**推算**：26 族几何平均，B 原始 `{gm["B_raw"]:.6f}`，B 一致化 `{gm["B_consistent"]:.6f}`，A `{gm["A"]:.6f}`，q 前融合 `{gm["fusion_pre_q"]:.6f}`，q 后融合 `{gm["final"]:.6f}`。q 后融合相对当前旧 B 原始误差降低 **{improve:.2f}%**；q 相对 q 前变化 `{100*(gm["final"]/gm["fusion_pre_q"]-1):+.2f}%`。')
add(f'**按预注册的校正判据**：A 对 B 原始稳赢 **{len(ar)} 族**：{names(ar)}；A 对 B 一致化稳赢 **{len(ac)} 族**：{names(ac)}。对 B 原始，B 稳赢 {len(br)} 族，未达到任一方向稳赢判据 {len(uncertain)} 族。均值上 A 较好 {len(rawA)} 族、B 较好 {len(rawB)} 族；均值领先与稳赢分开判断。')
add('老大登记的「A 仍有用」停止线针对**新 B**：融合比只用新 B 至少好 5%，并且至少 5 族 A 经校正稳赢。本轮使用现存 cnet0，尚未实现或重训新 B，所以只能提供旧 B 参照，不能宣布新 B 的停止线通过或失败。'+f' 若仅机械套用到当前旧 B，误差降低 ≥5%：{"满足" if improve>=5 else "不满足"}；A 稳赢 ≥5 族：{"满足" if len(ar)>=5 else "不满足"}。')
add('### 数据边界、冻结与可复现性')
add('**已核实**：显式参数 `--diag-seed 81063`，实际调用 `AD.all_fams(81063,30,P,True)`；每族 30 条、每谱 4 次 real 带噪读数，共 26 族、780 条谱、3120 条读数；噪声流仍为 `4321`。波段、误差定义、参考功率与旧 B1 一致，没有理想模式。所有族仍在训练范围，原源码日志中的“留出·”只是历史名字，CSV 已去掉此前缀。')
add('生成器内部派生流已按源码核实：主流 81063；同分布五类的 gen5 实际 seed=81064；CIE 随机平移/斜坡增强 seed=81396；ood4 seed=81563；零件组合 seed=81963。这里没有将 81063 替换成 VSEED+13；派生种子及调用阶段见 `generation_provenance.json`。')
add('**已核实**：B 直接加载旧 `cnet0.pt`；q 直接加载旧 B1 `q_validation_only.pt`，λ=0.01，未重训或重选。旧 B1 未保存完整融合模型，因此 A 先验、一致化及融合小网按旧 B1 的相同训练/验证流程在 31337/31438 上重建；K=32、SS α=0.75、σ_e=0.08，κ=0.25。生成 81063 之前，26 族融合验证结果的 A、B 一致化及平均 wA 已与旧 B1 逐项核对（绝对误差 <1e-9），源码和 B 权重哈希相符。')
add('模型重建后才生成 81063，随后禁用 κ 协方差拟合、融合训练、B 训练、K 选择和 q 训练入口。评估前后 B、融合小网与 q 的模型状态哈希一致。81063 没有用于选 K、权重、κ、早停、重训、选种子或挑方案。没有生成或读取 6161/4242 测试谱、测试指标或测试 q 权重。')
add('**已核实**：81063 谱与训练库、调参验证集、融合验证集、零件网络训练池、高斯先验用谱以及旧 SS 调参集 31451，按小数点后 9 位逐行比对，完全重合数为 0；见 `overlap.json`。没有为了查重去生成禁用测试谱，所以不声称完成与 6161/4242 的逐谱查重。')
add('### 配对 bootstrap 与多重比较')
add('**事前已固定的实现口径**：对每条谱的 4 次配对读数，先计算 `log(NRMSE_A/NRMSE_B)`，再对 4 次取平均；以谱为单位成对重采样 200000 次（seed=1901122）。点估计是这些谱级对数比的均值及其指数。每族双侧 α=0.05/26，百分位区间分位点为 0.00096153846、0.99903846154，26 族 Bonferroni 校正。A 稳赢要求 log 比区间上界 <0，且指数点估计 ≤0.95。B 稳赢用对称规则。B 原始和 B 一致化分别报告，每组校正 26 族；不能当成两组合并 52 项的同时覆盖保证。')
add('主 NRMSE 表与族间几何平均沿用旧 B1 的“先算族内算术均值，再做族间几何平均”；bootstrap 的配对几何比不是这两个族内算术均值之比。`family_metrics.csv` 同时保存两种比值，避免混用。旧 2000 次未校正 B−A 区间列仅为兼容保留，不用于本次“稳赢”分界。')
add('下表都是**已核实的仿真输出及其推算**；区间是经校正的 log(A/B) 区间，负值倾向 A。')
add(table(['谱族','B 原始','B 一致化','A','q 前','q 后'],[
    [r['family']]+[f'{float(r[k]):.5f}' for k in ('B_raw','B_consistent','A','fusion_pre_q','final')] for r in rows]))
add('对应的配对比值与校正区间如下；“未定”表示未达到任一方向的完整稳赢判据。')
add(table(['谱族','A/B 配对几何比','校正 log 区间','对 B 原始','对 B 一致化'],[
    [r['family'],f'{float(r["A_vs_B_raw_ratio"]):.4f}',
     f'[{float(r["A_vs_B_raw_log_ci_low"]):+.4f}, {float(r["A_vs_B_raw_log_ci_high"]):+.4f}]',
     'A 稳赢' if r['family'] in ar else ('B 稳赢' if r['family'] in br else '未定'),
     'A 稳赢' if r['family'] in ac else ('B 稳赢' if r['family'] in stats['B_stable_vs_B_consistent'] else '未定')]
    for r in rows]))
add('### 与旧 31451 结果比较')
add('**必须纠正旧叙述**：旧 B1 的 31451=VSEED+13、每族30条、jitter=True，就是 A 的 SS 调参谱；SS 的覆盖族、α、σa、τ 曾在这批谱上选定。它不是针对 SS 的独立诊断集，旧 A/B 分界可能偏向 A。新结论以本节 81063 为准，旧数字保留作为历史诊断。')
add('旧 B1 又把 build3 的调参谱从60条适配为31438成员的40条；调参读数与融合验证读数共用随机流 g，前者条数改变会改变随后整份融合验证噪声，所以 κ、融合小网、q 属于这一适配配置，不能声称逐位复现正式 A18。此次验证重建的是**同一个旧 B1 适配配置**，并沿用旧 B1 q 权重；不是恢复正式 A18 的测试运行。')
oldgm=j['old_B1_geometric_means']
labels={'B_raw':'B 原始','B_consistent':'B 一致化','A':'A','fusion_pre_q':'q 前融合','final':'q 后融合'}
add(table(['阶段','旧 31451','新 81063','相对旧变化'],[
    [labels[k],f'{oldgm[k]:.6f}',f'{gm[k]:.6f}',f'{100*(gm[k]/oldgm[k]-1):+.2f}%'] for k in labels]))
add(f'将旧 31451 的逐读数误差也用本次相同的 log 比/200000次/Bonferroni/5% 判据重算：A 对 B 原始稳赢 {len(oldstats["A_stable_vs_B_raw"])} 族（{names(oldstats["A_stable_vs_B_raw"])}），新 81063 为 {len(ar)} 族；A 对 B 一致化由 {len(oldstats["A_stable_vs_B_consistent"])} 族变为 {len(ac)} 族。复算仅使用旧 CSV，未重新生成31451性能读数；旧重算表写到新目录，未覆盖旧 B1 文件。')
add('旧审查中“校正后3族”使用的是 B−A 的差值区间；10:45预注册后来明确改为 log(A/B) 加5%点估计门槛。这里按后者统一复算两个seed，不能把两种统计量下的族数当成同一结果直接相减。')
add('两个 seed 的整体变化同时包含抽到的谱形差异和读数差异，不能把变化幅度直接等同于“去泄漏代价”，也不能从两个 seed 选择更好的一份报告。')
add('### 重新划定分工与融合问题')
add(f'**推算，A 保护集第一档（达到预注册稳赢判据）**：{names(ar)}。')
add(f'**推算，暂留边界（族内算术均值 A 较好，但未达到稳赢判据）**：{names(boundary)}。这部分保留观察，不标为 A 已可靠赢下。')
add(f'**推算，B 明确优势族**：{names(br)}。其余未定族：{names(uncertain)}。维持“以 B 为主，A 处理可核实的互补优势”方向；不会因为光谱名称属于多线或周期类，就强行把未达判据的族划给 A。')
add(f'**推算，融合需独立处理**：q 后融合有 {len(degrade_raw)} 族劣于 A/B 原始两支中更好的一支：{names(degrade_raw)}；按实际融合的 A/B 一致化比较，有 {len(degrade_cons)} 族：{names(degrade_cons)}。更好单支使用真值才能识别，只是诊断参照，不能称可部署路由。')
add('后续新 B 的开发应另设未使用的验证种子；81063 现在是看过结果的评估集，禁止转作训练、早停或权重选择。B3 最终评估另取新 seed。对保护族保留旧 B 辅助量的冻结对照，逐项检查 A 对 B 的亮度/SS噪声估计依赖；按同口径增加 q前/后融合和事前固定路由对照，不在81063上挑路由。')
add('本轮没有实现旧三种改进方案。若后续验证刷新谱池，须与已有 cnetS/按 A 误差加权的 bwA 失败先例区分：等预算、固定族采样的单项刷新对照；不把没有泛化差距的零件组合一并解释成过拟合；不复用未分折的 gate_lib 生成可靠性权重。先做哪项仍由老大决定。')
add('### 局限、文件与核验')
add('bootstrap 仅描述这批生成谱在既定噪声模型中的抽样波动；同一器件 R、同一生成机制和固定 CIE 原谱库没有改变，不能代表跨器件或新光源的置信度。每族仅30条谱；多重校正后的尾部区间仍有 Monte Carlo 误差。全部结果不能当成 OPD 实测、实测分辨率或论文最终盲测数字。')
add(f'**已核实**：诊断计算 {j["seconds"]:.1f} s，torch 峰值分配显存 {j["peak_cuda_GB"]:.2f} GB；Cholesky兜底累计元素/调用数 {j["chol_fallback"][0]}/{j["chol_fallback"][1]}。运行计时不包括后续报告生成与核验；显存是本进程torch分配值。')
if (OUT/'scope_review.json').exists():
    add('**并发范围核验说明**：所有数值和汇总已保存后，原进程在最后的全目录差异断言以退出码1结束：运行期间 `HANDOFF_A18-B2.md` 被外部追加了署名“科研牛马（实验）”的副导师审查。B1运行时审计拦截器禁止写诊断目录外文件；旧B1的25个文件及82份算法源码哈希未变。本轮未覆盖或回滚B2，保留原始 `process.log` 的断言错误及 `existing_file_changes.json`，并用 `scope_review.json` 单独登记。没有重跑数值或调整参数；最终有效性由独立 `verify_b1.py` 核验，不把原进程退出码改写为0。')
add('新增目录：`A18/diag/B1_新种子81063_20261104/`。关键交付：`diagnose_b1.py`、`paired_stats.py`、`family_metrics.csv`、`per_reading_metrics.csv`、`summary.json`、`run.log`、`process.log`、`protocol.json`、`generation_provenance.json`、`model_freeze.json`、`model_post_evaluation.json`、`overlap.json`、`diagnostic_spectra.npz`、`source_manifest.json`、`old_artifacts_before.json`、`family_metrics_31451_reanalysis.csv`、`old_31451_reanalysis_summary.json`、`write_report.py`、`verify_b1.py`、`verification.json`。核验状态以 `verification.json` 为准。')
add('复现命令（当前结果目录受防覆盖断言保护；重跑需先复制脚本到另一独立诊断目录）：\n\n```powershell\npy -I -X utf8 A18\\diag\\B1_新种子81063_20261104\\diagnose_b1.py --diag-seed 81063\npy -I -X utf8 A18\\diag\\B1_新种子81063_20261104\\write_report.py\npy -I -X utf8 A18\\diag\\B1_新种子81063_20261104\\verify_b1.py\n```')
record_status=load('session_record_status.json') if (OUT/'session_record_status.json').exists() else dict(status='pending_location_confirmation',detail='总项目根目录的 memory 超出本轮写权限；已询问是否归入算法子项目 memory，未收到确认前不另存。')
add('文件变更范围：新增上述诊断目录；在 `HANDOFF_A18-B1.md` 末尾追加本节。旧 B1 目录、算法源码、已存权重和测试结果均未改动；原交接内容逐字节保留，见核验文件。')
add('普通会话记录状态：'+record_status['detail'])
new_report='\n\n'.join(content)+'\n'
report=OUT/'report_81063.md'
previous=report.read_text(encoding='utf-8') if report.exists() else None
old_bytes=(OUT/'HANDOFF_before.md').read_bytes()
hand=WORK/'A18/HANDOFF_A18-B1.md'
current=hand.read_bytes()
expected_previous=old_bytes+b'\n\n'+previous.encode('utf-8') if previous is not None else None
assert current==old_bytes or current==expected_previous, 'Handoff changed externally; do not overwrite.'
report.write_bytes(new_report.encode('utf-8'))
hand.write_bytes(old_bytes+b'\n\n'+new_report.encode('utf-8'))
print(json.dumps(dict(report=str(report),appended_handoff=str(hand),
    A_stable_raw=ar,A_stable_consistent=ac,improvement_percent=improve,
    old_handoff_preserved=True),ensure_ascii=False,indent=2))
