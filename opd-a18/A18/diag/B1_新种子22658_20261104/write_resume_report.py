"""AI (Codex): append a reviewable report, preserving all previous handoff bytes."""
from pathlib import Path
import csv
import datetime
import hashlib
import json
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
from evaluation_support import annotate, summarize, csvwrite

OUT=Path(__file__).resolve().parent
WORK=OUT.parents[2]
load=lambda name:json.loads((OUT/name).read_text(encoding='utf-8'))
read=lambda p:list(csv.DictReader(p.open(encoding='utf-8-sig')))
names=lambda a:'、'.join(a) if a else '无'
def table(head,rows):
    return '\n'.join(['| '+' | '.join(head)+' |','|'+'|'.join(['---']*len(head))+'|']+
                     ['| '+' | '.join(str(x) for x in r)+' |' for r in rows])

j=load('summary.json'); rows=read(OUT/'family_metrics.csv')
qr=read(OUT/'q_ablation/family_metrics.csv'); q=j['q_decision']; st=j['statistics']
oldroot=OUT.parent/'B1_新种子81063_20261104'
oldj=json.loads((oldroot/'summary.json').read_text(encoding='utf-8'))
if (OUT/'comparison_81063_same_rule.json').exists():
    hashes=load('comparison_81063_input_hashes.json')
    assert all(hashlib.sha256((oldroot/k).read_bytes()).hexdigest()==v for k,v in hashes.items())
    oldst=load('comparison_81063_same_rule.json')
else:
    oldrows=annotate(read(oldroot/'family_metrics.csv'),read(oldroot/'per_reading_metrics.csv'))
    csvwrite(OUT/'comparison_81063_same_rule.csv',oldrows)
    oldst=summarize(oldrows)
    (OUT/'comparison_81063_same_rule.json').write_text(json.dumps(oldst,ensure_ascii=False,indent=2),encoding='utf-8')
    hashes={n:hashlib.sha256((oldroot/n).read_bytes()).hexdigest() for n in ('family_metrics.csv','per_reading_metrics.csv')}
    (OUT/'comparison_81063_input_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
astable=st['A_stable_vs_B_raw']; bstable=st['B_stable_vs_B_raw']
oldA=oldst['A_stable_vs_B_raw']; oldB=oldst['B_stable_vs_B_raw']
boundary=[r['family'] for r in rows if float(r['A'])<float(r['B_raw']) and r['family'] not in astable]
addA=[k for k in astable if k not in oldA]; removeA=[k for k in oldA if k not in astable]
gm=j['geometric_means']; qgm=lambda k:__import__('math').exp(sum(__import__('math').log(float(r[k])) for r in qr)/26)
now=datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %z')
parts=[]; add=parts.append
add('## 2026-11-04 22658 大样本评估与 q 消融（AI：Codex）')
add(f'作者：**AI（Codex）**。报告生成时间：{now}。以下全部为 Uddin 公开响应矩阵和既定器件误差模型下的**仿真结果及其统计推算**；不是 Terry 器件实测。执行依据为本文件「2026-11-04 12:00 预注册」。本节取代 81063 的 A/B 分工建议，历史内容保留以供审计。')
add(f'**主结论**：对旧 cnet0 原始输出，A 达到保守稳赢判据 **{len(astable)} 族**（{names(astable)}），B 达到判据 **{len(bstable)} 族**，其余 {len(st["uncertain_vs_B_raw"])} 族未定。q 在 31438 的预注册保留判据 **{"通过，B3 建议保留" if q["retain_q"] else "未通过，B3 不用 q"}**。没有实现、重训或选择新 B，不能据此宣布新 B 的停止线通过。')
add('### 数据、冻结和适用范围')
add('只调用一次 `all_fams(22658,150,P,True)`：26 族、每族 150 谱，每谱 4 次 real 带噪读数，共 3,900 谱、15,600 读数。所有族在训练范围内，jitter=True；500–800 nm、4 nm 网格的形状 NRMSE 定义沿用旧 B1。新增读数随机流固定为 1022658，避免沿用 81063 的 4321 流；因此两批变化包含谱形和噪声抽样差异。派生谱种子与实际生成阶段见 `generation_provenance.json`，不是另选评估种子。')
add('参考光功率固定为 10.5423395868 µW；未额外启用 --rerr 标定矩阵扰动。误差计算为：重建谱、真谱分别峰值归一，波段内 RMSE 再除以归一真谱的极差；先对每族全部谱及四次读数取算术平均，再对26族均值取几何平均。配对统计保留四次相关读数，独立抽样单位始终是谱。')
add('旧 81063 没有保存可直接加载的完整 A/融合 checkpoint。本轮只能按其相同旧 B1 适配流程，在原训练 31337、验证 31438 上确定性重建 A、κ 和融合小网；逐族验证 A、B 一致化、wA 与旧 B1 相符（绝对误差 <1e-9），B/融合/q 三个网络状态哈希与 81063 逐项一致。生成 22658 前冻结全部推理状态并禁用拟合入口。`full_model_freeze.json` 与 `full_model_post_evaluation.json` 内容相同，涵盖 A 高斯先验、零件模型、SS 字典/配置、一致化协方差、响应矩阵、融合及 q 的归一化和权重。81063 当时没有完整 A 指纹，所以跨运行的完整 A 逐字节一致性不能追溯证明。')
add('这仍是**旧 B1 适配配置**：旧调参谱入口从 60 条改为 31438 五类成员的 40 条，σ_e 固定 0.08，K=32，SS α=0.75，κ=0.25。B 一致化、融合、路由数字不代表正式 A18 测试运行。A 使用旧 cnet0 的亮度和 SS 噪声估计，不能把本结果当完全独立于 B 的 A；也不能给新 B 人为降质以维持 A 的作用。')
add('22658 与训练库、原调参/融合验证、零件网络训练池、高斯先验用谱、SS 调参集 31451，按小数点后 9 位逐谱比较无完全重复。只证明该精度下的完全重复为零；未证明近重复独立，也未生成 6161/4242/81063/63862 来做额外查重。未重新评估这些旧种子；81063 仅离线读取既有 CSV 对照，所有旧文件按内容 SHA-256 核验。')
add('### 12:18 中断后的续跑留痕')
resume=load('resume_before.json'); full=load('full_model_freeze.json'); freeze=load('model_freeze.json')
add('12:18 的原 run.log、process.log、诊断脚本及 q_ablation_partial_1218/ 均保留。没有复用部分 q 数值拼接结果；本次在全新 q_ablation/ 从第一族重做，使用同一 31438、同一噪声流 9090，先完成 q 决策才生成 22658。恢复采用 resume_diagnose_b1.py；旧 diagnose_b1.py 保留为中断版本，不把旧命令误称为本次运行命令。原配置重建只允许原训练/验证流程，进入评估后全部拟合入口禁用；完整状态与 15:16 登记逐项相等才放行。')
add(f'15:16 登记及本次实核的完整推理状态 SHA-256：`{full["sha256"]}`。原冻结文件逐字节未变。B、融合、q 网络状态 SHA-256 分别为 `{freeze["model_state_sha256"]["B"]}`、`{freeze["model_state_sha256"]["fusion"]}`、`{freeze["model_state_sha256"]["q"]}`；原 q checkpoint 内容 SHA-256 为 `{freeze["q_checkpoint_sha256"]}`。证据见 resume_before.json、q_decision_before_evaluation.json、resume_run.log 和完整前后状态指纹。')
add('### 保守统计判据与逐族结果')
add('每谱先对四次配对读数取 `log(NRMSE_A/NRMSE_B)` 平均，再以谱为单位重采样 200,000 次（seed=1901122）。百分位 bootstrap 与 Student t 两种区间分别报出，每族双侧 α=0.05/26，取两区间的外包络。A 稳赢要求保守上界 <0 且配对几何比 ≤0.95；B 稳赢要求保守下界 >0 且 B/A≤0.95。对 B 原始为主判据；对 B 一致化另作 26 族校正的辅助对照，不合称 52 项同时覆盖。未定不能解释成相等。族内算术平均 NRMSE 及其跨族几何平均，与配对 log 比是不同统计量。')
add(table(['谱族','B 原始','B 一致化','A','硬路由原始B','软融合 q前','软融合 q后'],[
    [r['family']]+[f'{float(r[k]):.5f}' for k in ('B_raw','B_consistent','A','hard_raw','fusion_pre_q','final')] for r in rows]))
add(table(['谱族','A/B 配对比','bootstrap log区间','t log区间','对原始B','对一致化B'],[
    [r['family'],f'{float(r["A_vs_B_raw_ratio"]):.4f}',
     f'[{float(r["A_vs_B_raw_bootstrap_log_low"]):+.4f}, {float(r["A_vs_B_raw_bootstrap_log_high"]):+.4f}]',
     f'[{float(r["A_vs_B_raw_t_log_low"]):+.4f}, {float(r["A_vs_B_raw_t_log_high"]):+.4f}]',
     'A稳赢' if r['family'] in astable else ('B稳赢' if r['family'] in bstable else '未定'),
     'A稳赢' if r['family'] in st['A_stable_vs_B_consistent'] else ('B稳赢' if r['family'] in st['B_stable_vs_B_consistent'] else '未定')]
    for r in rows]))
add(f'**A 保护集**：{names(astable)}。**暂留边界**（A 族均值较好但未达保守判据）：{names(boundary)}。**B 稳赢族**：{names(bstable)}。这些名单用于后续开发约束和分工讨论，不能用真值族名作为部署时可获得的路由输入。22658 看过结果后不再是新开发的盲评集；后续新 B 另设未使用评估种子。')
add('### 固定硬路由的准确率与条件误差')
add('固定规则 `wA>0.5` 选 A，否则选 B；没有在 22658 上调阈值。主表硬路由直接选 A 或 B 原始，辅助结果用实际融合候选 B 一致化，均不加 q。准确率按每次读数是否选到误差较小的一支计算，平局归 B；同谱四次读数相关，600 条读数不是 600 条独立谱。软融合可能优于两支，因此硬分类准确率不能替代融合误差。')
add('“分对时 A 误差”有两种条件：①所有分对读数上 A 本身的误差（含正确选 B 的读数）；②正确选 A 且 A 确实更好的读数上 A 的误差。下表同时报出，避免条件含混；后一条件是用真值诊断的子集，不能视作可预知的部署性能。')
add(table(['谱族','平均wA','wA>0.5占比','对原始B准确率','对一致化B准确率','正确选A数/600','所有分对时A误差','正确选A时A误差'],[
    [r['family'],f'{float(r["wA_mean"]):.3f}',f'{float(r["A_weight_gt_half"]):.1%}',f'{float(r["route_vs_B_raw_accuracy"]):.1%}',f'{float(r["route_vs_B_consistent_accuracy"]):.1%}',
     r['route_vs_B_raw_correct_A_n'],
     f'{float(r["route_vs_B_raw_A_error_when_correct"]):.5f}' if r['route_vs_B_raw_A_error_when_correct'] else '无样本',
     f'{float(r["route_vs_B_raw_A_error_when_correct_A"]):.5f}' if r['route_vs_B_raw_A_error_when_correct_A'] else '无样本'] for r in rows]))
add('### 与 81063 对照：相同保守判据重新算旧 CSV')
add('只读取 81063 既有逐读数 CSV，在新目录重算百分位+t 外包络；没有重新生成该种子的谱、读数或模型输出，未修改旧目录。不能直接把其原先仅用百分位区间的“4族”与本轮族数作同口径相减。')
add(f'同口径旧 81063：A 稳赢 {len(oldA)} 族（{names(oldA)}），B 稳赢 {len(oldB)} 族。本轮 A 新进入保护集：{names(addA)}；退出保护集：{names(removeA)}。样本量由每族30增至150，两批是不同谱池，不能归因为纯样本量效应、去泄漏代价或新算法收益。')
add(table(['阶段','81063（每族30）','22658（每族150）','相对变化'],[
    [k,f'{oldj["geometric_means"][k]:.6f}',f'{gm[k]:.6f}',f'{100*(gm[k]/oldj["geometric_means"][k]-1):+.2f}%']
    for k in ('B_raw','B_consistent','A','fusion_pre_q','final')]))
add('### q 消融：仅使用验证谱 31438')
add('加载原 q（λ=0.01）并冻结，不重训、不重选 λ。仅验证种子 31438、每族40谱×4次 real 读数；随机流9090，逐次对同一读数比较软融合加 q/不加 q。该谱池曾用于融合、q 拟合和早停；本结果是按授权进行的开发决策，不能当独立泛化证据。q 的去留在生成 22658 前已写入 `q_ablation/summary.json`，22658 的 q 后列仅用于完整诊断，不参与 q 决策。')
add(f'验证集跨族几何平均：q 前 `{qgm("q_off"):.6f}`，q 后 `{qgm("q_on"):.6f}`。预设整体配对几何比 `{q["ratio"]:.6f}`；95% bootstrap log 区间 `[{q["bootstrap_log_ci"][0]:+.6f}, {q["bootstrap_log_ci"][1]:+.6f}]`，分层 t 区间 `[{q["t_log_ci"][0]:+.6f}, {q["t_log_ci"][1]:+.6f}]`。整体区间上界须 <0，且没有任何一族的算术均值 NRMSE 变差超过10%；超过10%者：{names(q["any_family_worse_over_10_percent"])}。结论：**{"保留 q" if q["retain_q"] else "B3 不用 q"}**。只作这一个预设去留判断，不根据 22658 换 q。')
add(table(['谱族','q 前','q 后','变化'],[
    [r['family'],f'{float(r["q_off"]):.5f}',f'{float(r["q_on"]):.5f}',f'{float(r["change_percent"]):+.2f}%'] for r in qr]))
add('### 后续约束、核验与文件变更')
add(f'26族几何平均：硬路由 A/原始B `{gm["hard_raw"]:.6f}`；硬路由 A/一致化B `{gm["hard_consistent"]:.6f}`。软融合 q前相对原始B变化 `{100*(gm["fusion_pre_q"]/gm["B_raw"]-1):+.2f}%`，q后 `{100*(gm["final"]/gm["B_raw"]-1):+.2f}%`。这些是旧 B 下的点估计，不能冒充新 B 的“至少改善5%且至少5族A稳赢”停止线，也不能把软融合收益当硬路由收益。')
add(f'运行耗时 `{j["seconds"]:.1f} s`，本进程 torch 峰值分配显存 `{j["peak_cuda_GB"]:.2f} GB`；Cholesky 兜底元素/调用数 `{j["chol_fallback"][0]}/{j["chol_fallback"][1]}`。该计时包括配置重建、q 消融和新评估，不是 B 训练耗时。置信区间只描述既定生成器与噪声模型中的谱抽样波动；不覆盖新器件、新标定矩阵和现实光源分布。')
add('新增 `A18/diag/B1_新种子22658_20261104/`：评估脚本、预注册实施协议、内容哈希清单、完整推理状态指纹、逐读数/逐族数据、q_ablation、81063 同口径离线对照及本报告；仅向 `HANDOFF_A18-B1.md` 末尾追加本节。原算法、原权重和旧评估文件均不改。独立核验结果以本目录 `verification.json` 为准。')
add('实际续跑命令（输出目录禁止覆盖，勿直接在原目录重跑）：')
add('```powershell\npy -I -X utf8 A18\\diag\\B1_新种子22658_20261104\\resume_diagnose_b1.py --diag-seed 22658\npy -I -X utf8 A18\\diag\\B1_新种子22658_20261104\\write_resume_report.py\npy -I -X utf8 A18\\diag\\B1_新种子22658_20261104\\verify_resume.py\n```')
add('普通会话记录：已读取助手记忆管理规则；已识别总项目根目录的 memory 在当前写权限之外，本轮未改写该目录，也未另建替代记录来绕过权限。未收到本轮 Hook 的 memory_turn_id/绑定文件，不能声称 Hook 已核验。')
report='\n\n'.join(parts)+'\n'
(OUT/'report_22658.md').write_text(report,encoding='utf-8')
hand=WORK/'A18/HANDOFF_A18-B1.md'
current=hand.read_bytes()
before=(OUT/'HANDOFF_before.md').read_bytes()
assert current.startswith(before),'Existing handoff prefix changed; review before appending'
marker=parts[0].encode('utf-8')
assert marker not in current, 'Never append a duplicate result section'
added=b'\r\n\r\n'+report.replace('\n','\r\n').encode('utf-8')
hand.write_bytes(current+added)
(OUT/'handoff_append.json').write_text(json.dumps(dict(
    bytes_before_append=len(current),prefix_sha256=hashlib.sha256(current).hexdigest(),
    appendix_sha256=hashlib.sha256(added).hexdigest(),
    report_sha256=hashlib.sha256(report.encode('utf-8')).hexdigest()),indent=2),encoding='utf-8')
print(json.dumps(dict(report='report_22658.md',A_stable=astable,B_stable=bstable,
    q_retain=q['retain_q'],status='Report appended; independent verification pending'),ensure_ascii=False))
