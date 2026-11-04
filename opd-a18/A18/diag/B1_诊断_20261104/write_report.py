"""Build the A18-B1 handoff from verified, validation-only diagnostic artifacts."""
from pathlib import Path
import json, csv, hashlib
import numpy as np

OUT=Path(__file__).resolve().parent
WORK=OUT.parents[2]
HANDOFF=WORK/'A18/HANDOFF_A18-B1.md'
before=json.loads((OUT/'existing_files_before.json').read_text(encoding='utf-8'))
assert str(HANDOFF.relative_to(WORK)) not in before, 'Never overwrite a pre-task handoff.'
if HANDOFF.exists():
    assert 'task_id: A18-B1' in HANDOFF.read_text(encoding='utf-8'), 'Only regenerate this task artifact.'
j=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
def readcsv(name):
    return list(csv.DictReader((OUT/name).open(encoding='utf-8-sig')))
rows=readcsv('family_metrics.csv'); gaps=readcsv('train_validation_gap.csv')
f=lambda r,k:float(r[k])
betterB=[r for r in rows if f(r,'B_minus_A')<0]
betterA=[r for r in rows if f(r,'B_minus_A')>0]
sigB=[r for r in rows if f(r,'paired_delta_ci_high')<0]
sigA=[r for r in rows if f(r,'paired_delta_ci_low')>0]
unc=[r for r in rows if f(r,'paired_delta_ci_low')<=0<=f(r,'paired_delta_ci_high')]
pa=[r for r in rows if f(r,'wA_mean')>.5]
pb=[r for r in rows if f(r,'wA_mean')<=.5]
gm=j['geometric_means']; gg=j['generalization_means']
gap=gg['validation']['shape_nrmse']-gg['train']['shape_nrmse']
diag_gap=gg['diagnostic']['shape_nrmse']-gg['train']['shape_nrmse']
objgap=gg['validation']['training_objective']-gg['train']['training_objective']
names=lambda rs:'、'.join(r['family'] for r in rs) or '无'
topB=sorted(betterB,key=lambda r:f(r,'B_vs_A_percent'))[:5]
topA=sorted(betterA,key=lambda r:f(r,'B_vs_A_percent'),reverse=True)[:5]
lines=[]
def add(s=''):lines.append(s)
add('---\ntask_id: A18-B1\nbranch: B 诊断\nauthor: AI（Codex）\ndate: 2026-11-04\nstatus: 诊断完成；方案未实现\n---')
add('# A18-B1：A、B 的分工边界诊断')
add('本报告所有误差均为**仿真结果**，不是 Terry 器件实测。采用 Uddin 公开实测响应矩阵和项目既定噪声模型。本轮目标是让重新设计的 B 与 A 分工互补，取代旧交接文档中“B 不动”的任务方向；不要求 B 赢遍所有谱族。')
add('证据标签：**已核实**＝本轮亲自读到源代码/原日志或实际跑出；**推算**＝由这些仿真输出计算的差值、百分比、统计区间与判断；**推测**＝尚未实施的方案及其收益、成本估计。下文每张表在标题中统一声明其数字的等级，不把方案预期当结果。')
add('## 0. 执行范围与数据边界')
add('先读了指定的纠正记录、A18 交接文档、README、R1 交接和版本说明；`06_agent/_当前焦点.md` 不存在（已核实）。先检查 `A18/diag/scale_ev.out`、`fuse_diag2.out` 与 A17 诊断日志：已有缩放/SS 数字，缺少本轮所需的完整逐族 A/B/最终融合及同口径训练—验证误差。历史文档中的测试结果未用于本轮分析或方案取舍。')
add('B 与零件网络直接读取已存训练权重；A 先验、κ、融合小网沿用现行 R1 数值函数从训练/验证集重建，q 按原选择流程在验证谱上重新拟合。没有重训 B，没有改任何原算法、原结果或原文档，没有读取测试目录里的 q 权重；这是一份“原数值函数在允许数据上重建”的诊断，不声称逐位复现旧测试运行。')
add('执行中发现旧 build3 另外调用种子 3 的调参验证谱，第一次尝试已在 q 特征阶段停止，其日志保留为 `attempt1_legacy_validation_excluded.log`，其数字全部排除。第二次适配出现重新选择 σ 的入口，亦在建模阶段停止，见 `attempt2_sigma_selection_excluded.log`。最终有效运行只用 VSEED=31438 中“同分布五类”的完整 40 条谱作为原调参入口数据；该成员实际派生种子为 31439，并将 σ_e 固定为既有允许诊断日志中的 0.08，避免改成另一配置的 A。没有另增验证谱，也不使用前次输出选择参数。适配说明见 `strict_validation_adapter.json`，本报告所有新数值仅来自 `run_validated.log`。这是必要的数据来源适配，会使重建模型与旧运行有差异。')
add('**已核实的数据设定**：训练主种子 `31337`，训练库 `87,000` 条（`26×3,000` 加同分布五类 `9,000` 条）；融合验证谱 `VSEED=31438`，每族 `40` 条；独立于本轮拟合的诊断谱 `VSEED+13=31451`，每族 `30` 条、`jitter=True`，每条 `4` 次带噪读数。主逐族表用 `real` 模式和噪声随机流 `4321`，共 `3,120` 条读数；不使用理想模式。')
add('**已核实的仿真误差设定**（`forward.py:53` 与 `a11_device.py:211–284`）：暗电流整体漂移标准差 `1.5%`，对数增益标准差 `0.15`，白噪声标准差 `0.3 nA`；训练时光功率在 `0.5–2 P0` 内按对数均匀抽样。主表 `real` 模式取参考功率，不额外启用 `--rerr` 标定矩阵扰动。')
add('VSEED+21 的固定 CIE LED 原谱与测试谱重复，本轮整组不用；VSEED+13 的 CIE LED 用随机平移/斜坡增强版。诊断谱与训练库、调参验证谱、融合验证谱、零件网络训练池、高斯先验用谱按小数点后 `9` 位逐行比较，重合数为 `0`（已核实：`overlap.json`）。没有生成测试谱来查重；因此这里只证明与允许的开发数据不重合，不声称完成了与测试集的查重。')
add('运行适配只在内存中屏蔽导入时的作者测试电流读取及无关测试对象初始化，输出路径改至诊断目录；数值函数沿用原代码。模拟器参考功率固定为历史标定常数 `10.5423395868 µW`（已核实：原配置数值），没有重新利用作者测试读数拟合它。变更列表见 `runtime_initialization_changes.json`；脚本带目录写入与测试结果读取拦截。')
add(f'实际本次计算用时 `{j["seconds"]:.1f} s`，torch 峰值显存 `{j["peak_cuda_GB"]:.2f} GB`（已核实；含现拟融合与 q，不是单独 B 训练时间）。')
add(f'本次 Cholesky 兜底累计为“元素数 / 有触发的调用数” `{j["chol_fallback"][0]} / {j["chol_fallback"][1]}`（已核实）。它只描述本次允许数据的运行，不用于判断旧测试指标段。')
add('## 1. B（cnet0）到底是什么')
a=j['architecture']
add(f'**已核实**：`算法对照/a15_three.py:148–164` 定义 `CNet`；本轮加载权重实数统计为 **{a["parameters"]:,} 个可训练参数**。输入是 `{a["input"]}` 个偏压读数，经绝对值和的 L1 归一化。结构是 `Linear(41,512)`，随后 `4` 个残差块，每块为 `LayerNorm(512) → Linear(512,512) → GELU → Linear(512,512)`，最后 `LayerNorm → GELU → Linear(512,76)`。没有卷积、注意力、显式响应矩阵层或谱族专用输出头。')
add('**已核实**：训练输出经 Softplus 保证非负，再除以波段内峰值加 `1e-6`；输出覆盖 `500–800 nm`、间隔 `4 nm` 的 `76` 点。推理 `t3_infer` 把输出嵌回 `176` 点公共网格，波段外为零，用响应矩阵与读数的非负最小二乘标量定幅度。B 的形状网络本身不输入 R，R 用于生成训练读数和推理后的定幅度/一致化。')
add('**已核实的训练损失**（`a15_three.py:167–232`）：')
add(r'$$\mathcal L_B=\operatorname{mean}(\hat s-s)^2+\tfrac12 r\,\operatorname{mean}(\Delta\hat s-\Delta s)^2,\qquad r=\frac{\langle s^2\rangle_{\rm train}}{\langle(\Delta s)^2\rangle_{\rm train}}.$$')
add(f'`Δ` 是相邻波长点的一阶差分；两谱均按峰值归一。`r={j["training_objective_r"]:.6f}`（已核实：从训练库重算；原日志四舍五入为 `44.4`）。`cnet0` 仅去掉 cnet 的互补样本加权，导数项仍在；没有按族加权，也没有读数一致性损失。')
add('**已核实的训练设置**：AdamW，初始/OneCycle 最大 LR `1e-3`，weight decay `1e-4`，OneCycleLR；`30,000` 步，batch `2,048`，初始化种子 `11`。在 `87,000` 条固定谱中有放回抽样，每步重新抽功率与器件噪声；numpy RNG 每 `50` 步由 `11×100000+step` 重建，torch 也参与噪声抽样。总呈现 `61,440,000` 条训练读数、约相当于训练库 `706.2` 次抽样（推算；不是独立光谱条数）。cnet0 没有单独验证早停，保存最后一步权重。')
add('**已核实的原训练日志**：`算法对照/45_Uddin2024_真实R矩阵/results/run_a15_train.txt:9–16` 记录 step `0` 损失 `1.67317`，step `25,000` 为 `0.06353`，step `29,999` 为 `0.05411`，耗时 `151 s`。这些是训练中各次随机批的目标值，日志没有相同目标下的验证曲线；不能把最后一批损失直接当验证形状误差。')
add('**来源边界（已核实：本地代码/文档有写）**：`a15_three.py:7–15` 明说借自 **Xie 2026b** 的“L1 归一读数 + 残差 MLP 骨架”。本地《OPD光谱重建_已有算法与效果.md》末尾参考文献将其标成 *Advanced Optical Materials* **14, e71860**, DOI `10.1002/adom.71860`。cnet0 的 pre-LN/GELU、残差块数和导数项是本项目的改写，不能说直接复制原网络；也没有证据把这些工程差异称为已验证创新。本轮没有另行核验该论文全文或把原论文性能当本项目性能。')
add('## 2. 各族上谁更好、差多少')
add('口径（已核实：`ladder.py:270–274`）：先将重建与真谱分别峰值归一，波段内 RMSE 再除以归一真谱的极差。每族先平均所有谱与噪声抽样；族间总括指标为各族均值的几何平均。B＝原始 cnet0；A＝现行 A18 中经 escale、ParamE/零件组合及 SS 的完整门控解；最终＝B 一致化后与 A 融合，再经 q。A 仍依赖 B 的亮度估计和 SS 噪声估计，不是完全移除 B 的独立算法。')
add(f'**已核实，诊断集仿真几何平均**：B 原始 **{gm["B_raw"]:.6f}**，B 一致化 **{gm["B_consistent"]:.6f}**，A **{gm["A"]:.6f}**，融合 q 前 **{gm["fusion_pre_q"]:.6f}**，最终融合 **{gm["final"]:.6f}**。')
add(f'**推算**：最终相对 B 原始变化 `{100*(gm["final"]/gm["B_raw"]-1):+.2f}%`；q 相对 q 前变化 `{100*(gm["final"]/gm["fusion_pre_q"]-1):+.2f}%`。按族均值，B 优于 A `{len(betterB)}` 族，A 优于 B `{len(betterA)}` 族；这不是逐条样本都同样优劣。')
add(f'本次 q 选择的 λ 为 `{j["q_lambda"]}`（已核实：`q_selection.json`）。该文件的峰位 MAE 是所有满足 `single_peak` 条件的验证样本的平均，不是专门的窄线族峰位精度。')
add('下表 B/A/最终为**已核实的仿真 NRMSE**；“B−A”及百分比为**推算**。负数＝B 更好，正数＝A 更好；百分比分母是 A。完整精度与 B 一致化、q 前融合见 `family_metrics.csv`。')
add('| 谱族 | B 原始 | A | 最终 | B−A | (B/A−1) % |\n|---|---:|---:|---:|---:|---:|')
for r in rows:add(f'| {r["family"]} | {f(r,"B_raw"):.5f} | {f(r,"A"):.5f} | {f(r,"final"):.5f} | {f(r,"B_minus_A"):+.5f} | {f(r,"B_vs_A_percent"):+.1f} |')
add(f'**推算，分工边界**：B 的优势族为：{names(betterB)}。A 的优势族为：{names(betterA)}。B 优势较大的例子：'+ '；'.join(f'{r["family"]}低于 A {abs(f(r,"B_vs_A_percent")):.1f}%' for r in topB)+'。A 优势较大的例子：'+'；'.join(f'{r["family"]}中 B 高于 A {f(r,"B_vs_A_percent"):.1f}%' for r in topA)+'。')
add(f'**推算的稳定性检查**：以谱为单位对 B−A 做 `2,000` 次成对 bootstrap，保留每条谱的全部 `4` 次噪声抽样。经验 `95%` 区间全负 `{len(sigB)}` 族、全正 `{len(sigA)}` 族、跨零 `{len(unc)}` 族。跨零族：{names(unc)}。这里只是单个诊断种子下的样本波动估计，未校正多族比较，不能当跨器件置信度。')
add('## 3. 训练误差与验证误差：是否过拟合')
add('**已核实的对齐方法**：每族从训练库固定抽 `400` 条，验证集用原 `40` 条，诊断集用 `30` 条；都用训练模式的随机功率 `0.5–2 P0` 和同一器件噪声模型、每谱 `4` 次读数。分别重算原损失的值项/导数项，以及项目形状 NRMSE；不拿主表固定参考功率的 `real` 数字直接和训练比较。抽取训练谱后没有更新权重。')
add('下表为**已核实的仿真误差**，各族等权算术平均；与上节几何平均含义不同。')
add('| 数据 | B 形状 NRMSE | B 原训练目标 |\n|---|---:|---:|')
for l,cn in [('train','训练谱抽样'),('validation','验证谱 31438'),('diagnostic','诊断谱 31451')]:add(f'| {cn} | {gg[l]["shape_nrmse"]:.6f} | {gg[l]["training_objective"]:.6f} |')
add(f'**推算**：验证−训练形状差 `{gap:+.6f}`（相对训练 `{100*gap/gg["train"]["shape_nrmse"]:+.2f}%`），诊断−训练 `{diag_gap:+.6f}`；验证−训练目标差 `{objgap:+.6f}`。不能用这一个冻结 checkpoint 确诊“训练不足”或“网络容量不足”；需要后续验证学习曲线及针对结构的消融。')
add(f'**推算的判断：存在局部过拟合迹象，同时有难族拟合/可辨识性瓶颈，不能只贴一个标签。** 原训练目标的验证均值比训练高 `{100*objgap/gg["train"]["training_objective"]:.1f}%`，比形状平均差距更突出。')
for name in ('荧光灯式多线','窄线 4–10 nm','激光线+宽背景'):
    z=next(r for r in gaps if r['family']==name); trv=f(z,'train_shape_nrmse'); vv=f(z,'validation_shape_nrmse'); dv=f(z,'diagnostic_shape_nrmse')
    add(f'{name}：训练 `{trv:.5f}`、验证 `{vv:.5f}`、诊断 `{dv:.5f}`（已核实，同训练噪声口径）；验证高于训练 `{100*(vv/trv-1):.1f}%`（推算）。这些族在新谱上的退化值得单独处理，不能用“每步重新抽噪声”推断不会过拟合固定谱库。')
add('阶梯谱训练/验证约 `0.241/0.231`、零件组合约 `0.172/0.169`（已核实的仿真误差）都偏高，且验证未高于训练；更符合拟合偏差、表示限制或读数信息不足的候选解释（推测），不支持把这些族的问题叫过拟合。仍需学习曲线/结构消融才能把欠拟合与信息限制分开。')
add('逐族形状差如下：训练/验证为**已核实的仿真输出**，“差”为**推算**。完整导数项、诊断模式和样本数见 `train_validation_gap.csv`。')
add('| 谱族 | 训练 | 验证 | 验证−训练 |\n|---|---:|---:|---:|')
for r in gaps:add(f'| {r["family"]} | {f(r,"train_shape_nrmse"):.5f} | {f(r,"validation_shape_nrmse"):.5f} | {f(r,"validation_shape_nrmse")-f(r,"train_shape_nrmse"):+.5f} |')
add('## 4. 现在的融合偏向谁')
add('门控内部的族后验负责混合 A 的先验；它与 A/B 最终权重是不同层次。最终融合输入是 `8` 个误差/差异/粗糙度等特征，加多簇后验合回族级的 `24` 个对数后验，形成 `32→32(tanh)→2 logits→softmax` 小网（已核实）。`w_A` 是门控支权重，`w_B=1−w_A`；本表按诊断集平均权重是否超过 `0.5` 描述偏向（推算的汇总规则，不是代码中额外设的硬阈值）。')
add(f'**推算**：平均偏 A `{len(pa)}` 族：{names(pa)}。平均偏 B `{len(pb)}` 族：{names(pb)}。')
add('两个需要注意的交叉（已核实的仿真统计）：窄线族 A 均值更好，但平均 A 权重只有约 `0.356`；激光线+宽背景族 B 均值更好，平均 A 权重仍约 `0.511`，融合却比任一单支更好。白光 LED 也是 B 均值更好却平均偏 A，且融合退化。这说明“单支优势边界”和“现有软融合权重边界”并不重合。')
add('下表权重、样本比例为**已核实的仿真输出统计**；A 胜比例与融合实际比较对象 B 一致化相比较。软权重不是“选对分支的概率”，即便某支平均更差，少量混合仍可能有利。')
add('| 谱族 | 平均 w_A | w_A>0.5 占比 | A 误差<B一致化 占比 |\n|---|---:|---:|---:|')
for r in rows:add(f'| {r["family"]} | {f(r,"wA_mean"):.3f} | {100*f(r,"A_weight_gt_half"):.1f}% | {100*f(r,"A_win_vs_Bc_fraction"):.1f}% |')
add('`fusion_validation_holdout.csv` 另保存原融合验证集内部留出部分的 A、B 一致化和权重，供核对。该留出部分曾参与早停，且按读数随机划分，同一真谱的不同噪声抽样可落入两边；误差协方差和 κ 又用过整份验证数据，因此它不是干净泛化评估。本报告的主结论使用单独的 VSEED+13 诊断谱。q 只改变最后谱的平移，不重新改变 A/B 权重。')
rb=next(r for r in rows if r['family']=='窄带通')
add(f'**已核实的阶段分解例子**：窄带通族 B 原始 `{f(rb,"B_raw"):.6f}` → B 一致化 `{f(rb,"B_consistent"):.6f}` → q 前融合 `{f(rb,"fusion_pre_q"):.6f}` → q 后 `{f(rb,"final"):.6f}`。其 q 阶段相对变化为 `{100*(f(rb,"final")/f(rb,"fusion_pre_q")-1):+.2f}%`（推算）。因此最终退化不能一概归到 B；重新设计 B 时必须保留 q 前后的独立统计。')
add('## 5. 按本项目数据重新设计 B：仅方案，未实现')
add('先冻结 A、SS、escale 的配置，以本轮 A 的优势族作为保护集；B 只承担与它互补的目标族。每个方案先单独替换 B，观察 B 单支变化，再按同一验证流程重拟一致化、融合及 q，分开报告 B、A、q 前/后融合，避免把收益错归给 B。A 的输出仍依赖 B，需另报“旧 B 提供 A 辅助量”的冻结辅助量消融，判断收益来自 B 谱形还是间接改变 A。')
add(f'**推算的共同保护集**：A 的经验区间稳定优势族为 {names(sigA)}。窄线、阶梯和透射谱在本轮均值上由 A 领先，但差值区间跨零，先作为暂留 A 的边界族；不把它们当成已经定死的永久分工。以下三个方案都保留这组 A 胜区，把 B 的主要训练预算放在其余目标族。')
add('### 方案一：按响应矩阵设计的多尺度谱形解码器')
add('把读数在 R 的左奇异向量上投影，按模拟噪声尺度归一，同时保留完整残差信息；用小型全连接编码器输出宽背景基系数、局部峰基系数及谱域一维残差。波长头用非负约束，不把相邻偏压强行当成具有平移不变性的图像。基函数和可辨识子空间从训练谱/R 定，不用诊断真谱挑基数。')
add('预期主攻激光线+宽背景、零件组合、4–6 个峰，以及宽峰/不对称峰和斜坡+单峰：用宽背景头与局部峰头拆开目前一个全连接输出头承担的任务。其中 4–6 峰的现有 A/B 差距不稳定，属于待争取的 B 目标族。A 的稳定胜区及暂留边界族按上述保护集处理，尤其不把荧光多线、窄线和阶梯一并要求 B 解决。')
add('收益与成本均为**推测**：目标族几何平均误差希望相对旧 B 下降 `5–15%`；只是验证目标，可能无收益。B 训练计算量约旧 B 的 `0.7–1.5×`，按原日志 `151 s` 推算约 `2–4 min`，另需准备基函数与验证流水线，不能视为实测时长。')
add('验证只用验证谱：将 VSEED=31438 按真谱编号分开发段/检查段，所有噪声抽样跟随真谱；比较旧 B、只改输入、只改输出、多尺度完整版。冻结检查段，预先登记目标族与 A 保护族，报告目标族收益、A 保护族的融合退化和逐谱互补比例，不按总平均吞掉族间变化。')
add('### 方案二：把器件读数约束纳入 B 的训练')
add('保留一个谱形头，单设正的亮度/整体增益辅助量；损失加入带既定噪声协方差的读数残差，暗电流漂移作为有界干扰量边缘化或联合估计。按噪声可辨识方向约束残差，不要求把带噪 y 精确拟合。尤其要避免全局增益与光功率不可分辨时，把任一辅助量称为真实绝对功率。')
add('预期主攻激光线+宽背景、零件组合、斜坡+单峰和宽带通，减小亮度估计误差传到 A/escale；本轮 A 的稳定胜区与暂留边界族继续留给 A。尤其对激光线+宽背景，B 形状已经优于 A，但整体误差仍大且会影响 A 的尺度，因此形状与亮度辅助量应分头评价。物理残差的作用范围限于读数可辨识方向。')
add('收益与成本均为**推测**：目标族形状误差希望下降 `0–10%`，主要可能收益是缩放鲁棒性；训练约旧 B 的 `1.2–2.5×`，按日志约 `3–6 min`，加入噪声边缘化后可能更慢。')
add('仅用验证集比较形状损失、加普通读数残差、加噪声加权/干扰量三个版本；同一真谱重复测量，覆盖训练功率范围。先检查是否仅降低读数残差而形状变差，再评估 escale 和最终融合；预先固定损失权重候选，按谱分组早停，不接触测试。')
add('### 方案三：以 A 的可靠优势为边界，重设 B 的目标与采样')
add('保持 cnet0 作为结构对照，用训练谱交叉拟合 A 的误差与不确定性；按训练数据中 A 不可靠且 B 有可学结构的族/样本提高 B 权重。通过训练内分折得到权重。针对固定谱库的局部泛化差距，周期刷新目标族的训练谱形池，而不只重新抽噪声；先重点刷新激光线位置/宽度/背景参数和零件组合，保留原有功率/噪声模型。导数项改为按谱粗糙度/族标度归一，避免一个全库 r 强迫所有谱形用同一种频率权衡。谱形刷新、加权和导数项分别消融。')
add('预期主攻激光线+宽背景的泛化差距、零件组合的高训练误差，并保持宽峰、带通、锐边缘等已有 B 胜区。荧光多线、窄线的泛化问题主要交给 A 的现有路径，其余 A 保护族同上。这里的分工通过采样预算和损失目标体现。')
add('收益与成本均为**推测**：目标族误差希望下降 `5–15%`，最终融合希望下降 `1–5%`；若目标族 B 变好但融合不变，方案不算整体成功。含周期性谱池刷新的 B 训练约旧 B 的 `1.2–2.0×`，按日志约 `3–5 min`；训练内 A 交叉拟合的准备成本可能超过 `30 min`，需先分段估时再决定运行方式。本轮没有执行任何这些方案。')
add('仅用验证谱做等预算消融：原损失/均衡族采样/按 A 可靠性加权/重标导数项；同样按真谱分组，报告 B 的目标族收益、A 的保护族保留情况、最终融合、A 权重和逐谱胜负。不要用已看过的诊断集反复挑具体超参数后再声称其为独立验证。')
add('建议优先用方案三区分“固定谱库泛化问题”和“损失权衡问题”，再用方案一检查表示能力；方案二独立检验物理约束对亮度/读数一致性的帮助（推测的执行优先级）。上述训练时间只指 B，完整融合与 q 的验证流水线还需另计，成本可参考本轮实测总时长。')
add('## 6. 与真实器件相比可能乐观在哪里')
add('已有白噪声、整体增益漂移和暗电流漂移，不是无噪声模拟；但训练和评估都使用同一固定名义 R，默认没有额外工艺/器件间偏差，也没有实测确认的偏压扫描历史、非平稳漂移、温度依赖、相关低频噪声或模型失配。固定噪声模型由既往作者实验标定，未在 Terry 自己的 OPD 上验证。Uddin 是无机器件，不能直接把结论外推为 OPD 实证。')
add('原响应矩阵以 `10 nm` 采样再线性插到 `4 nm`（已核实：源码/交接），窄谱结果大量依赖生成器先验，不能据此宣称实测光谱分辨率。训练、验证、诊断共用相同的合成谱族生成机制；CIE LED 增强仍源于固定实谱库，不是独立光源实测验证。形状 NRMSE 又去掉了绝对幅度，必须另查缩放和功率误差。')
add('VSEED+13 在旧 SS 诊断中已经出现，本轮虽未拿它重拟模型，仍不能当全项目从未看过的盲集。bootstrap 只表示这批模拟谱的统计波动，不含不同器件、R 标定和噪声模型不确定性。')
add('## 7. 文件、复现与原文件保护')
add('本轮新增 `A18/diag/B1_诊断_20261104/` 下的诊断脚本、日志、数值表、校验记录及验证专用 q 权重；另新增本交接文件。原代码、结果、文档没有增删改。最终核验见 `verification.json`。')
add('关键文件：`diagnose_b1.py`（受限读写及诊断执行），`run_validated.log`，`architecture.json`（含源代码/B权重 SHA256），`family_metrics.csv`，`per_reading_metrics.csv`，`train_validation_gap.csv/.json`，`fusion_validation_holdout.csv`，`summary.json`，`q_selection.json`，`q_validation_only.pt`，`overlap.json`，`runtime_initialization_changes.json`，`strict_validation_adapter.json`，`existing_files_before.json`，`existing_file_changes.json`，`write_report.py`，`verify_b1.py`，`verification.json`。`attempt1_*`、`attempt2_*`、`run.log` 与 `run_strict.log` 仅保留被排除的未完成尝试，不能用于结论；`runtime/` 只容纳库临时文件。')
add('复现命令（从 `06_重建算法` 运行）：')
add('```powershell\npy -I -X utf8 A18\\diag\\B1_诊断_20261104\\diagnose_b1.py\n```')
add('当前目录已有完成结果，脚本会拒绝覆盖。复现时应为新一轮结果配置独立输出位置，保留已有结果。本轮有效计算在半小时内完成；完整测试指标段原估时超过半小时且属于禁用测试流程，本轮没有运行。')
add('按本轮“新文件只限诊断目录与交接文件”的限制，没有另写项目根目录 `memory` 会话记录；没有提供 Hook 绑定上下文，也不声称完成会话记录核验。')
HANDOFF.write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
print(HANDOFF)
print(json.dumps(dict(B_better=len(betterB),A_better=len(betterA),preferA=len(pa),train_val_gap=gap),ensure_ascii=False))
