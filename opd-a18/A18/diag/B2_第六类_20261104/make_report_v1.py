"""Write new A18-B2 handoff and plot; preserve every pre-existing artifact."""
from pathlib import Path
import os,sys,json,csv,time,hashlib
from datetime import datetime,timezone
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;WORK=OUT.parents[2]
REPORT=WORK/'A18/HANDOFF_A18-B2.md'
assert not REPORT.exists(),'Do not overwrite an existing handoff.'
os.environ['MPLCONFIGDIR']=str(OUT/'plot_cache_report_v1')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10})

def rows(name):
    with (OUT/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
r=rows('family_metrics_complete_resume_v2.csv')
o=rows('oracle_metrics_complete_resume_v2.csv')
stats=rows('paired_comparisons_resume_v2.csv')
models=json.loads((OUT/'network_summary.json').read_text(encoding='utf-8'))['models']
fig,axs=plt.subplots(1,2,figsize=(13,5),gridspec_kw={'width_ratios':[1.5,1]})
x=np.arange(4);width=.15
for i,(key,label,color) in enumerate([('B','原 B','#4d78a8'),('A','现行 A*','#a6a6a6'),('oracle','真族 oracle','#44a17e'),
                                   ('specialist_512x4','专用 512×4','#efaf4b'),('specialist_768x6','专用 768×6','#cd6b64')]):
    axs[0].bar(x+(i-2)*width,[float(z[key]) for z in r[:4]],width,label=label,color=color)
axs[0].set_xticks(x,[z['family'] for z in r[:4]])
axs[0].set_ylabel('形状 NRMSE（越低越好）');axs[0].set_ylim(0,.29)
axs[0].legend(fontsize=9,ncols=2,frameon=False);axs[0].grid(axis='y',alpha=.2);axs[0].set_axisbelow(True)
for m,color in zip(models,['#efaf4b','#cd6b64']):
    c=rows(m['name']+'_curve.csv')
    axs[1].plot([int(v['step']) for v in c],[float(v['validation_equal_family']) for v in c],label=m['name'],color=color)
axs[1].set_xlabel('训练步数');axs[1].set_ylabel('验证集四族等权平均 NRMSE');axs[1].grid(alpha=.2)
axs[1].legend(fontsize=8,frameon=False)
fig.suptitle('A18-B2：仿真诊断；oracle 为近似参考，不能证明严格信息下界',fontsize=12)
fig.text(.04,.012,'* VSEED+13 曾用于 A 的 SS 调参；阶梯、零件等 SS 覆盖族的 A 数字偏乐观。B 为原始 cnet0；每谱四次噪声读数。',fontsize=9)
fig.tight_layout(rect=[0,.05,1,.94])
plot=OUT/'B2_comparison_resume_v1.png';assert not plot.exists();fig.savefig(plot,dpi=180);plt.close(fig)

def table(data):
    s='| 谱族 | B | A* | oracle | 严格零空间能量 % | 专用 512×4 | 专用 768×6 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for z in data:
        s+=f"| {z['family']} | {float(z['B']):.5f} | {float(z['A']):.5f} | {float(z['oracle']):.5f} | {float(z['null_energy_percent']):.5f} | {float(z['specialist_512x4']):.5f} | {float(z['specialist_768x6']):.5f} |\n"
    return s
def stat(family,comp):return next(z for z in stats if z['family']==family and z['comparison']==comp+' minus B')
def interval(z):return f"[{float(z['ci_low']):+.5f}, {float(z['ci_high']):+.5f}]"

now=datetime.now(timezone.utc)
begin=datetime(2026,11,4,2,26,59,tzinfo=timezone.utc)
resume=datetime(2026,11,4,3,12,12,tzinfo=timezone.utc)
interrupt=datetime(2026,11,4,2,37,32,tzinfo=timezone.utc)
elapsed=(now-begin).total_seconds();pause=(resume-interrupt).total_seconds();active=elapsed-pause
timing=dict(start_utc=begin.isoformat(),report_created_utc=now.isoformat(),elapsed_seconds=elapsed,
    approximate_interruption_seconds=pause,approximate_active_seconds=active,
    interruption_boundary_source='last pre-interruption log timestamp 09:37:32; user said interrupted at 09:37',
    network_training_seconds=sum(m['seconds'] for m in models),evidence='clock timestamps 已核实; elapsed/active 推算, interruption is approximate')
with (OUT/'timing_report_v1.json').open('x',encoding='utf-8') as f:json.dump(timing,f,ensure_ascii=False,indent=2)

text=f'''---
task_id: A18-B2
author: AI（Codex）
date: 2026-11-04
status: 完成；开发诊断，未改主算法
---

# A18-B2：第六类是信息不够，还是 B 能力不够？

**结论：不能把第六类整体认定为“换什么网络都没用”，也没有证据证明单纯扩大 B 就能解决。** 阶梯谱有小幅专用训练收益；零件组合最接近本轮近似 oracle；随机光滑、同分布五类存在更合适先验可利用的空间，但两种专用网络都未兑现。属于混合情况，暂不宣布任何一族达到严格信息极限。

全部为**仿真**：公开实测 R 加项目既定误差模型，不是 Terry 器件实测。

证据等级：**已核实**＝实际运行输出、回读文件或核实源代码；**推算**＝据输出计算的比例、投影、区间及比较；**推测**＝待验证的原因或建议。各表的表前说明统一覆盖表内全部数字。

## 1. 主结果

下表 B、A、oracle、两种对照网络误差均为**已核实的仿真 NRMSE**；零空间能量百分数为**推算**。B 与 B1 在相同读数上复核，最大差约 1.1e-7（已核实，浮点差异）。A 直接引用 B1 的同读数逐条结果，未重训或另选 A。

{table(r[:4])}

**副导师审查发现，已核实于代码：诊断集 VSEED+13=31451 是 A 的 SS 调参集。A 在 SS 覆盖族（阶梯、零件组合等）上的数字偏乐观。** `算法对照/a16_ss.py:6` 明说 U、α、各类型参数在 VSEED+13 选择；`A17/diag/ss_int.py:65` 与 `ss_cls.py:65` 明确构建该调参谱集。噪声随机流不同不能消除同一真谱已参与选择的偏差。本表因此只能作开发诊断，不能将 A 的这些数字当独立泛化成绩，也不能据其微弱优势决定归 A。A 的其他谱族还可能经软门控接受 SS 混合，不宜把它们整体宣布为完全无偏。没有修改 B1 原报告。

对照族同口径数字如下；这两族未进入专用网络训练，专用网络的误差只检验专用化损失，不能据此评价其一般能力。证据等级同上表。

{table(r[4:])}

## 2. 四族判断与建议

### 阶梯谱：优先归 B 改进候选，兼有噪声／先验限制

**推算**：同结构专用 B 相对原 B 改善 6.49%，更大专用 B 改善 7.30%；oracle 改善 14.16%。更大网络仍为 0.22343，oracle 为 0.20690（已核实仿真），说明本轮没有接近低误差重建。

**推算**：以谱为单位保留四次读数的成对 bootstrap，专用 512×4−B 的经验区间为 {interval(stat('阶梯谱','specialist_512x4'))}，专用 768×6−B 为 {interval(stat('阶梯谱','specialist_768x6'))}；都跨零。oracle−B 为 {interval(stat('阶梯谱','oracle'))}，不跨零。网络改善是均值趋势，尚不能称稳定显著。A 的微弱优势还有 SS 选择偏差。

**推测／建议**：保留 B 为主，下一轮验证阶梯的族权重、形状损失或专用输出头；不据本轮换成整套大网络，不因现行 A 数字而直接归 A。剩余大误差更符合噪声下难辨细节与先验近似共同限制的候选解释，尚不是严格信息极限。

### 零件组合：信息受限候选，当前保留 B

**推算**：oracle 仅比 B 低 5.71%，oracle−B 区间为 {interval(stat('零件组合','oracle'))}，跨零；两种专用网络反而平均差 1.33% 和 3.07%。现行 A 即使在偏乐观的开发谱集上仍更差。

**推测／建议**：四族中最支持“本模型和先验下的可辨识性是主要瓶颈”的候选。当前仍交原 B，不归现行 A，不优先追加网络参数。只能称接近本轮近似 oracle，不能称接近所有可能网络的最优误差。

### 随机光滑：有先验空间；扩大 B 未解决

**推算**：oracle 比 B 低 12.07%，oracle−B 区间 {interval(stat('随机光滑','oracle'))} 不跨零；同结构／更大专用网络却差 4.62%／18.84%。

**推测／建议**：不能说读数里完全没有可用信息。当前用原 B，后续开发优先归 A 的随机光滑先验／门控诊断，或 B 的损失与先验表达改进；**不是已证明现行 A 更好**。oracle 已知真族，现行解码器不知道，收益可能同时来自免去判族和先验拟合，不能全归结为网络容量。

### 同分布五类：先验分组有价值；容量不足未被证实

训练谱已有五种生成子类标签，仅在训练阶段按子类建混合先验；推理仍混合所有子类，未提供验证或诊断子类真标签。

**已核实仿真**：单高斯 oracle 为 0.20718；不分子类的混合先验为 0.09456；保留训练子类结构后为 0.08023。选择基于验证误差，未用诊断误差挑先验。该改进说明“粗糙高斯 oracle≈B，因此是信息极限”的推断不可靠。

**推算**：最终 oracle 比 B 低 14.92%，但 oracle−B 区间 {interval(stat('同分布五类','oracle'))} 跨零；两种专用网络差 3.21%／2.79%。**推测／建议**：当前仍用原 B，下一轮优先查 A 的子类先验与路由，或 B 的子类表达；不认定为信息极限，也不把增加宽度当有效解法。

## 3. oracle 具体算了什么

先验只使用同一训练库中本族谱。把谱除以波段内总和得到单位面积谱 x，拟合均值、协方差和高斯混合权重。推理输入只取带噪读数的方向 u，不提供真谱的峰值、总和、亮度或子类标签；“作弊”仅为知道顶层真族。

**已核实的模型口径**：在单位面积坐标下，z=g R_n x+ε；ε 的协方差是原 `AD.DEVCOV`，含暗电流整体漂移和白噪声，增益 g 沿用原对数正态分布并单独积分。每个高斯分量下先求条件高斯均值，再对未知增益和正径向幅度积分，最后按后验权重混合。径向积分包含 a^(n−1) 的方向密度雅可比。输入始终来自原 real 读数，未生成理想模式性能表。

**已核实的候选与数值设置**：基础 K 为 1、8、32、64，协方差收缩为 1e-4 或 1e-2；若验证集上 64 比 32 再改善超过 5%，才扩至 128、256。同分布五类另比较按训练子类分簇的混合先验。每簇至少 3 条训练谱，过小簇剔除后重新归一权重。最终实际分量数按四族顺序是 64、59、63、320，对照族各 128；完整选择与实际 K 见 `oracle_metrics_complete_resume_v2.csv`。

**已核实的数值检查**：增益／径向积分节点由 15／48 提到 25／80，逐族平均 NRMSE 变化的绝对值均小于 5e-8（最终所选六族；见原始完整精度列）。补跑验证同分布五类读数任意正缩放后预测最大差为 6.2e-13，证明推理没有从真谱幅度取信息。

**关键边界**：这是所拟合高斯混合模型下、单位面积平方损失的 MMSE，再经非负裁剪与峰值归一转换成项目 NRMSE。真实生成族并不严格高斯，有限训练库、混合近似、非高斯尾部、不同损失及裁剪都使它**不是严格下界**。它可以比 B 差；它比 B 好时只证明某个已知真族的估计器利用到了更多先验，不能保证一个自动判族网络能达到同样误差。靠近它不能证明“换什么网络都没用”。

## 4. 零空间与噪声下不可辨方向

**已核实／推算**：R 为 41×76；以最大奇异值的 1e-10 为数值秩阈值，秩为 31。它来自 31 个实测波长节点的插值，不能把 41 个偏压直接当成 41 个独立谱方向。按奇异值比 2% 截断，本次矩阵保留 6 个方向；本轮依原矩阵重算，不沿用旧交接的约数。

主表严格零空间比例定义为：诊断真谱投影到 R 零空间后的平方能量总和，除以真谱总平方能量。它很小，不能解释全部大误差；噪声会淹没接近零但非零的方向。

另按原噪声协方差白化 R，投影掉未知整体幅度方向；用本族训练谱在各奇异方向上的方差计算信号方差／噪声方差，小于 1 的方向记为弱可辨。下表为诊断谱**相对本族训练均值的单位面积谱变化能量**落入这些方向的比例，均为**推算**，分母与主表“总谱能量”不同。

| 谱族 | 噪声下弱可辨变化能量 % |
|---|---:|
'''
for z in r:text+=f"| {z['family']} | {float(z['noise_weak_variation_percent']):.2f} |\n"
text+='''
该比例是方向诊断，不是不可消除误差下界：先验相关性可以由强方向预测弱方向。宽带通对照的比例也较高，而 B 重建仍不错，正好说明不能只看零空间或弱方向比例就判定谱族无解。

## 5. 两个网络对照与泛化

**已核实的训练设定**：两者从头训练，只取原库的第六类四族，合计 21,000 条；其中同分布五类 12,000 条，其他每族 3,000 条。没有生成额外训练谱。与原 cnet0 一样为 30,000 步、batch 2,048、初始化种子 11、AdamW、最大 LR 1e-3、weight decay 1e-4、OneCycleLR、每步重新抽原 train 噪声和 0.5–2 P0 功率。保留全库算出的原导数损失比例 r=44.404861，避免把专用训练和改损失混成一个实验。

每 1,000 步在 VSEED=31438 的固定 real 读数上计算四族等权平均 NRMSE，保存最好的 checkpoint；步数预算固定，checkpoint 选择只看验证集。原 B 没有验证早停，因此此处同时改变了训练分配与 checkpoint 选择，不能把收益严格分拆成二者的贡献。

**已核实**：同结构 512×4 为 2,166,860 参数，选 step 29,000，训练耗时 113.35 s；更大 768×6 为 7,188,556 参数，选 step 25,000，耗时 223.38 s。单独测得的完整训练估时分别约 135 s、214 s（推算），都低于用户的单次 30 分钟限制；实际两次训练总计 336.73 s（推算）。仅一个初始化种子，不能断言所有训练配置都失败。

更大网络训练误差降低，却未改善诊断误差：例如随机光滑训练／验证／诊断 NRMSE 为 0.05947／0.07752／0.09712（已核实仿真）。这更像专用化后的泛化或先验表达问题，不能把高训练误差统一叫“网络太小”。训练—验证各族详见两个 `*_gap.csv`；这些 gap 使用 real 模式，不与 B1 的随机功率 train-mode gap 直接比较。

两个强族未参与专用训练；宽带通退化到约 0.20（已核实仿真），因此不能用专用 checkpoint 替换通用 B。

## 6. 数据边界与乐观性

**已核实**：训练主种子 31337，原库同分布额外训练种子 31344；验证 31438，诊断 31451；谱分别按原生成器的派生规则生成。全部谱族均使用 jitter=True。主逐族诊断使用原 B1 的噪声流 4321，每族 30 条谱、每谱 4 次带噪读数，固定参考功率 P0=10.5423395868 µW。验证每族 40 条，固定噪声流 54321。NRMSE 调用原 `ladder.band_nrmse`：重建／真谱分别峰值归一，RMSE 再除以归一真谱极差。

**已核实**：暗电流漂移标准差 1.5%、对数增益标准差 0.15、白噪声 0.3 nA。未启用额外 R 标定扰动。训练、验证、诊断按九位小数真谱逐行查重，相交均为 0。没有生成或读取测试谱／测试结果，没有使用 6161／4242 数据调参；也未为查重生成它们。因此不声称本轮完成了与所有历史测试数据的逐条查重。

可能比真实器件乐观之处：模拟读数和解码共用同一插值 R；原噪声模型未完整覆盖波长相关标定误差、偏压序列漂移、迟滞、温漂、长期老化和光源失配；oracle 知道真族，先验与诊断共享生成机制。4 nm 计算网格细于实测 R 的 10 nm 学习步长，不代表器件具备 4 nm 实测分辨率。另有前述 SS 调参集复用偏差。

成对区间按 30 条真谱重抽 5,000 次，每次保留四次读数，取经验 95% 区间（已核实设置；区间为推算）。未校正多族／多模型比较，不代表跨随机种子或跨器件置信度。

## 7. 中断恢复与文件清单

恢复时先检查完整性，没有重训网络，也没有重拟任何完整 prior。前四族的最新指标与 prior 已完整落盘；两个对照族较早一轮的完整 summary、prior 和逐条误差仍可用，选中配置与当前代码一致。仅同分布五类最新 prior 的逐条误差缺失，本次从已选 prior 补算，均值和中断前 CSV 一致。

旧 `oracle_summary.json` 与 `oracle_errors.npz` 中同分布五类仍是较早的 K=256 版本，**不可当最终结果**；旧 `oracle_metrics.csv` 仅有最新四族。保留这些旧文件不覆盖；正式结果以以下带 `complete_resume_v2` 的文件为准。

主要交付（相对本文件的路径）：

`diag/B2_第六类_20261104/family_metrics_complete_resume_v2.csv`：六族 B／A／oracle／零空间／两个网络的主表。

`diag/B2_第六类_20261104/oracle_metrics_complete_resume_v2.csv`：最终选择、实际分量数、数值收敛和逐项来源。

`diag/B2_第六类_20261104/nullspace_metrics_complete_resume_v2.csv`：严格零空间与噪声弱方向诊断。

`diag/B2_第六类_20261104/per_reading_complete_resume_v2.csv`、`paired_comparisons_resume_v2.csv`：逐读数误差与成对比较。

`diag/B2_第六类_20261104/oracle_errors_complete_resume_v2.npz`、`five_subtype_missing_predictions_resume_v2.npz`：补齐后的误差和缺失预测。

`diag/B2_第六类_20261104/B2_comparison_resume_v1.png`：四族结果和验证学习曲线。

`diag/B2_第六类_20261104/completion_inventory_resume_v2.json`、`resume_verification_v2.json`、`timing_report_v1.json`：复用／部分文件说明、恢复前后不改旧文件核验、计时。

原始训练交付保留：`run_networks.py`、`bootstrap_b2.py`、两个 `specialist_*.pt`（验证最佳）及 `*_last.pt`、`*_curve.csv`、`*_gap.csv`、`*_summary.json`。原始 oracle 实现在 `run_oracles.py`，断点恢复实现在 `complete_resume_v2.py`。

`complete_resume_v1.py` 为恢复检查的失败尝试：最初误把请求 K 当实际 K，遇到小簇被剔除时断言停止；没有覆盖旧文件，也没有使用其结果。v2 按实际 K 校验后通过；该失败尝试及其快照按用户要求保留。

从干净的新输出位置重新计算时可参考命令：

```powershell
py -I -X utf8 A18\diag\B2_第六类_20261104\run_networks.py
py -I -X utf8 A18\diag\B2_第六类_20261104\run_oracles.py
```

**不要在当前目录重新执行以上旧驱动**：它们会写原生成文件。本次恢复只使用新命名的 `complete_resume_v2.py`，其中所有既存文件写入均被审计拦截；它已运行完成，再次执行也会拒绝覆盖自己的结果。进一步工作应新命名输出。

没有修改任何主算法、原权重、B1 报告、原结果或已有交接文件。仅在本轮允许位置新增结果和本交接。遵守本轮更具体的写入限制，未另写项目 memory 目录；未声称完成 Hook 会话记忆核验。

## 8. 耗时

'''
text+=f'''报告生成时，首轮开始至今约 {elapsed/60:.1f} 分钟（推算），其中中断等待约 {pause/60:.1f} 分钟；实际参与处理约 {active/60:.1f} 分钟（推算，中断边界取末条旧日志，含阅读、开发、核验）。两次纯训练共 336.73 s，约 5.61 分钟（推算，原计时输出已核实）；恢复只补缺失后验，未重训。准确时间戳见 `timing_report_v1.json`；最终核验会另记交付时间。
'''
with REPORT.open('x',encoding='utf-8') as f:f.write(text)
print(f'Wrote {REPORT}; {len(text)} characters; elapsed={elapsed/60:.2f} min, active~{active/60:.2f} min',flush=True)
