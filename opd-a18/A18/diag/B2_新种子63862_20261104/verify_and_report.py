"""Independent raw-array verification, report and append-only B2 handoff."""
from pathlib import Path
import sys, csv, json, hashlib
from datetime import datetime
sys.dont_write_bytecode=True
import numpy as np
OUT=Path(__file__).resolve().parent
WORK=OUT.parents[2]
OLD=OUT.parent/'B2_第六类_20261104'
HANDOFF=WORK/'A18/HANDOFF_A18-B2.md'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(name,obj):
    with (OUT/name).open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
def main():
    assert not (OUT/'verification.json').exists(),'Do not overwrite verification.'
    summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    freeze=json.loads((OUT/'model_freeze.json').read_text(encoding='utf-8'))
    inputs=json.loads((OUT/'input_inventory_before.json').read_text(encoding='utf-8'))
    assert summary['seed']==63862 and summary['diagnostic_generated_once']
    assert summary['no_network_training'] and summary['no_parameter_choice_on_63862']
    assert summary['unknown_candidate_families']==6 and summary['K']==[64,256]
    assert (OUT/'diagnostic_spectra.npz').stat().st_mtime_ns>(OUT/'model_freeze.json').stat().st_mtime_ns
    assert HANDOFF.read_bytes().startswith((OUT/'HANDOFF_before_execution.md').read_bytes())
    for rel,h in freeze['files'].items():assert sha(WORK/rel)==h,rel
    for rel,x in inputs['old_B2'].items():assert sha(WORK/rel)==x['sha256'],rel
    for rel,x in inputs['source'].items():assert sha(WORK/rel)==x['sha256'],rel
    assert sha(WORK/inputs['B']['path'])==inputs['B']['sha256']
    repro=rows(OUT/'reproduction_31451.csv')
    assert len(repro)==30 and all(abs(float(r['difference']))<1e-6 for r in repro)
    prior=rows(OUT/'frozen_prior_manifest.csv')
    assert len(prior)==12 and {int(p['requested_K']) for p in prior}=={64,256}
    for p in prior:
        assert sha(OUT/p['file'])==p['sha256']
        with np.load(OUT/p['file']) as q:
            assert len(q['mu'])==int(p['actual_K'])
            assert 1<=len(q['mu'])<=int(p['requested_K']) and np.isclose(q['weight'].sum(),1)
    frows=rows(OUT/'family_metrics.csv'); samples=rows(OUT/'per_reading_metrics.csv')
    assert len(frows)==6 and len(samples)==720
    families=[r['family'] for r in frows];methods=[k for k in frows[0] if k not in ('family','role')]
    assert len(methods)==9
    raw=np.load(OUT/'all_errors.npz')
    pred=np.load(OUT/'predictions.npz')
    spectra=np.load(OUT/'diagnostic_spectra.npz')
    with np.load(OLD/'allowed_data.npz') as archive:band=archive['BM']
    def recompute_nrmse(x,s):
        x=np.maximum(x,0);x=x/np.maximum(x.max(axis=1,keepdims=True),1e-30)
        t=s[:,band];t=t/t.max(axis=1,keepdims=True)
        return np.sqrt(np.mean((x-t)**2,axis=1))/(t.max(axis=1)-t.min(axis=1))
    for j,k in enumerate(families):
        ss=[r for r in samples if r['family']==k]
        assert len(ss)==120 and len({(r['spectrum'],r['draw']) for r in ss})==120
        for m in methods:
            e=raw[f'{j}__{m}'];assert e.shape==(4,30) and np.all(np.isfinite(e)) and np.all(e>0)
            assert abs(float(e.mean())-float(frows[j][m]))<1e-12,(k,m)
            for row in ss:assert abs(float(row[m])-e[int(row['draw']),int(row['spectrum'])])<1e-12
    for K in (64,256):
        w=pred[f'unknown_K{K}_family_weights']
        assert w.shape==(720,6) and np.allclose(w.sum(1),1)
        assert pred[f'unknown_K{K}'].shape==(720,76)
    prediction_checks=0
    for j,k in enumerate(families):
        truth=np.tile(spectra[k],(4,1))
        for m in ('oracle_selected','oracle_K64','oracle_K256','unknown_oracle_K64','unknown_oracle_K256'):
            x=pred[f'{m}_{j}'] if not m.startswith('unknown') else pred[f'unknown_K{m.split("K")[-1]}'][j*120:(j+1)*120]
            recomputed=recompute_nrmse(x,truth).reshape(4,30)
            assert np.allclose(recomputed,raw[f'{j}__{m}'],atol=1e-12,rtol=0),(k,m)
            prediction_checks+=1
    stats=rows(OUT/'paired_comparisons.csv')
    oldstats=rows(OUT/'paired_31451_reanalysis.csv')
    oldraw={k:{m:np.empty((4,30)) for m in ('B','A','oracle','specialist_512x4','specialist_768x6')} for k in families}
    for r in rows(OLD/'per_reading_complete_resume_v2.csv'):
        for m in oldraw[r['family']]:oldraw[r['family']][m][int(r['draw']),int(r['spectrum'])]=float(r[m])
    checks=0
    # Independent implementation: one spectrum-level vector, same fixed RNG draws.
    for old_flag, rs in ((False,stats),(True,oldstats)):
        for r in rs:
            k=r['family'];j=families.index(k);m=r['comparison'].split('/')[0]
            a=oldraw[k][m] if old_flag else raw[f'{j}__{m}']
            b=oldraw[k]['B'] if old_flag else raw[f'{j}__B']
            v=(np.log(a)-np.log(b)).mean(axis=0)
            boot=[];g=np.random.default_rng(1901122)
            for _ in range(20):boot.append(v[g.integers(30,size=(10000,30))].mean(axis=1))
            boot=np.concatenate(boot)
            q=np.quantile(boot,[.025,.975,.00625,.99375])
            assert abs(float(r['log_ratio'])-v.mean())<1e-12
            assert abs(float(r['geometric_ratio'])-np.exp(v.mean()))<1e-12
            assert np.allclose([float(r['log_ci95_low']),float(r['log_ci95_high'])],q[:2],atol=1e-12,rtol=0)
            if frows[j]['role']=='target':
                assert np.allclose([float(r['log_ci_bonf_low']),float(r['log_ci_bonf_high'])],q[2:],atol=1e-12,rtol=0)
                assert (r['stable_5pct_improvement']=='True')==bool(q[3]<0 and np.exp(v.mean())<=.95)
            else:assert r['log_ci_bonf_low']==''
            checks+=1
    overlap=json.loads((OUT/'overlap.json').read_text(encoding='utf-8'))
    assert not any(overlap['overlaps'].values())
    extended=json.loads((OUT/'extended_overlap.json').read_text(encoding='utf-8'))
    assert extended['status']=='PASS' and not any(extended['overlaps'].values())
    assert not extended['new_seed_regenerated']
    stream=json.loads((OUT/'streaming_numerical_check.json').read_text(encoding='utf-8'))
    assert stream['max_abs_difference']<1e-10
    noise=json.loads((OUT/'noise_estimation.json').read_text(encoding='utf-8'))
    assert not noise['inference_uses_true_P0'] and not noise['inference_uses_true_DEVCOV']
    with np.load(OUT/'frozen_estimated_noise.npz') as q:assert np.linalg.eigvalsh(q['cov']).min()>0

    # Source numerical anchors: use the originally frozen files, not a re-fit.
    report='# B2 新种子 63862 评估\n\nAI：Codex。全部为固定响应矩阵与误差模型下的仿真，不是器件实测。\n\n'
    report+='先重建、复现、冻结，再生成一次 63862；未在新结果上重训、挑 K、挑先验或改路由。本轮不参与 B1／B3 决策。\n\n'
    report+='为保持原 26 族噪声流的推进顺序，一次生成全部 26 族（780 条谱）与读数，只对预注册的六族计算模型误差；没有再生成新种子数据。\n\n'
    report+='## 复现与冻结\n\n'
    report+=f'31451 六族、五种方法共 30 项均值 NRMSE 复现全部通过，最大差 {summary["maximum_reproduction_difference"]:.3g}，阈值 1e-6。六个旧最终 prior 的均值、协方差、权重与旧文件最大差均小于 1e-12。完整 A 数值状态、两档 prior、估计噪声及执行代码已在新谱生成前记录 SHA256 并追加到 HANDOFF。\n\n'
    report+='首次重建预执行因白名单缺历史高斯拟合种子 11 停止，未生成 63862；随后只补入该历史训练种子。原尝试代码保留为 `reevaluate_attempt1_before_fix.py`，没有用诊断成绩修改设定。\n\n'
    report+='## 六族主表\n\n所有数值为同一真谱、同一四次带噪读数上的平均 NRMSE。K 指请求的 spectral-kmeans 总分量数；小簇剔除导致实际 K 更少。old-selected 为原最终选择，包含同分布五类的训练子类 K=320，只作单独的旧配置复评。\n\n'
    report+='| 谱族 | B | A | 知族 K64 | 知族 K256 | 不知族 K64 | 不知族 K256 | 专用 512×4 | 专用 768×6 | old-selected |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in frows:
        keys=['B','A','oracle_K64','oracle_K256','unknown_oracle_K64','unknown_oracle_K256','specialist_512x4','specialist_768x6','oracle_selected']
        report+='| '+r['family']+' | '+' | '.join(f'{float(r[key]):.6f}' for key in keys)+' |\n'
    report+='\n## 成对区间与判断\n\n统计量先在每条谱的四次读数上平均 log(方法误差/B误差)，再对 30 条谱平均；按谱重抽 200000 次。四目标族对每项比较分别做 Bonferroni，校正区间为 98.75%；同时提供未校正 95% 区间。这里没有对全部模型、K 和比较再统一校正。改善至少 5% 且校正 log 区间上限小于零才标为稳定改善；跨零不能证明等效。对照族只报探索性未校正区间。\n\n'
    report+='| 谱族 | 方法/B | 几何误差比 | 未校正 95% 比值区间 | 四族校正比值区间 | 稳定改善 ≥5% |\n|---|---|---:|---|---|---|\n'
    for r in stats:
        if r['family'] in families[:4]:
            report+=f'| {r["family"]} | {r["comparison"]} | {float(r["geometric_ratio"]):.4f} | [{float(r["ratio_ci95_low"]):.4f}, {float(r["ratio_ci95_high"]):.4f}] | [{float(r["ratio_ci_bonf_low"]):.4f}, {float(r["ratio_ci_bonf_high"]):.4f}] | {"是" if r["stable_5pct_improvement"]=="True" else "否"} |\n'
    report+='\n## 四族结论\n\n'
    for k in families[:4]:
        rs=[r for r in stats if r['family']==k]
        stable=[r['comparison'].split('/')[0] for r in rs if r['stable_5pct_improvement']=='True']
        uncertain=[r['comparison'].split('/')[0] for r in rs if float(r['log_ci_bonf_low'])<=0<=float(r['log_ci_bonf_high'])]
        report+=f'### {k}\n\n'
        report+='校正后达到至少 5% 稳定改善的方法：'+('、'.join(stable) if stable else '无')+'。'
        if uncertain:report+='校正区间跨零：'+'、'.join(uncertain)+'。'
        report+='这些结果只评价冻结估计器在该生成族、功率与噪声条件下的表现，不能宣布严格信息极限。\n\n'
    report+='## 对照族与不知族边界\n\n'
    report+='不知族估计器只覆盖六个候选族，族先验各 1/6，推理不输入真族。它和知族两档使用同一组 prior，但噪声协方差由旧 31438 校准谱和重复读数估计，不能把二者差值全解释为判族代价。校准利用旧谱的已知形状，未利用 63862 真谱、真实亮度、真实功率或真实噪声协方差。也未验证 26 族任意输入或新功率分布。\n\n'
    report+=f'估计的等效参考功率 {noise["P_est_W"]*1e6:.6f} µW，仅用于解释所估噪声量级，推理直接使用冻结的估计协方差。\n\n'
    for k in families[4:]:
        rr=next(r for r in frows if r['family']==k)
        report+=k+'：知族两档相对 B 的均值误差下降为 '+f'{100*(1-float(rr["oracle_K64"])/float(rr["B"])):.2f}%／{100*(1-float(rr["oracle_K256"])/float(rr["B"])):.2f}%'+'。这说明更匹配的族先验在强族上同样可能受益，知族收益不能专门归因于第六类网络欠拟合。两个 specialist 未训练这两个对照族，其退化只能说明专用化代价。\n\n'
    report+='## 31451 同口径重算与旧结论修订\n\n'
    report+='`paired_31451_reanalysis.csv` 从旧逐读数文件重新计算同一 log 比值统计与四族校正，未重新选择模型。31451 同时参与过 A 的 SS 调参、目标族挑选，且旧 oracle 候选先后修改过两次；旧同分布五类诊断误差历经约 0.1094→0.0946→0.0802。旧结果仅是开发诊断。\n\n'
    report+='撤回“零件组合最接近信息极限”的说法：旧 K 截断，增加混合分量可继续改变误差。不能从两种 specialist 的单初始化实验推出网络容量无效；只可按区间陈述本轮专用训练的收益与不确定性。训练库固定、谱重复曝光和先验表达均是候选解释。损失比例 r 的准确值为 44.404842417764215。\n\n'
    report+='## 文件与核验\n\n'
    report+='`summary.json`：数值概要与边界；`run.log`：阶段日志；`family_metrics.csv`：主表；`per_reading_metrics.csv`、`all_errors.npz`：逐读数误差；`paired_comparisons.csv`：新种子成对区间；`paired_31451_reanalysis.csv`：旧种子同口径重算；`reproduction_31451.csv`：复现门槛；`model_freeze.json`、`frozen_prior_manifest.csv`：冻结文件及 SHA256；`noise_estimation.json`：噪声校准方法和数值。\n\n'
    report+=f'独立从原始数组重算 {checks} 项 bootstrap 比较，并核对主表、720 条逐读数记录、prior 权重、冻结哈希及 {len(inputs["old_B2"])} 个旧 B2 文件哈希。旧 B2、主算法和原权重未改；HANDOFF 只追加。\n\n'
    report+='项目会话记录未写入：实际项目根目录的 memory 在当前可写工作区之外；没有用临时 cwd 替代，也没有声称通过 Hook 记录核验。\n'
    with (OUT/'report_63862.md').open('x',encoding='utf-8') as f:f.write(report)
    # Final HANDOFF summary preserves all historical text and the freeze append.
    append='\n\n### 63862 完成记录（AI：Codex）\n\n'
    append+='已完成一次冻结评估：六族 × 每族 30 条真谱 × 4 次真实模式读数，共 720 条。31451 复现门槛先通过，未重训 B／specialist，未用新结果选择 K 或先验；不用于 B1／B3 决策。\n\n'
    append+='主表（平均 NRMSE；知族／不知族各固定报告 K64 和 K256）：\n\n'
    append+='| 谱族 | B | A | 知族64 | 知族256 | 不知族64 | 不知族256 | 专用512×4 | 专用768×6 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in frows:
        append+='| '+r['family']+' | '+' | '.join(f'{float(r[key]):.6f}' for key in ['B','A','oracle_K64','oracle_K256','unknown_oracle_K64','unknown_oracle_K256','specialist_512x4','specialist_768x6'])+' |\n'
    append+='\n按谱配对 log 误差比，200000 次 bootstrap；每项比较分别对四目标族 Bonferroni，并同时报告未校正区间。没有对全部模型／K 再统一校正。达到至少 5% 且校正区间上限小于零的方法：\n\n'
    for k in families[:4]:
        stable=[r['comparison'].split('/')[0] for r in stats if r['family']==k and r['stable_5pct_improvement']=='True']
        append+=k+'：'+('、'.join(stable) if stable else '无')+'。\n\n'
    append+='旧开发诊断曾两次修改 oracle 候选，且复用了 A 的 SS 调参谱和挑目标族；31451 的同口径重算不作独立证据。撤回“零件组合最像信息极限”与“扩大网络已证明无效”等过强表述。更丰富 prior 不是严格误差下界；专用网络仍有单初始化、固定训练库曝光等混杂。r 更正为 44.404842417764215。\n\n'
    append+='不知族只覆盖六族，使用旧校准数据估计噪声；它与知族的差别同时包括族信息和噪声估计，不能据差值单独量化判族代价。对照族也受益于知族先验，不能把 oracle 优势全部解释为第六类容量不足。全部为仿真。\n\n'
    append+='完整结果与区间：`diag/B2_新种子63862_20261104/report_63862.md`；数值：`family_metrics.csv`、`per_reading_metrics.csv`、`paired_comparisons.csv`；旧种子重算：`paired_31451_reanalysis.csv`；概要／日志／核验：`summary.json`、`run.log`、`verification.json`。\n\n'
    append+='只新增本轮目录文件并追加本 HANDOFF；旧 B2 的 66 个文件、源码与原权重哈希未变。首次执行因历史 seed 11 白名单不足停在重建阶段，未接触新种子，原尝试留存。项目根目录 memory 超出写入权限，未写会话记录或声称 Hook 通过。\n'
    with HANDOFF.open('ab') as f:f.write(append.encode('utf-8'))
    assert HANDOFF.read_bytes().startswith((OUT/'HANDOFF_after_freeze.md').read_bytes())
    # UTF-8 roundtrip and Chinese content check for every generated Markdown.
    for p in (OUT/'report_63862.md',HANDOFF):
        text=p.read_text(encoding='utf-8');assert '\ufffd' not in text and '????' not in text
        assert sum('\u4e00'<=c<='\u9fff' for c in text)>100
    verification=dict(status='PASS',verified_at=datetime.now().astimezone().isoformat(),
        seed=63862,diagnostic_generation_count=1,families=6,spectra=180,readings=720,
        raw_bootstrap_comparisons_recomputed=checks,bootstrap_replicates=200000,
        nrmse_checks_from_saved_predictions=prediction_checks,
        old_B2_files_unchanged=len(inputs['old_B2']),frozen_files_unchanged=len(freeze['files']),
        original_handoff_bytes_preserved=len((OUT/'HANDOFF_before_execution.md').read_bytes()),
        handoff_sha256=sha(HANDOFF),report_sha256=sha(OUT/'report_63862.md'),
        no_network_training=True,old_31451_reproduction_passed=True,
        extended_historical_overlap_check=extended,
        all_prior_and_model_hashes_unchanged=True,unknown_family_universe=6,
        unknown_noise_calibration_only=True,utf8_verified=True)
    write('verification.json',verification)
    # Summary is the only new output finalized in place after independent checks.
    summary['status']='COMPLETE_VERIFIED';summary['verification_file']='verification.json'
    summary['verification_sha256']=sha(OUT/'verification.json')
    summary['report_file']='report_63862.md'
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (OUT/'run.log').open('a',encoding='utf-8') as f:f.write('INDEPENDENT VERIFICATION PASS; report and append-only HANDOFF saved.\n')
    print(json.dumps(verification,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
