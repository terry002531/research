"""AI (Codex): disclose the scratch assertion and verify the two-phase recovery."""
from pathlib import Path
import ast

OUT=Path(__file__).resolve().parent
assert not (OUT/'write_final_report.py').exists()
src=(OUT/'write_resume_report.py').read_text(encoding='utf-8')
src=src.replace('`full_model_freeze.json` 与 `full_model_post_evaluation.json` 内容相同，涵盖',
    '`full_model_freeze.json` 与恢复批次缓存后的 `full_model_post_evaluation.json` 内容相同（缓存变化单独留痕，见下一节），涵盖',1)
src=src.replace('恢复采用 resume_diagnose_b1.py；旧 diagnose_b1.py 保留为中断版本，不把旧命令误称为本次运行命令。',
    'q 完整重算采用 resume_diagnose_b1.py；其末尾缓存断言触发后，最终评估采用 evaluate_after_q.py。旧 diagnose_b1.py 保留为 12:18 中断版本，不把旧命令误称为最终运行命令。',1)
src=src.replace('完整状态与 15:16 登记逐项相等才放行。',
    '参数、先验及归一化与 15:16 登记逐项相等才放行；批次缓存单独严格审计。',1)
anchor="add('### 保守统计判据与逐族结果')"
insert="""add('**完整状态断言的中止和恢复**：resume_diagnose_b1.py 在 q 的26族数据、统计与去留决定全部保存后，因完整状态断言以退出码1结束；此时 22658 尚未生成。源码 algo5.GaussCal._solve 每次将当前批次的马氏距离平方写入 GF._m2；原完整指纹包含了这个工作缓存。本轮保留原错误日志，不把退出码改写为0。')
add('evaluate_after_q.py 只在原训练/验证上重建同一配置，再次严格匹配 15:16 完整指纹；加载已完成且内容哈希锁定的 q 数据与决定，仅复算其统计，不再运行整批 q 消融、不重训或重选。随后在已用验证谱31438上做两次相同读数的缓存探针：原始状态差异只允许 GF._m2 一项，所有其他权重、先验、协方差、归一化、字典和配置必须逐项相同；恢复该缓存前后的 A、B原始、B一致化、软融合及wA输出最大绝对差均为0。缓存恢复后完整状态必须重新等于原指纹。以上断言全部通过后，才允许生成22658。')
add('22658 评估结束后同样先保存 scratch_state_audit_after_22658.json，核实只有该批次缓存变化，再恢复进入评估前的缓存并保存完整 post 指纹。故“完整指纹相同”指严格审计并恢复缓存后的比较，不能解读成推理过程中连工作缓存都从未变化；没有放宽任何参数/先验检查，没有利用22658更新模型。证据：scratch_state_audit_validation_probe_1.json、scratch_state_audit_validation_probe_2.json、scratch_probe_functional_check.json、scratch_state_audit_after_22658.json、completed_q_before_evaluation.json、evaluation_run.log、evaluation_process.log。')
"""+anchor
assert anchor in src
src=src.replace(anchor,insert,1)
src=src.replace('运行耗时 `{j["seconds"]:.1f} s`，',
    '最终恢复/评估进程耗时 `{j["seconds"]:.1f} s`，另有前一完整 q 消融进程约 `{j["prior_q_process_seconds"]:.1f} s`（因缓存断言中止），',1)
src=src.replace('该计时包括配置重建、q 消融和新评估，不是 B 训练耗时。',
    '最终进程计时包括再次重建、已存 q 的统计复核、缓存探针和新评估；前一进程计时包括第一次重建及完整 q 消融。均不是 B 训练耗时，不包含12:18之前的中断计算。',1)
src=src.replace('resume_diagnose_b1.py --diag-seed 22658\\npy',
    'evaluate_after_q.py --diag-seed 22658\\npy',1)
src=src.replace('write_resume_report.py\\npy','write_final_report.py\\npy',1)
src=src.replace('verify_resume.py\\n```','verify_final.py\\n```',1)
src=src.replace('实际续跑命令（输出目录禁止覆盖，勿直接在原目录重跑）：',
    '最终恢复与评估命令（须已有完整 q 决策和冻结证据；输出目录禁止覆盖，勿直接在原目录重跑）：',1)
ast.parse(src)
(OUT/'write_final_report.py').write_text(src,encoding='utf-8')

src=(OUT/'verify_resume.py').read_text(encoding='utf-8')
src=src.replace("(OUT/'resume_run.log')", "(OUT/'evaluation_run.log')",1)
src=src.replace("(OUT/'resume_diagnose_b1.py')", "(OUT/'evaluate_after_q.py')",1)
src=src.replace("(OUT/'resume_process.log')", "(OUT/'evaluation_process.log')",1)
anchor="append=load('handoff_append.json');"
i=src.index(anchor)
src=src[:i]+"""# The failed q-process remains preserved; completed evidence is immutable.
completed=load('completed_q_before_evaluation.json')
for relative,expected in completed['protected_files'].items():
    check(sha(OUT/relative)==expected,'Completed q/process evidence preserved: '+relative)
failed=(OUT/'resume_process.log').read_bytes().decode('utf-8',errors='replace')
check('AssertionError' in failed,'Original strict scratch assertion remains in the failed process log')
check('Generating 22658 exactly once:' not in (OUT/'resume_run.log').read_text(encoding='utf-8'),'Failed q process did not begin 22658 generation')
scratch_key='model.decoder.closure.base.closure.gate_of.closure.GF._m2'
for stage in ('validation_probe_1','validation_probe_2','after_22658'):
    audit=load('scratch_state_audit_'+stage+'.json')
    check(set(audit['differences'])=={scratch_key},'Raw inference state changes only the declared batch scratch: '+stage)
    check(audit['parameter_or_prior_changes']==0 and audit['scratch_restored_after_comparison'],'All parameters/priors remain frozen: '+stage)
    check(audit['differences'][scratch_key]['before']==fr['entries'][scratch_key],'Scratch baseline matches the registered freeze: '+stage)
probe=load('scratch_probe_functional_check.json')
check(probe['status']=='passed' and probe['allowed_data_seed']==31438 and not probe['evaluation_generated'],'Cache probe used only existing validation before evaluation')
check(not probe['excluded_parameter_paths'] and probe['original_full_state_recovered'],'No model parameter paths excluded from strict verification')
check(all(v==0 for v in probe['maximum_absolute_output_difference'].values()),'Restoring batch scratch leaves A/B/consistency/fusion/weights exactly unchanged')
check(runlog.index('Scratch probe passed:') < runlog.index('Generating 22658 exactly once:'),'Scratch explanation was validated before 22658 was generated')
check('assert set(differences)<={scratch_key}' in newsrc and 'assert restored==full_freeze' in newsrc,'Runner enforces scratch-only differences and full-state restoration')
"""+src[i:]
ast.parse(src)
(OUT/'verify_final.py').write_text(src,encoding='utf-8')
print('Prepared final reporting and two-phase verification with explicit scratch-buffer disclosure.')
