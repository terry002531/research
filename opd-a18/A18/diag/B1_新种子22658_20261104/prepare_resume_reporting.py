"""AI (Codex): preserve interrupted-run helpers and prepare resumed reporting."""
from pathlib import Path
import ast

OUT = Path(__file__).resolve().parent
assert not (OUT / 'write_resume_report.py').exists()
src = (OUT / 'write_report.py').read_text(encoding='utf-8')
anchor = "add('### 保守统计判据与逐族结果')"
insert = """add('### 12:18 中断后的续跑留痕')
resume=load('resume_before.json'); full=load('full_model_freeze.json'); freeze=load('model_freeze.json')
add('12:18 的原 run.log、process.log、诊断脚本及 q_ablation_partial_1218/ 均保留。没有复用部分 q 数值拼接结果；本次在全新 q_ablation/ 从第一族重做，使用同一 31438、同一噪声流 9090，先完成 q 决策才生成 22658。恢复采用 resume_diagnose_b1.py；旧 diagnose_b1.py 保留为中断版本，不把旧命令误称为本次运行命令。原配置重建只允许原训练/验证流程，进入评估后全部拟合入口禁用；完整状态与 15:16 登记逐项相等才放行。')
add(f'15:16 登记及本次实核的完整推理状态 SHA-256：`{full["sha256"]}`。原冻结文件逐字节未变。B、融合、q 网络状态 SHA-256 分别为 `{freeze["model_state_sha256"]["B"]}`、`{freeze["model_state_sha256"]["fusion"]}`、`{freeze["model_state_sha256"]["q"]}`；原 q checkpoint 内容 SHA-256 为 `{freeze["q_checkpoint_sha256"]}`。证据见 resume_before.json、q_decision_before_evaluation.json、resume_run.log 和完整前后状态指纹。')
""" + anchor
assert anchor in src
src = src.replace(anchor, insert, 1)
src = src.replace("['谱族','对原始B准确率','对一致化B准确率','正确选A数/600','所有分对时A误差','正确选A时A误差']", "['谱族','平均wA','wA>0.5占比','对原始B准确率','对一致化B准确率','正确选A数/600','所有分对时A误差','正确选A时A误差']", 1)
old = "[r['family'],f'{float(r[\"route_vs_B_raw_accuracy\"]):.1%}',f'{float(r[\"route_vs_B_consistent_accuracy\"]):.1%}',"
new = "[r['family'],f'{float(r[\"wA_mean\"]):.3f}',f'{float(r[\"A_weight_gt_half\"]):.1%}',f'{float(r[\"route_vs_B_raw_accuracy\"]):.1%}',f'{float(r[\"route_vs_B_consistent_accuracy\"]):.1%}',"
assert old in src
src = src.replace(old, new, 1)
anchor = "add('普通会话记录："
i = src.index(anchor)
src = src[:i] + """add('实际续跑命令（输出目录禁止覆盖，勿直接在原目录重跑）：')
add('```powershell\\npy -I -X utf8 A18\\\\diag\\\\B1_新种子22658_20261104\\\\resume_diagnose_b1.py --diag-seed 22658\\npy -I -X utf8 A18\\\\diag\\\\B1_新种子22658_20261104\\\\write_resume_report.py\\npy -I -X utf8 A18\\\\diag\\\\B1_新种子22658_20261104\\\\verify_resume.py\\n```')
""" + src[i:]
ast.parse(src)
(OUT / 'write_resume_report.py').write_text(src, encoding='utf-8')

src = (OUT / 'verify_b1.py').read_text(encoding='utf-8')
src = src.replace("(OUT/'run.log')", "(OUT/'resume_run.log')", 1)
src = src.replace("(OUT/'diagnose_b1.py')", "(OUT/'resume_diagnose_b1.py')", 1)
anchor = "append=load('handoff_append.json');"
i = src.index(anchor)
src = src[:i] + """# Preserve all pre-interruption evidence, including partial q and original logs.
resume=load('resume_before.json')
for relative,expected in resume['local_existing_files'].items():
    check(sha(OUT/relative)==expected,'Pre-interruption artifact preserved: '+relative)
check(fr['sha256']==resume['full_inference_state_sha256']=='59f19648abb6ceab81cec07ca40d68c5ce188b38988d923738405e4c749976ce','Full state matches the 15:16 preregistered digest')
preq=load('q_decision_before_evaluation.json')
check(preq['retain_q']==q['retain_q'] and preq['q_summary_sha256']==sha(OUT/'q_ablation/summary.json'),'q decision persisted and fingerprinted before evaluation')
check(not preq['evaluation_spectra_exist'] and not preq['evaluation_generation_called'],'No 22658 spectra existed or had been generated when q was decided')
check((OUT/'q_ablation/summary.json').stat().st_mtime <= (OUT/'diagnostic_spectra.npz').stat().st_mtime,'q decision file predates evaluation spectra file')
resume_changes={}
for relative,expected in resume['outside_existing_files'].items():
    path=WORK/relative; actual=sha(path) if path.exists() else None
    if actual!=expected:resume_changes[relative]=dict(before=expected,after=actual)
check(set(resume_changes) <= {'A18\\\\HANDOFF_A18-B1.md'},'Only the authorized B1 handoff changed outside the diagnostic directory during the resumed run')
(OUT/'resume_external_changes.json').write_text(json.dumps(resume_changes,ensure_ascii=False,indent=2),encoding='utf-8')
process=(OUT/'resume_process.log').read_bytes().decode('utf-8',errors='replace')
check('Traceback (most recent call last)' not in process,'Resumed numerical process has no traceback')
check('Numerical evaluation finished; independent verification and report pending.' in runlog,'Resumed numerical process reached the end')
""" + src[i:]
src = src.replace("'old_file_changes':changes", "'old_file_changes':changes")
src = src.replace("old_file_changes=changes,source_files_checked=", "old_file_changes=changes,resume_external_changes=resume_changes,pre_interruption_files_preserved=len(resume['local_existing_files']),source_files_checked=", 1)
ast.parse(src)
(OUT / 'verify_resume.py').write_text(src, encoding='utf-8')
print('Prepared write_resume_report.py and verify_resume.py; interrupted-run helpers preserved.')
