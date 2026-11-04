"""Review a concurrent B2 document update; retain the original failed scope audit."""
from pathlib import Path
import json,hashlib,datetime,csv

OUT=Path(__file__).resolve().parent; WORK=OUT.parents[2]
load=lambda name:json.loads((OUT/name).read_text(encoding='utf-8'))
changes=load('existing_file_changes.json')
assert set(changes)=={'A18\\HANDOFF_A18-B2.md'}, changes
hand=WORK/'A18/HANDOFF_A18-B2.md'
txt=hand.read_text(encoding='utf-8')
assert '2026-11-04 10:50 副导师审查' in txt and '科研牛马（实验）' in txt
assert hand.stat().st_size==changes['A18\\HANDOFF_A18-B2.md']['after']['size']
for rel,expected in load('old_artifacts_before.json').items():
    assert hashlib.sha256((WORK/rel).read_bytes()).hexdigest()==expected,rel
for rel,expected in load('source_manifest.json')['sources'].items():
    assert hashlib.sha256((WORK.parent/rel).read_bytes()).hexdigest()==expected,rel
for filename,n in [('family_metrics.csv',26),('per_reading_metrics.csv',3120)]:
    assert len(list(csv.DictReader((OUT/filename).open(encoding='utf-8-sig'))))==n
assert load('model_post_evaluation.json')['unchanged']
result=dict(status='external_concurrent_change_reviewed',reviewed_at=datetime.datetime.now().astimezone().isoformat(),
    raw_process_exit_code=1,failed_stage='final directory-wide before/after assertion, after all numerical outputs were saved',
    numerical_evaluation_rerun=False,original_change_record_preserved=True,
    external_changes={str(hand.relative_to(WORK)):dict(**changes[str(hand.relative_to(WORK))],
        current_sha256=hashlib.sha256(hand.read_bytes()).hexdigest())},
    evidence='B2 append is attributed to 科研牛马（实验）; running B1 audit hook forbids writes outside its diagnostic directory.',
    old_B1_artifacts_and_algorithm_sources_unchanged=True,
    conclusion='Do not overwrite B2 or conceal the original audit failure; independently verify all B1 numerical data and protected files.')
(OUT/'scope_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
with (OUT/'run.log').open('a',encoding='utf-8') as f:
    f.write('[post-run scope review] Numerical evaluation saved: 26 families, 3120 readings. Raw process exit=1 only at final file-scope assertion due to concurrently updated HANDOFF_A18-B2.md. Original error and change record retained; see scope_review.json. No numerical rerun or parameter changes.\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
