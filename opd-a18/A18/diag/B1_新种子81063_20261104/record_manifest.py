"""Hash old artifacts and source files without opening any test results."""
from pathlib import Path
import hashlib,json,datetime

OUT=Path(__file__).resolve().parent
WORK=OUT.parents[2]
OLD=OUT.parent/'B1_诊断_20261104'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (OUT/'old_artifacts_before.json').exists()
before={str(p.relative_to(WORK)):sha(p) for p in OLD.rglob('*') if p.is_file()}
(OUT/'old_artifacts_before.json').write_text(json.dumps(before,ensure_ascii=False,indent=2),encoding='utf-8')
hand=WORK/'A18/HANDOFF_A18-B1.md'
(OUT/'HANDOFF_before.md').write_bytes(hand.read_bytes())
roots=[WORK/'算法对照',WORK.parent/'03_仿真/41_重建算法二_形状自适应解码/scripts']
sources={str(p.relative_to(WORK.parent)):sha(p) for root in roots for p in root.glob('*.py')}
(OUT/'source_manifest.json').write_text(json.dumps(dict(recorded=datetime.datetime.now().isoformat(),
    sources=sources,old_B1_artifacts=len(before),test_results_read=False),ensure_ascii=False,indent=2),encoding='utf-8')
print(f'Hashed {len(before)} old B1 artifacts and {len(sources)} sources; backed up handoff bytes.')
