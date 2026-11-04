"""Read-only B2 preregistration audit. Does not generate evaluation spectra.

This script deliberately does not import any project runner or fit any model.
Run once with: py -I -X utf8 <this file>
"""
from pathlib import Path
import ast
import csv
from datetime import datetime
import hashlib
import json
import sys

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
WORK = OUT.parents[2]
OLD = OUT.parent / 'B2_第六类_20261104'
HANDOFF = WORK / 'A18/HANDOFF_A18-B2.md'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(name, obj):
    with (OUT / name).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def describe(path):
    return dict(path=str(path.relative_to(WORK)), size=path.stat().st_size,
                sha256=sha(path))


def main():
    assert not (OUT / 'preflight.json').exists(), 'Preserve the original audit.'
    before = {
        str(p.relative_to(WORK)): dict(size=p.stat().st_size,
                                      mtime_ns=p.stat().st_mtime_ns)
        for p in WORK.rglob('*')
        if p.is_file() and not p.is_relative_to(OUT)
    }
    save('existing_files_before.json', before)
    old_files = {str(p.relative_to(WORK)): describe(p)
                 for p in OLD.rglob('*') if p.is_file()}
    source = {}
    protocol = json.loads((OLD / 'protocol.json').read_text(encoding='utf-8'))
    for rel, expected in protocol['source_sha256'].items():
        p = WORK / rel
        item = describe(p)
        item.update(expected_sha256=expected, matches_original=item['sha256'] == expected)
        source[rel] = item
    bp = WORK / '算法对照/45_Uddin2024_真实R矩阵/results/a15_nets/cnet0.pt'
    b = describe(bp)
    b.update(expected_sha256=protocol['B_sha256'],
             matches_original=sha(bp) == protocol['B_sha256'])
    assert b['matches_original'] and all(s['matches_original'] for s in source.values())
    save('input_inventory_before.json', dict(old_B2=old_files, source=source, B=b,
        handoff=describe(HANDOFF), diagnostic_generated=False))

    # Import NumPy only: loading array metadata does not execute project drivers.
    import numpy as np
    with (OLD / 'oracle_metrics_complete_resume_v2.csv').open(encoding='utf-8-sig', newline='') as f:
        selected = list(csv.DictReader(f))
    priors = []
    for j, row in enumerate(selected):
        p = OLD / f'prior_{j}.npz'
        with np.load(p, allow_pickle=False) as z:
            n, d = z['mu'].shape
            assert z['cov'].shape == (n, d, d) and z['weight'].shape == (n,)
            assert np.isclose(z['weight'].sum(), 1)
            assert n == int(row['actual_K'])
            priors.append(dict(family=row['family'], requested_K=int(float(row['selected_K'])),
                actual_K=n, dimension=d, prior_kind=row['prior_kind'],
                shrink=float(row['shrink']), **describe(p)))

    with np.load(OLD / 'allowed_data.npz', allow_pickle=False) as z:
        assert z['full_train'].shape == (87000, 176)
        training_counts = {str(f): len(z[f'tr{j}']) for j, f in enumerate(z['family_names'])}
        r = float(np.mean(z['full_train'][:, z['BM']] ** 2) /
                  np.mean(np.diff(z['full_train'][:, z['BM']], axis=1) ** 2))
        assert abs(r - protocol['loss_r']) < 1e-10
        allowed_data = dict(path=str((OLD / 'allowed_data.npz').relative_to(WORK)),
            arrays=z.files, training_counts=training_counts, loss_r=r,
            note='Archive contains old training/validation/development spectra only; no new spectra generated.')
    definitions = ast.parse((OLD / 'run_oracles.py').read_text(encoding='utf-8'))
    fit = next(n for n in definitions.body if isinstance(n, ast.FunctionDef) and n.name == 'fit_prior')
    main_node = next(n for n in definitions.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    assert 'n_clusters=K' in ast.unparse(main_node)
    assert 'if len(z) < 3' in ast.unparse(fit)
    audit = dict(task='A18-B2 seed 63862 preregistration preflight', author='AI（Codex）',
        timestamp=datetime.now().astimezone().isoformat(),
        status='PENDING_REQUIRED_INPUT_CLARIFICATION',
        diagnostic_generated=False, evaluation_executed=False,
        training_executed=False, prior_fitting_executed=False,
        old_source_and_B_hashes_match=True, existing_selected_priors=priors,
        old_allowed_data=allowed_data,
        blockers=[
            'Complete saved priors for both requested K=64 and K=256 were not found. Existing prior archives contain only each family\'s selected configuration.',
            'A complete loadable frozen A model was not found. The B1 rerun reconstructs A from historical training/validation inputs.',
            'The preregistration does not specify the unknown-family oracle mixture, family universe, or mixture weights.',
        ],
        questions_sent=True,
        no_63862_result_or_model_choice_made=True,
        log_line_240='Final old subtype-mixture run; nominal K=320, not K=64 or K=256.',
        proposed_statistics=dict(n_spectra=30, draws_per_spectrum=4, nbootstrap=200000,
            bootstrap_seed=1901122, bootstrap_unit='spectrum, retain 4 paired draws',
            statistic='mean over spectra of mean over draws log(NRMSE_method/NRMSE_B)',
            uncorrected_quantiles=[0.025, 0.975],
            four_family_bonferroni_quantiles=[0.00625, 0.99375],
            correction_scope='Four target families separately for each comparison; not across all methods or K values.',
            controls='Report separately; no selection on control results.'),
        project_session_record=dict(
            intended_directory=str(WORK.parent / 'memory'),
            written=False,
            reason='Actual project root is above the writable workspace; no write permission.'))
    save('preflight.json', audit)
    with (OUT / 'run.log').open('x', encoding='utf-8') as f:
        f.write(f"{audit['timestamp']} AI（Codex） preflight only.\n")
        f.write('No evaluation, spectrum generation, model fitting or training executed.\n')
        for item in audit['blockers']:
            f.write('PENDING: ' + item + '\n')
    changes = {}
    for rel, old in before.items():
        p = WORK / rel
        now = dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) if p.exists() else None
        if now != old:
            changes[rel] = dict(before=old, after=now)
    changed_hashes = [rel for rel, old in old_files.items() if sha(WORK / rel) != old['sha256']]
    assert not changed_hashes, changed_hashes
    save('preflight_verification.json', dict(old_B2_hashes_checked=len(old_files),
        old_B2_hash_changes=changed_hashes, observed_external_metadata_changes=changes,
        new_diagnostic_has_no_results=True, diagnostic_generated=False))
    print(json.dumps(dict(status=audit['status'], prior_actual_K=[p['actual_K'] for p in priors],
        old_B2_hash_changes=changed_hashes, observed_external_changes=list(changes),
        loss_r=r, diagnostic_generated=False), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
