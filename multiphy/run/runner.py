# [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: one runner loop for every backend and dataset kind | date: 2026-09-22]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: docstring names run_selection as the real-day caller; multiday_runner is retired (Plan B Task 8) | date: 2026-09-28]
"""Run one backend on one dataset, group by group, and leave a run directory.

Replaces ``runners.run_backend_dataset`` and the in-process path in ``gui.py``. The
dataset is duck-typed: it has ``units``, ``pairs``, ``metadata``, ``subset(ids)``,
``waveform_bundle()`` and optionally ``features`` and ``labels``. ``group_column`` is
``recording_group``, for benchmarks and for the real days ``project_run.run_selection`` passes.
"""
# [AGENT EDIT END]
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import fields as dataclass_fields, replace
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import sys
import tempfile
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: named files a caller writes into the run folder (Plan B Task 2) | date: 2026-09-28]
from typing import Optional
# [AGENT EDIT END]

import pandas as pd

from ..contract import registry
from ..contract.backend import RunContext
from ..contract.errors import BackendUnavailable, ContractViolation, MissingInputs
from ..contract.inputs import MatchingInput
# [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: inputs that only a waveform bundle supplies (whole-branch review fix 5) | date: 2026-09-23]
from ..contract.inputs import Input
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: backend_label delegates to contract.names (whole-branch review fix 3) | date: 2026-09-23]
from ..contract import names
# [AGENT EDIT END]
from ..contract.provenance import Provenance, json_safe
from ..contract.results import MatchResult, single_units
from . import config, options as opt
from .run_dir import save_result, write_json

# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: no fitted model is shared across groups (Task 2) | date: 2026-09-27]
SHARED_PROVENANCE_KEYS = ('assignment', 'threshold', 'score_kind', 'calibration_status', 'calibration_reference',
                          'enforce_session_unique', 'official_source', 'verified_api_revision', 'pair_reduction',
                          'feature_stream', 'score_meaning', 'evidence_display_name', 'evidence_columns')
# [AGENT EDIT END]
_TYPED_PROVENANCE = frozenset(f.name for f in dataclass_fields(Provenance)) - {'extra'}
# [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: inputs that only a waveform bundle supplies (whole-branch review fix 5) | date: 2026-09-23]
WAVEFORM_DERIVED = frozenset({Input.WAVEFORM_HALVES, Input.GEOMETRY, Input.SPIKE_TIMES, Input.FINGERPRINT})
# [AGENT EDIT END]


@contextmanager
def cache_runtime(root=None):
    """Contain upstream temporary files and library caches under the cache root."""
    runtime = config.resolve_cache_path('_runtime', root=root)
    runtime.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='runner-', dir=runtime) as work:
        previous_tempdir, previous_bytecode = tempfile.tempdir, sys.dont_write_bytecode
        variables = {'TMPDIR': work, 'TMP': work, 'TEMP': work, 'MPLCONFIGDIR': str(Path(work) / 'matplotlib'),
                     'XDG_CACHE_HOME': str(Path(work) / 'cache'), 'TORCH_HOME': str(Path(work) / 'torch'),
                     'NUMBA_CACHE_DIR': str(Path(work) / 'numba'), 'PYTHONDONTWRITEBYTECODE': '1'}
        previous = {key: os.environ.get(key) for key in variables}
        try:
            os.environ.update(variables)
            tempfile.tempdir = work
            sys.dont_write_bytecode = True
            yield Path(work)
        finally:
            tempfile.tempdir, sys.dont_write_bytecode = previous_tempdir, previous_bytecode
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _select_partition(dataset, partition):
    if partition in (None, 'all'):
        return dataset
    labels = getattr(dataset, 'labels', None)
    if labels is None or 'partition' not in labels:
        raise ValueError('Partition selection needs benchmark labels with a partition column')
    ids = labels.loc[labels.partition == partition, 'cellid'].tolist()
    if not ids:
        raise ValueError(f'Dataset has no units in partition {partition!r}')
    return dataset.subset(ids)


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the extra inputs a caller may supply, named once (Task 2 fix round 1) | date: 2026-09-27]
EXTRA_INPUTS = ('timing', 'checkpoint', 'fingerprint', 'fingerprint_bin_keys')
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: backend input units carry n_spikes from features, NaN when absent (native-same-day Task 2) | date: 2026-10-11]
def build_input(subset, group, *, extra_inputs=None, provenance=None) -> MatchingInput:
    """One MatchingInput for one group; waveforms only when the subset can provide them.

    The units carry ``n_spikes`` from each unit's features, NaN when absent, for the native
    backends' one-unit-per-session tie-break.
    """
    extra = dict(extra_inputs or {})
    features = getattr(subset, 'features', None)
    bundle = None
    # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: keep why the waveform bundle could not be built, in provenance, for every backend (whole-branch review fix 5) | date: 2026-09-23]
    if hasattr(subset, 'waveform_bundle'):
        try:
            bundle = subset.waveform_bundle()
        except (ValueError, KeyError) as exc:
            bundle = None
            provenance = (provenance or Provenance()).merged(extra={'waveform_bundle_error': str(exc)})
    # [AGENT EDIT END]
    units = subset.units.copy()
    units['n_spikes'] = [float(v) if v is not None else float('nan')
                         for v in ((features or {}).get(str(c), {}).get('n_spikes') for c in units.cellid)]
    common = dict(group=str(group), pairs=subset.pairs, features=features, provenance=provenance)
    fields = {k: v for k, v in extra.items() if not k.startswith('_')}
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: an undeclared extra input (e.g. a removed fitted_model) is refused, never dropped (Task 2 fix round 1) | date: 2026-09-27]
    unknown = sorted(set(fields) - set(EXTRA_INPUTS))
    if unknown:
        raise ValueError(f'Unknown extra inputs {unknown}; accepted: {list(EXTRA_INPUTS)}')
    # [AGENT EDIT END]
    if bundle is not None:
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: only timing and checkpoint are forwarded; fingerprint is replaced below (Task 2) | date: 2026-09-27]
        inp = MatchingInput.from_bundle(units, bundle, **common, **{k: v for k, v in fields.items()
                                                                   if k in ('timing', 'checkpoint')})
        # [AGENT EDIT END]
        if 'fingerprint' in fields:
            inp = replace(inp, fingerprint=fields['fingerprint'],
                          fingerprint_bin_keys=tuple(fields.get('fingerprint_bin_keys', ())))
        elif inp.fingerprint is not None and len(inp.fingerprint_bin_keys) < 2:
            # An archive whose shared conditions leave fewer than two aligned bins has no usable
            # fingerprint: withhold it (a backend requiring one then fails as MissingInputs) and say so.
            inp = replace(inp, fingerprint=None, fingerprint_bin_keys=(),
                          provenance=inp.provenance.merged(extra={'fingerprint_withheld':
                              f'{len(inp.fingerprint_bin_keys)} aligned stimulus-time bin(s); at least two are required'}))
    else:
        inp = MatchingInput(units, str(group), candidate_pairs=subset.pairs, features=features,
                            provenance=provenance or Provenance(), **fields)
    return inp.validate()
# [AGENT EDIT END]


def merge_results(results, backend_id, output_name, provenance) -> MatchResult:
    if not results:
        raise ValueError('No groups were run')
    if len({r.version for r in results}) != 1:
        raise ContractViolation('Backend version changed between groups')
    first = results[0]
    combined = provenance.merged(extra={'independent_groups': True,
        'group_runs': {str(r.provenance.get('group')): {'run_id': r.run_id, 'provenance': r.provenance.to_json()} for r in results}})
    shared = {}
    for key in SHARED_PROVENANCE_KEYS:
        value = first.provenance.get(key)
        if value is not None and all(r.provenance.get(key) == value for r in results):
            shared[key] = value
    # Typed fields (calibration_status, verified_api_revision) are set as fields, never shadowed in extra.
    combined = combined.merged(**{k: v for k, v in shared.items() if k in _TYPED_PROVENANCE},
                               extra={k: v for k, v in shared.items() if k not in _TYPED_PROVENANCE})
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the merged score names every group's calibration reference, not only the first group's (Task 8 fix round 1) | date: 2026-09-27]
    references = list(dict.fromkeys(r.score.calibration_reference for r in results
                                    if r.score.calibration_reference))
    score = replace(first.score, calibration_reference='; '.join(references)) if references else first.score
    # [AGENT EDIT END]
    return MatchResult(backend_id, first.version, output_name,
                       pd.concat([r.units for r in results], ignore_index=True),
                       pd.concat([r.pairs for r in results], ignore_index=True),
                       tuple(t for r in results for t in r.tracks), tuple(u for r in results for u in r.unmatched),
                       tuple(u for r in results for u in r.excluded_units), score, first.track_kind,
                       first.tracking_rule, combined).validate()


# [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: one fallback implementation, in contract.names (whole-branch review fix 3) | date: 2026-09-23]
def backend_label(backend_id) -> str:
    """Registered display name; a third-party or test backend without one shows its id."""
    return names.backend_label(backend_id)
# [AGENT EDIT END]


def _methods_md(output, backend_id, group_column, partition, status):
    return (f'# Matching run — methods notes\n\n- **Output folder:** `{output}`\n'
            '- **Artifacts:** `result.json`, `run_manifest.json`, and `evaluation.json` when labels exist, else `availability.json`\n'
            f'- **Generated by:** `multiphy/run/runner.py:run()` · **Updated:** {datetime.now(timezone.utc).date()}\n\n'
            f'## Goal\n\nRun {backend_label(backend_id)} independently within each `{group_column}` group'
            f'{" of the " + partition + " partition" if partition else ""}. Tracks and unmatched units are proposals; '
            f'this run alone does not establish biological identity. Run status: {status}.\n\n'
            '## Data lineage\n\n- Inputs, splitting, extraction and labels are documented in the dataset manifest and METHODS.md.\n'
            '- `result.json` ← the backend named above through the shared contract; implementation details and parameters are in result provenance.\n'
            '- `evaluation.json` ← `evaluation.evaluate_result()` against source-cluster labels, per group and combined.\n\n'
            '## Panels\n\nNo figures.\n')


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a project run writes its selection and pairs before the manifest (Plan B Task 2) | date: 2026-09-28]
# Files the runner itself writes into a run folder; a caller's ``files`` cannot take these names.
RUNNER_FILES = ('run_manifest.json', 'METHODS.md', 'result.json', 'evaluation.json', 'availability.json')


def _check_file_names(files) -> None:
    for name in files:
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a NUL byte is refused before the folder exists (Plan B Task 2 fix round 1) | date: 2026-09-28]
        if name in RUNNER_FILES or '/' in name or '\0' in name or name in ('', '.', '..'):
        # [AGENT EDIT END]
            raise ValueError(f'run file name {name!r} is reserved or not a plain file name')


def run(dataset, backend_id, options=None, *, group_column, partition=None, output_dir=None, root=None,
        evaluate=True, extra_inputs=None, files: Optional[dict] = None) -> Path:
    """Steps 1 to 9 of the spec: resolve, validate options, group, check inputs, run, merge, write.

    ``files`` (``{name: bytes}``) are written once into the new run folder before its manifest,
    so they stay beside a failed run.
    """
    files = dict(files or {})
    _check_file_names(files)
    # [AGENT EDIT END]
    backend = registry.get(backend_id)
    spec = backend.spec
    availability = backend.available()
    if not availability.ok:
        raise BackendUnavailable(f'{backend_id}: {availability.reason}')
    resolved = opt.validate(spec, options)
    metadata = dict(dataset.metadata)  # the caller's metadata; a partition subset need not carry it
    dataset = _select_partition(dataset, partition)
    expected_partition = None if partition in (None, 'all') else partition
    if group_column not in dataset.units or dataset.units[group_column].isna().any():
        raise ValueError(f'Every unit requires a {group_column!r} value')
    dataset_id = str(metadata['dataset_id'])
    input_path = metadata.get('input_path')
    manifest_sha = metadata.get('input_manifest_sha256')
    base = Provenance.from_json(metadata).merged(dataset_id=dataset_id, partition=partition,
                                                        input_manifest_sha256=manifest_sha)
    groups = []
    for group, rows in dataset.units.groupby(group_column, sort=True):
        subset = dataset.subset(rows.cellid.tolist())
        inp = build_input(subset, group, extra_inputs=extra_inputs, provenance=base.merged(group=str(group)))
        missing = spec.requires - inp.available()
        if missing:
            # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: a missing waveform-derived input names the bundle error behind it (whole-branch review fix 5) | date: 2026-09-23]
            bundle_error = inp.provenance.get('waveform_bundle_error')
            detail = (f'waveform bundle could not be built: {bundle_error}'
                      if bundle_error and missing & WAVEFORM_DERIVED else None)
            raise MissingInputs(backend_id, str(group), missing, detail)
            # [AGENT EDIT END]
        groups.append((str(group), rows, subset, inp))
    output = (config.new_run_path(dataset_id, root=root) if output_dir is None
              else config.resolve_cache_path(output_dir, root=root))
    output.mkdir(parents=True, exist_ok=False)
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the caller's files are written once, before the first manifest write (Plan B Task 2) | date: 2026-09-28]
    for name, content in files.items():
        with (output / name).open('xb') as stream:
            stream.write(content)
    # [AGENT EDIT END]
    manifest = dict(status='running', backend=backend_id, group_column=group_column, partition=partition,
                    dataset_id=dataset_id, input_path=input_path, input_manifest_sha256=manifest_sha,
                    options=json_safe(resolved), output_dir=str(output),
                    started_at=datetime.now(timezone.utc).isoformat())
    write_json(output / 'run_manifest.json', manifest, exclusive=True)
    (output / 'METHODS.md').write_text(_methods_md(output, backend_id, group_column, partition, 'running'))
    try:
        with cache_runtime(root) as work:
            results, evaluations = [], {}
            for group, rows, subset, inp in groups:
                token = hashlib.sha256(group.encode()).hexdigest()[:16]
                ctx = RunContext(f'{output.name}_{token}', Path(work) / token, inp.provenance)
                result = backend.run(inp, resolved, ctx).validate(spec)
                expected, excluded = single_units(inp.units)
                if set(result.units.cellid) != set(expected.cellid) or set(result.excluded_units) != set(excluded):
                    raise ContractViolation(f'{backend_id} must account for every input single unit in group {group!r}')
                if result.run_id != ctx.run_id:
                    raise ContractViolation(f'{backend_id} changed the run identity')
                # Dates come from the dataset's own rows: a subset container may not carry them.
                if 'session_date' in rows:
                    dates = rows.assign(cellid=rows.cellid.astype(str)).set_index('cellid').session_date
                    if any(dates.loc[list(t)].duplicated().any() for t in result.tracks):
                        raise ContractViolation('A proposed track cannot contain two observations from one actual day')
                results.append(result)
                labels = getattr(subset, 'labels', None)
                if evaluate and labels is not None:
                    from ..evaluation import evaluate_result
                    evaluations[group] = evaluate_result(result, labels, expected_partition=expected_partition)
            combined = merge_results(results, backend_id, output.name, base)
            save_result(combined, output / 'result.json')
            labels = getattr(dataset, 'labels', None)
            if evaluate and labels is not None:
                from ..evaluation import evaluate_result
                write_json(output / 'evaluation.json', dict(
                    combined=evaluate_result(combined, labels, expected_partition=expected_partition),
                    groups=evaluations,
                    interpretation='Source-sort conditional identity; groups are never pooled for matching'),
                    exclusive=True)
            else:
                write_json(output / 'availability.json', dict(
                    interpretation='Descriptive output coverage only; across-day correctness is unknown',
                    n_observations=len(combined.units), n_proposed_tracks=len(combined.tracks),
                    n_reported_unmatched=len(combined.unmatched), n_candidate_pair_rows=len(combined.pairs)), exclusive=True)
        manifest.update(status='completed', run_id=combined.run_id, result_path=str(output / 'result.json'),
                        finished_at=datetime.now(timezone.utc).isoformat())
        (output / 'METHODS.md').write_text(_methods_md(output, backend_id, group_column, partition, 'completed'))
    except BaseException as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}', finished_at=datetime.now(timezone.utc).isoformat())
        (output / 'METHODS.md').write_text(_methods_md(output, backend_id, group_column, partition, 'failed; inspect run_manifest.json'))
        raise
    finally:
        write_json(output / 'run_manifest.json', manifest)
    return output
# [AGENT EDIT END]
