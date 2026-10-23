"""Extract a benchmark from an explicit JSON source manifest.

Input spike arrays must already be zero-based rows of the raw voltage stream.
Trial-relative spike times require an independently verified conversion first.
Compressed voltage is traversed forwards once; by default selected snippets are
temporarily stored. Opt-in shared-profile streaming summaries avoid snippet files.
The default scratch location is the matching cache; an
explicit external scratch directory can be supplied without relocating results.
"""
import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .benchmark import SourceRecording, build_dataset, save_dataset
from .extract.raw_pass import FilteredSnippetReader, SnippetReader, _voltage_stream, plan_snippets
from .extract.raw_pass import raw_segments
from .run.config import extraction_scratch_root, resolve_cache_path
from .evaluation import validate_partitions




def _input_path(value, manifest_dir):
    path = Path(value).expanduser()
    return path if path.is_absolute() else manifest_dir / path


@contextmanager
def open_recording_sources(manifest, manifest_path, *, options, root=None, scratch_root=None,
                           waveform_storage='snippets', hash_raw=False):
    """Open verified readers while the caller computes its own archive format.

    Callers determine observation windows and validate their assignment scope.
    This layer enforces raw-clock, preprocessing and snippet-plan integrity;
    no identity labels or benchmark partition semantics are inferred here.
    Readers and temporary selected snippets close when the context exits.
    waveform_storage='streaming_summary' retains verified summaries in memory
    instead of writing snippets; it requires the shared preprocessing profile.
    scratch_root optionally relocates temporary snippets, never saved summaries.
    hash_raw=True has each reader hash a gzip raw file's stored bytes during its
    pass, into reader.raw_sha256; a flat raw file given as one path is never hashed.
    A raw segment list is hashed as one sha256 of every segment's stored bytes, in
    clock order, flat segments included.
    """
    manifest_path = Path(manifest_path).resolve()
    if waveform_storage not in ('snippets', 'streaming_summary'):
        raise ValueError('waveform_storage must be snippets or streaming_summary.')
    if waveform_storage == 'streaming_summary' and any(not s.get('waveform_preprocessing') for s in manifest['recordings']):
        raise ValueError('Streaming summary storage requires the shared waveform preprocessing profile.')
    before, after = options.get('before', 41), options.get('after', 41)
    runtime = extraction_scratch_root(scratch_root, root=root)
    runtime.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        work = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='extract-', dir=runtime)))
        print(f'Temporary extraction working directory: {work} (removed on normal context exit)', flush=True)
        sources = []
        for i, spec in enumerate(manifest['recordings']):
            clock = spec.get('clock_verification', {})
            if clock.get('sample_index_origin') != 'zero_based_voltage_row' or not clock.get('method'):
                raise ValueError('Each source needs verified zero-based voltage-row spike indices and its verification method.')
            provenance = dict(spec.get('provenance', {}))
            if isinstance(spec['raw_voltage'], (list, tuple)):
                raw = [[_input_path(path, manifest_path.parent), n] for path, n in spec['raw_voltage']]
            else:
                raw = _input_path(spec['raw_voltage'], manifest_path.parent)
            raw_paths = [path for path, _ in raw_segments(raw, spec['n_samples'])]
            spikes_path = _input_path(spec['spikes_file'], manifest_path.parent)
            geometry_path = _input_path(spec['geometry_file'], manifest_path.parent)
            if not spec.get('preprocessing'):
                raise ValueError('Describe acquisition filtering and chosen voltage preprocessing explicitly.')
            with np.load(spikes_path, allow_pickle=False) as saved:
                spikes = {key: saved[key].copy() for key in spec['single_unit_ids']}
            geometry = np.load(geometry_path, allow_pickle=False)
            windows = np.asarray(spec['windows'])
            if (windows.ndim != 2 or windows.shape[1] != 2 or not windows.size
                    or not np.isfinite(windows).all() or not np.equal(windows, np.floor(windows)).all()
                    or windows.min() < 0 or windows.max() > spec['n_samples']
                    or (windows[:, 1] <= windows[:, 0]).any() or (windows[1:, 0] < windows[:-1, 1]).any()):
                raise ValueError('Temporal windows must fit the declared raw sample count.')
            windows = windows.astype(np.int64)
            summary_plan = None
            if waveform_storage == 'streaming_summary':
                from .streaming_waveforms import plan_waveform_summaries
                summary_plan = plan_waveform_summaries(spikes, windows, **options)
            samples = summary_plan.samples if summary_plan is not None else plan_snippets(spikes, windows, **options)
            print(f"Extracting {len(samples)} selected voltage snippets for {spec['source_session']}", flush=True)
            shared = spec.get('waveform_preprocessing')
            snippet_bytes = len(samples) * (before + after) * len(geometry) * (
                4 if shared else np.dtype(spec.get('dtype', '<i2')).itemsize)
            if summary_plan is None:
                print(f'Planned temporary snippet storage: {snippet_bytes:,} bytes '
                      f'({snippet_bytes / 2**30:.2f} GiB); no full recording is unpacked', flush=True)
            else:
                bound = len(summary_plan.selections) * (before + after) * len(geometry) * 16
                print(f'Streaming {len(summary_plan.selections)} exact summaries; mean/M2 upper bound '
                      f'{bound / 2**20:.2f} MiB plus filtering workspace; no temporary snippet files', flush=True)
            if shared and (spec.get('baseline_samples', 0) or spec.get('common_median', False)):
                raise ValueError('Shared preprocessing replaces legacy common_median/baseline_samples options.')
            reader_type = FilteredSnippetReader if shared else SnippetReader
            processing_args = (dict(sample_rate=spec['sample_rate'], preprocessing=shared) if shared else
                dict(common_median=spec.get('common_median', False), baseline_samples=spec.get('baseline_samples', 0)))
            if summary_plan is not None:
                processing_args['summary_plan'] = summary_plan
            reader = reader_type(raw, samples, n_channels=len(geometry), n_samples=spec['n_samples'],
                directory=work / f'recording_{i}', before=before, after=after,
                dtype=spec.get('dtype', '<i2'), voltage_scale=spec.get('voltage_scale', 1.),
                **processing_args,
                progress=lambda message: print(message, flush=True), hash_raw=hash_raw)
            stack.callback(reader.close)
            raw_stats = [path.stat() for path in raw_paths]
            provenance.update(raw_source=os.pathsep.join(str(path) for path in raw_paths),
                raw_size_bytes=sum(s.st_size for s in raw_stats),
                raw_mtime_ns=max(s.st_mtime_ns for s in raw_stats), spikes_source=str(spikes_path),
                spikes_sha256=hashlib.sha256(spikes_path.read_bytes()).hexdigest(),
                geometry_source=str(geometry_path), geometry_sha256=hashlib.sha256(geometry_path.read_bytes()).hexdigest(),
                clock_verification=clock, n_samples=spec['n_samples'],
                voltage_scale=spec.get('voltage_scale', 1.), common_median=spec.get('common_median', False),
                baseline_samples=spec.get('baseline_samples', 0),
                waveform_processing='raw acquired AP samples; optional samplewise channel median and pre-spike mean subtraction; no additional digital filter')
            provenance['waveform_storage'] = waveform_storage
            if shared:
                provenance['waveform_preprocessing'] = reader.processing_metadata
                provenance['waveform_processing'] = 'Acquisition scaling; buffered zero-phase high-pass; median reference; unwhitened physical waveform'
                provenance['common_median'] = True
            sources.append(SourceRecording(spec['source_session'], spec['well'], spec['partition'],
                spec['sample_rate'], geometry, windows, spikes, tuple(spec['single_unit_ids']), reader,
                spec['preprocessing'], provenance, spec.get('stimuli'), spec.get('baseline_windows')))
        yield sources


def extract_manifest(manifest_path, *, output_dir=None, root=None, scratch_root=None,
                     waveform_storage='snippets'):
    """Run explicit extraction without guessing clock alignment or preprocessing."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema_version') != 1 or not manifest.get('recordings'):
        raise ValueError('Extraction manifest requires schema_version=1 and recording inputs.')
    source_groups = pd.DataFrame([{key: spec[key] for key in ('well', 'source_session', 'partition')}
                                  for spec in manifest['recordings']])
    validate_partitions(source_groups)
    if source_groups.duplicated(['well', 'source_session']).any():
        raise ValueError('An original source recording cannot appear twice in the extraction inputs.')
    if not set(source_groups.partition) <= {'train', 'calibration', 'test'}:
        raise ValueError('Use train, calibration, or test source partitions.')
    dataset_id = manifest['dataset_id']
    if not isinstance(dataset_id, str) or not dataset_id or any(c in dataset_id for c in '/\\') or dataset_id in ('.', '..'):
        raise ValueError('dataset_id must be a single nonempty directory name.')
    target = resolve_cache_path(output_dir or Path('benchmarks') / dataset_id / 'dataset', root=root)
    if target.exists():
        raise FileExistsError(f'Benchmark output already exists: {target}')
    options = dict(manifest.get('extraction_options', {}))
    before, after = options.get('before', 41), options.get('after', 41)
    options.update(before=before, after=after)
    with open_recording_sources(manifest, manifest_path, options=options, root=root,
                                scratch_root=scratch_root, waveform_storage=waveform_storage) as sources:
        dataset = build_dataset(sources, dataset_id=dataset_id,
            omitted=manifest.get('omitted', ()), extraction_options=options, pair_options=manifest.get('pair_options'))
        dataset.metadata['extraction_manifest_sha256'] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        dataset.metadata['waveform_storage'] = waveform_storage
        saved = save_dataset(dataset, target, cache_root=root)
        (saved / 'extraction_manifest.json').write_text(json.dumps(manifest, indent=2))
        methods = saved / 'METHODS.md'
        note = ('- Waveform accumulation ← `features.py:average_spike_waveforms()`; '
            f'operational storage mode `{waveform_storage}`. ' +
            ('`streaming_waveforms.py:StreamingWaveformSummaries` accumulates exact full/half selections during the '
             'verified forward voltage read, with float32 crop quantization followed by timestamp-ordered float64 '
             'Welford mean/standard-error calculation. No temporary snippet files are written. '
             if waveform_storage == 'streaming_summary' else
             'Selected snippets are stored temporarily and read in selected timestamp order for each summary. ') +
            'The storage choice does not change waveform preprocessing, selection seeds, physiology or stimulus fingerprints.\n')
        documentation = methods.read_text(encoding='utf-8').replace('## Data lineage\n', '## Data lineage\n\n' + note, 1)
        if waveform_storage == 'streaming_summary':
            documentation = documentation.replace('temporary processed snippets are removed.',
                'summary accumulation occurs in memory without temporary processed snippet files.')
        methods.write_text(documentation, encoding='utf-8')
    return saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, help='JSON with verified raw-clock spike arrays and source metadata')
    parser.add_argument('--output', help='New output directory beneath the existing unit-match cache root')
    parser.add_argument('--scratch-root', help='Absolute directory outside the code repository for temporary selected snippets; defaults to the matching cache _runtime directory')
    parser.add_argument('--waveform-storage', choices=('snippets', 'streaming_summary'), default='snippets',
                        help='Opt-in shared-profile streaming summaries avoid temporary snippet files; default preserves snippet storage')
    parser.add_argument('--root', metavar='DIR', help='lab root · else MULTIPHY_CACHE_ROOT')
    args = parser.parse_args(argv)
    print(f'Benchmark dataset saved: {extract_manifest(args.manifest, output_dir=args.output, root=args.root, scratch_root=args.scratch_root, waveform_storage=args.waveform_storage)}')


if __name__ == '__main__':
    main()
