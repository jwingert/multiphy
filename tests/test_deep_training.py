"""Optional source integration smoke; synthetic outputs are not scientific models."""
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from multiphy.backends import deepunitmatch
from multiphy.contract.inputs import RawWaveformBundle


def synthetic_np1010():
    rng = np.random.default_rng(734)
    n = 64
    geometry = np.column_stack((np.tile([43., 11., 59., 27.], 96),
                                np.repeat(np.arange(192) * 20., 2)))
    sessions = np.repeat(['train_part0', 'train_part1'], 32)
    ids = tuple(f'{session}_{i}' for i, session in enumerate(sessions))
    units = pd.DataFrame(dict(cellid=ids, session_id=sessions, su_type='SU'))
    waves = rng.normal(0, .02, (n, 82, 384, 2))
    for i in range(n):
        position = geometry[24 + (i % 32) * 10]
        spatial = np.exp(-np.sum((geometry - position) ** 2, axis=1) / (2 * 35. ** 2))
        temporal = -np.exp(-((np.arange(82) - 41) / (2 + (i % 7) * .15)) ** 2)
        waves[i] += (30 + i % 32) * temporal[:, None, None] * spatial[None, :, None]
    bundle = RawWaveformBundle(ids, waves, tuple(sessions),
        {s: geometry for s in set(sessions)}, provenance={
            'raw_source': 'synthetic test only; not a scientific training dataset',
            'split_method': 'independent synthetic perturbations', 'probe_type': 'NP1010',
            'sample_rate': 30000.,
            'source_groups': {s: 'training_well' for s in set(sessions)}})
    return units, bundle


def test_np1010_uses_measured_four_column_preprocessing():
    _, bundle = synthetic_np1010()
    geometry = list(bundle.channel_positions.values())
    params = deepunitmatch._deep_preprocessing(bundle.waveforms, geometry, 'NP1010')
    assert params['n_xchannelpos'] == 4
    assert params['RnChannels'] == 30 and params['RnTime'] == 60
    with pytest.raises(ValueError, match='conflicts'):
        deepunitmatch._deep_preprocessing(bundle.waveforms, geometry, 'NP1010', {'n_xchannelpos': 2})
    with pytest.raises(ValueError, match='probe_type'):
        deepunitmatch._deep_preprocessing(bundle.waveforms, geometry, None)


def test_training_refuses_missing_rate_and_fractional_epochs_before_output(tmp_path):
    units, bundle = synthetic_np1010()
    kwargs = dict(source_checkout=tmp_path, output_dir=tmp_path / 'out',
                  training_groups=['training_well'], evaluation_groups=['other_well'])
    provenance = dict(bundle.provenance)
    provenance.pop('sample_rate')
    with pytest.raises(ValueError, match='sample_rate'):
        deepunitmatch.prepare_deepunitmatch_training(units, replace(bundle, provenance=provenance), **kwargs)
    with pytest.raises(ValueError, match='integers'):
        deepunitmatch.prepare_deepunitmatch_training(units, bundle, autoencoder_epochs=1.5, **kwargs)
    assert not (tmp_path / 'out').exists()


def test_shared_voltage_signature_excludes_heldout_sources_but_rejects_mixed_training_profiles():
    _, bundle = synthetic_np1010()
    sources = [dict(well='training_well', provenance={'waveform_preprocessing':
               {'profile': 'shared_unwhitened_v1', 'highpass_hz': 150}}),
               dict(well='heldout_well', provenance={'waveform_preprocessing':
               {'profile': 'raw_no_filter'}})]
    bundle = replace(bundle, provenance=dict(bundle.provenance, sources=sources))
    assert deepunitmatch._deep_input_signature(bundle)['shared_voltage_preprocessing']['highpass_hz'] == 150
    sources.append(dict(well='training_well', provenance={'waveform_preprocessing':
                        {'profile': 'shared_unwhitened_v1', 'highpass_hz': 300}}))
    with pytest.raises(ValueError, match='inconsistent voltage'):
        deepunitmatch._deep_input_signature(bundle)


def test_checkpoint_signature_ignores_storage_only_and_detects_padding_scaling_and_unknown_transforms():
    import copy
    _, bundle = synthetic_np1010()
    source = dict(well='training_well', provenance=dict(voltage_scale=.195,
        common_median=True, baseline_samples=0, waveform_preprocessing=dict(
            profile='shared_unwhitened_v1', highpass_hz=150, filter_order=3,
            padding_seconds=.05, chunk_seconds=1., output_dtype='float32',
            storage='temporary snippets')))
    bundle = replace(bundle, provenance=dict(bundle.provenance, sources=[source]))
    expected = deepunitmatch._deep_input_signature(bundle)
    changed = copy.deepcopy(source)
    changed['provenance']['waveform_preprocessing']['storage'] = 'streaming summaries'
    candidate = replace(bundle, provenance=dict(bundle.provenance, sources=[source, changed]))
    assert deepunitmatch._deep_input_signature(candidate) == expected
    for key, value in (('padding_seconds', .1), ('output_dtype', 'float64'), ('new_filter', 'changed')):
        altered = copy.deepcopy(changed)
        altered['provenance']['waveform_preprocessing'][key] = value
        candidate = replace(bundle, provenance=dict(bundle.provenance, sources=[source, altered]))
        with pytest.raises(ValueError, match='inconsistent voltage'):
            deepunitmatch._deep_input_signature(candidate)
    changed['provenance']['voltage_scale'] = 1.
    candidate = replace(bundle, provenance=dict(bundle.provenance, sources=[source, changed]))
    with pytest.raises(ValueError, match='inconsistent voltage'):
        deepunitmatch._deep_input_signature(candidate)


@pytest.mark.skipif(not os.environ.get('UNITMATCH_SOURCE_CHECKOUT'),
                    reason='requires optional official source checkout and training dependencies')
def test_official_two_stage_synthetic_training_and_checkpoint_inference(tmp_path):
    units, bundle = synthetic_np1010()
    prepared = deepunitmatch.prepare_deepunitmatch_training(units, bundle,
        source_checkout=os.environ['UNITMATCH_SOURCE_CHECKOUT'], output_dir=tmp_path / 'synthetic',
        training_groups=['training_well'], evaluation_groups=['independent_evaluation_well'],
        autoencoder_epochs=2, contrastive_epochs=2, cpu_threads=1,
        seed=13)
    result = deepunitmatch.train_deepunitmatch(prepared)
    assert result['status'] == 'completed'
    assert Path(result['checkpoint']).is_file()
    import torch
    torch.set_num_threads(1)
    # Load the architecture actually trained by the official subprocess.
    model = deepunitmatch.load_deepunitmatch_checkpoint(result['checkpoint'])
    checkpoint = torch.load(result['checkpoint'], map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    import h5py
    data = next((prepared / 'data' / 'processed_waveforms').glob('*/*'))
    with h5py.File(data) as saved:
        waveform = torch.as_tensor(saved['waveform'][()][None, ..., 0], dtype=torch.float64)
    with torch.no_grad():
        embedding = model(waveform)
    assert tuple(embedding.shape) == (1, 256)
    assert torch.isfinite(embedding).all()
    ae_path = Path(result['autoencoder_checkpoint'])
    ae = torch.load(ae_path, map_location='cpu', weights_only=True)
    # Official stage 2 freezes convolution weights while updating encoder FC layers.
    assert torch.equal(ae['encoder']['ConvBlock1.conv1.weight'], checkpoint['model']['ConvBlock1.conv1.weight'])
    assert not torch.equal(ae['encoder']['FcBlock.0.weight'], checkpoint['model']['FcBlock.0.weight'])
    assert json.loads((prepared / 'training_manifest.json').read_text())['seed'] == 13
    assert result['checkpoint_sha256']
    assert result['preprocessing']['n_xchannelpos'] == 4
    assert result['execution']['device'] in ('cpu', 'cuda')
    assert [p.name for p in ae_path.parent.iterdir()] == ['ckpt_epoch_1']
    assert [p.name for p in Path(result['checkpoint']).parent.iterdir()] == ['ckpt_epoch_1']
    with pytest.raises(ValueError, match='overlap'):
        deepunitmatch._deep_training_metadata(result['checkpoint'], bundle, result['preprocessing'])
    inference_bundle = replace(bundle, provenance=dict(bundle.provenance,
        source_groups={s: 'independent_evaluation_well' for s in bundle.session_ids}))
    # Integration exercise only: synthetic training/inference reuse is not an accuracy benchmark.
    matched = deepunitmatch.run_deepunitmatch(units, waveform_bundle=inference_bundle,
        checkpoint=result['checkpoint'], checkpoint_probe='NP1010')
    assert len(matched.units) == 64
    assert np.isfinite(matched.pairs.match_score).all()
    assert matched.provenance.get('network_preprocessing')['n_xchannelpos'] == 4
    assert matched.provenance.get('training')['training_seed'] == 13
