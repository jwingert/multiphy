"""Every human-readable name, keyed by the internal id it explains.

Reports, the viewer and the CLI read from here and nowhere else. ``display`` raises on
an unknown key so an unexplained internal code can never reach a figure or a table.
Backend ids are the exception: ``backend_label``/``backend_short``/``backend_help`` fall
back to the id (empty help), since any installed backend may lack an entry.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Name:
    label: str
    short: str = ''
    help: str = ''


NAMES = {
    'backend': {
        'functional_likelihood': Name('Functional likelihood — firing, waveform and optional event-response likelihood ratios',
                                      'Functional likelihood',
                                      'Likelihood ratios from own halves (trials in event runs, else 5 s chunks) and same-day '
                                      'neighbours; a mixture posterior per day pair.'),
        'pair_logistic': Name('Pair logistic — calibrated classifier per project', 'Pair logistic',
                              'Logistic model on half-split pairs of alternate sessions, calibrated on the held-out sessions.'),
        'unitmatch': Name('UnitMatch — spike waveform and spatial shape', 'UnitMatch'),
        'dant': Name('DANT — waveform, spike timing, optional stimulus response', 'DANT'),
        'deepunitmatch': Name('DeepUnitMatch — learned waveform features', 'DeepUnitMatch'),
    },
    'tracking': {  # move TRACKING_NAMES + TRACKING_HELP from names.py and TRACKING from reporting_names.py here
        'native_intermediate': Name('Published rule: intermediate grouping', 'native intermediate tracking',
                                    'Intermediate: for each proposed link, both observations must also match members of the other track in the same or adjacent temporal parts.'),
        'native_conservative': Name('Published rule: conservative grouping', 'native conservative tracking',
                                    'Conservative: for each proposed link, both observations must also match every member of the other track, including distant temporal parts.'),
        'native_liberal': Name('Published rule: liberal grouping', 'native liberal tracking',
                               'Liberal: any qualifying pair can join two tracks; other members do not need direct supporting links.'),
        'common_constrained': Name('Our rule: one observation per part, with conflict checks', 'common constrained tracking',
                                   'Our rule: mutually preferred candidates join tracks with at most one observation per temporal part, ambiguity checks and rejection of strong measured conflicts.'),
        'mutual_margin_best_first': Name('Our rule: one observation per part, with conflict checks', 'common constrained tracking'),
        'common_constrained_operating_point': Name('Common tracking with a calibration-selected cutoff', 'common tracking with a calibration-selected cutoff'),
        'native_DANT': Name('DANT native clustering', 'native DANT clustering'),
    },
    'evidence': {
        'combined': Name('Stimulus fingerprint, firing physiology, waveform and location'),
        'stimulus': Name('Stimulus fingerprint and shared-stimulus firing rate'),
        'physiology': Name('Firing physiology, waveform and location'),
    },
    'metric': {  # reporting_names.METRICS wording
        'true_positive_pairs': Name('Correct same-neuron pairs'),
        'false_positive_pairs': Name('Incorrect same-neuron pairs'),
        'false_negative_pairs': Name('Missed same-neuron pairs'),
        'pair_precision': Name('Pair precision'),
        'pair_recall': Name('Pair recall'),
        'n_unmatched': Name('Reported unmatched observations'),
        'correctly_unmatched_units': Name('Correctly unmatched observations'),
        'true_unmatched_units': Name('Known unmatched observations'),
        'unmatched_precision': Name('Unmatched precision'),
        'unmatched_recall': Name('Unmatched recall'),
        'candidate_positive_coverage': Name('Known same-neuron pairs with measured model evidence'),
        'merged_tracks': Name('Tracks containing different known source neurons'),
        'split_source_identities': Name('Source neuron identities divided between predicted tracks'),
        'n_known_pairs': Name('Evaluated pairs within original recordings'),
        'n_units': Name('Evaluated single-unit temporal observations'),
        'brier_score': Name('Mean squared probability error on scored test pairs'),
    },
    'feature': {  # reporting_names.FEATURES
        'Waveform': Name('Waveform'),
        'AutoCorr': Name('Autocorrelogram'),
        'PETH': Name('Stimulus response fingerprint'),
        'ISI': Name('Interspike interval distribution'),
    },
    'dant_preset': {  # names.DANT_PRESET_NAMES
        'waveform_acg': Name('Waveform + spike timing'),
        'waveform_acg_stimulus': Name('Waveform + spike timing + stimulus fingerprint'),
        'upstream_example': Name('Published example: fingerprint for motion estimation only'),
    },
    'definitions': {  # reporting_names.DEFINITIONS, each Name(text)
        'reference': Name('A sorted unit is a spike cluster labeled as a single neuron by the original sorter and its curation. A temporal observation is that cluster recomputed over one contiguous time segment. A known same-neuron pair consists of two observations with the same original cluster label; “source neuron” means this conditional identity, not an independently verified biological neuron.'),
        'directions': Name('The two UnitMatch directions compare the first internal waveform half of observation A with the second internal half of B, and the first half of B with the second half of A. The displayed score is the smaller of those two Bayesian outputs; it is dimensionless and is not independently calibrated identity accuracy.'),
        'intermediate': Name("Native intermediate tracking first requires both directional scores to exceed the recorded threshold, processes eligible pairs in descending mean score, and joins groups only when the candidate observations agree with every group member in the same or adjacent temporal session. When a native group holds two or more observations from one session, the adapter keeps the one with the highest mean native score to the group's observations from other sessions (missing scores are skipped; a tie goes to more spikes, then the smaller cell ID); the others stay unmatched, and every pair inside the group that is not two members of one track is saved with decision proposed_link. Thus a pair score alone does not describe the full grouping decision."),
        'dant': Name('DANT means Density-based Across-day Neuron Tracking. It estimates depth drift (apparent movement along the probe), compares waveform shape and spike-timing evidence, and groups observations into proposed neuron tracks by density-based clustering. Its autocorrelogram describes how often pairs of spikes from one observation occur at each time lag. The event-response fingerprint is the concatenated mean firing rate in matching event conditions and event-relative time bins.'),
        'dant_weights': Name('These dimensionless signed coefficients are fitted by linear discriminant analysis to DANT’s own provisional cluster labels, then divided by their sum. The combined similarity is the sum of coefficient × feature similarity. Each feature similarity is a Fisher transform, atanh(r), of a Pearson correlation clipped to ±tanh(6); waveform similarity uses the larger value from the two units’ local peak-electrode neighborhoods, then the larger value across templates. No unit-variance standardization precedes this discriminant fit. Negative coefficients can subtract correlated evidence; they are not fractions of explanatory importance or probabilities.'),
        'dant_motion': Name('Motion values are per-session depth offsets in micrometers, ordered by the native session order, not an x/y coordinate vector. Positive motion is subtracted from measured electrode-depth coordinates for distance gating. Constant-only fitted offsets have zero mean across sessions; zero at initialization denotes no estimated displacement. They are not absolute anatomical positions.'),
        'dant_caveat': Name('The pinned upstream early-stop code can use an earlier waveform correction while the final spatial-distance gate uses the last saved motion estimate. These two stages can therefore disagree about drift. The adapter preserves this upstream behavior; the effect on individual matches has not been isolated.'),
    },
}


def display(namespace, key) -> str:
    """Fail rather than publish an unexplained internal code."""
    return NAMES[namespace][key].label


def short(namespace, key) -> str:
    n = NAMES[namespace][key]
    return n.short or n.label


def help_text(namespace, key) -> str:
    return NAMES[namespace][key].help


def backend_label(backend_id) -> str:
    """The backend's label, else its id: a plugin or test backend need not add a NAMES entry."""
    name = NAMES['backend'].get(backend_id)
    return name.label if name is not None else str(backend_id)


def backend_short(backend_id) -> str:
    name = NAMES['backend'].get(backend_id)
    return (name.short or name.label) if name is not None else str(backend_id)


def backend_help(backend_id) -> str:
    name = NAMES['backend'].get(backend_id)
    return name.help if name is not None else ''
