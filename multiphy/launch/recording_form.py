"""Copy and read-only checks for the launcher's recording form.

The CLI remains the authority when a command runs. These checks inspect only paths,
``params.py``, import dates and the project manifest; no voltage, arrays or hashes are read.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
from typing import Optional

from ..project.folder import MANIFEST_FILE, NAME_PATTERN, RESERVED_NAMES, Project
from ..sources.phy import READER as PHY_READER


INIT_MODES = ('manual', 'import')
SHARED_FIELDS = ('project', 'probe', 'uv_per_bit')
MANUAL_FIELDS = ('day', 'date', 'sort', 'events', 'raw')
IMPORT_FIELDS = ('from_dir',)

FIELD_COPY = {
    'project': {
        'label': 'Project', 'placeholder': 'e.g. mouse01_probeA',
        'help': 'Choose an existing project or enter a new name. The project is stored under the selected root.',
        'hint': '',
    },
    'day': {
        'label': 'Recording name', 'placeholder': 'e.g. baseline or D01',
        'help': 'A unique name within this project. Start with a letter or number; use letters, numbers, underscores, dots or hyphens.',
        'hint': 'A name you will recognize in the day browser.',
    },
    'date': {
        'label': 'Recording date', 'placeholder': 'YYYY-MM-DD',
        'help': 'The acquisition date, written as YYYY-MM-DD. Each project currently allows one recording per calendar date.',
        'hint': 'Acquisition date; one recording per date in each project.',
    },
    'sort': {
        'label': 'Phy/Kilosort folder', 'placeholder': 'Choose the sorting output folder',
        'help': 'The sorting output folder containing params.py, spike_times.npy, spike_clusters.npy, channel_positions.npy and cluster labels.',
        'hint': 'Choose the folder containing params.py.',
    },
    'events': {
        'label': 'Events CSV', 'placeholder': 'Optional events file',
        'help': 'Optional event timing CSV for this recording. Leave blank if there is no events file.',
        'hint': '',
    },
    'raw': {
        'label': 'Raw voltage file', 'placeholder': 'Auto-detect from params.py, or choose a file',
        'help': 'Optional flat, interleaved binary voltage file. If blank, the dat_path in params.py is used when the file exists and a voltage scale is supplied. Use the acquired stream, not Kilosort temp_wh.dat.',
        'hint': 'Leave blank to use the file named in params.py.',
    },
    'uv_per_bit': {
        'label': 'Voltage scale (µV/count)', 'placeholder': 'From acquisition metadata',
        'help': ('Microvolts per stored sample count; required to include raw voltage.\n\n'
                 'Open Ephys Binary recordings:\n'
                 '1. Open structure.oebin in the recording folder.\n'
                 '2. Find the continuous stream matching your raw file.\n'
                 '3. In its channels, check bit_volts and units.\n\n'
                 'Headstage values usually convert counts to µV; ADC values may convert to volts.\n'
                 'Multiply a volts/count value by 1,000,000 to obtain µV/count.\n'
                 'Use a scale that applies to every channel in the chosen file; no value is assumed.\n'
                 'For an import, this scale applies to every recording.'),
        'hint': 'Open Ephys: structure.oebin → channels → bit_volts + units. Hover for details.',
    },
    'from_dir': {
        'label': 'Import folder', 'placeholder': 'Folder containing recording subfolders',
        'help': 'Each recording subfolder must contain sort/ and date.txt (YYYY-MM-DD), with optional events.csv and one raw.* file. Subfolder names become recording names.',
        'hint': 'Each subfolder: sort/ + date.txt; optional events.csv and raw.*.',
    },
    'probe': {
        'label': 'Probe type', 'placeholder': 'Optional; e.g. Neuropixels 1.0',
        'help': 'Optional probe type, recorded once for the project. It must agree with the existing project when adding more recordings.',
        'hint': '',
    },
}


@dataclass(frozen=True)
class InitValidation:
    errors: dict
    raw_message: str
    raw_level: str = 'info'
    detected_raw: Optional[str] = None

    @property
    def ready(self) -> bool:
        return not self.errors

    @property
    def message(self) -> str:
        return next(iter(self.errors.values()), '')


def init_values(values: dict, mode: str = 'manual') -> dict:
    """Trim form text and omit the other mode's arguments, including stale hidden values."""
    if mode not in INIT_MODES:
        raise ValueError(f'Unknown recording mode: {mode}')
    active = SHARED_FIELDS + (MANUAL_FIELDS if mode == 'manual' else IMPORT_FIELDS)
    return {dest: str(values.get(dest) or '').strip() for dest in active}


def _date(text: str) -> Optional[datetime.date]:
    if not re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', text):
        return None
    try:
        return datetime.date.fromisoformat(text)
    except ValueError:
        return None


def _project(root: Path, values: dict, errors: dict) -> dict:
    name = values['project']
    if not name:
        errors['project'] = 'Enter a project name.'
        return {}
    try:
        name = Project._name_under(root, name)
        if name in RESERVED_NAMES:
            raise ValueError(f'{name} is a reserved project name.')
        candidate = Project(root, name)
        if not candidate.path.exists():
            return {}
        if not (candidate.path / MANIFEST_FILE).is_file():
            raise ValueError('The project folder already exists but has no project.json; choose a different name.')
        manifest = candidate.read()
        probe = values['probe']
        recorded_probe = manifest['identity']['probe']
        if probe and recorded_probe is not None and probe != recorded_probe:
            errors['probe'] = f'This project records probe type {recorded_probe}; use the same type or leave blank.'
        return manifest['days']
    except (OSError, ValueError, TypeError, KeyError) as exc:
        errors['project'] = str(exc)
        return {}


def _name_date(values: dict, recorded: dict, errors: dict) -> None:
    name, date = values['day'], values['date']
    if not name:
        errors['day'] = 'Enter a recording name.'
    elif not NAME_PATTERN.fullmatch(name):
        errors['day'] = 'Start the recording name with a letter or number; use only letters, numbers, _, . or -.'
    elif name in recorded:
        errors['day'] = f'Recording {name} already exists in this project; choose a unique name.'
    if not date:
        errors['date'] = 'Enter the acquisition date as YYYY-MM-DD.'
    elif _date(date) is None:
        errors['date'] = 'Enter a valid recording date as YYYY-MM-DD.'
    else:
        duplicate = next((day for day, record in recorded.items() if record['date'] == date), None)
        if duplicate is not None:
            errors['date'] = f'{date} is already used by {duplicate}; a project allows one recording per date.'


def _scale(text: str, errors: dict) -> Optional[float]:
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or value <= 0:
        errors['uv_per_bit'] = 'Enter a positive, finite voltage scale in µV/count.'
        return None
    return value


def _inputs(values: dict, scale: Optional[float], errors: dict) -> tuple:
    """Validate paths and return ``(message, level, detected_path)`` without reading arrays."""
    sort, explicit_raw = values['sort'], values['raw']
    detected = None
    sort_ok = False
    if not sort:
        errors['sort'] = 'Choose the Phy/Kilosort output folder.'
    elif not Path(sort).is_dir():
        errors['sort'] = 'The Phy/Kilosort folder does not exist.'
    elif not (Path(sort) / 'params.py').is_file():
        errors['sort'] = 'Choose a sorting output folder containing params.py.'
    else:
        try:
            options = {'sort_dir': str(Path(sort).absolute()), 'day': values['day']}
            detected = PHY_READER.dat_path(options)
            PHY_READER.files(options)
            sort_ok = True
        except (OSError, ValueError, TypeError, OverflowError) as exc:
            errors['sort'] = str(exc)
    if values['events'] and not Path(values['events']).is_file():
        errors['events'] = 'The events CSV does not exist.'
    if explicit_raw and not Path(explicit_raw).is_file():
        errors['raw'] = 'The raw voltage file does not exist.'
    if explicit_raw and scale is None:
        errors.setdefault('uv_per_bit', 'Enter the voltage scale to include the selected raw file.')
    raw = explicit_raw or (detected if detected and Path(detected).is_file() else None)
    if raw and scale is not None and sort_ok:
        try:
            PHY_READER.raw_layout(options)
        except (OSError, ValueError, TypeError, OverflowError) as exc:
            errors['sort'] = str(exc)
    if explicit_raw:
        if 'raw' in errors or 'uv_per_bit' in errors:
            return 'Raw voltage needs an existing file and its voltage scale before it can be added.', 'error', detected
        return f'Raw voltage will be included: {explicit_raw}', 'info', detected
    if detected and Path(detected).is_file():
        if scale is None:
            return (f'Detected in params.py: {detected}\n'
                    'No voltage scale: this recording will be added with sorting data only; raw voltage will be excluded.',
                    'warning', detected)
        return f'Detected in params.py; raw voltage will be included: {detected}', 'info', detected
    if detected:
        return (f'params.py names a missing raw file: {detected}\n'
                'Choose its current location to include raw voltage, or continue with sorting data only.', 'warning', detected)
    if sort_ok:
        return 'No raw file detected. This recording will be added with sorting data only.', 'info', None
    return 'Choose a sorting folder to preview raw voltage detection.', 'info', None


def _import(values: dict, recorded: dict, scale: Optional[float], errors: dict) -> InitValidation:
    text = values['from_dir']
    if not text:
        errors['from_dir'] = 'Choose the folder containing recording subfolders.'
        return InitValidation(errors, 'Choose an import folder to preview its recordings.')
    folder = Path(text)
    if not folder.is_dir():
        errors['from_dir'] = 'The import folder does not exist.'
        return InitValidation(errors, 'Choose an existing import folder.')
    try:
        children = sorted(path for path in folder.iterdir() if path.is_dir())
        if not children:
            errors['from_dir'] = 'The import folder contains no recording subfolders.'
        planned = dict(recorded)
        sorting_only = 0
        for child in children:
            if not (child / 'date.txt').is_file():
                errors['from_dir'] = f'{child.name}: missing date.txt (YYYY-MM-DD).'
                break
            raws = sorted(path for path in child.glob('raw.*') if not path.is_dir())
            if len(raws) > 1:
                errors['from_dir'] = f'{child.name}: more than one raw.* file; keep one acquired voltage file.'
                break
            item = dict(values, day=child.name, date=(child / 'date.txt').read_text(encoding='utf-8').strip(),
                        sort=str(child / 'sort'), raw=str(raws[0]) if raws else '',
                        events=str(child / 'events.csv') if os.path.lexists(child / 'events.csv') else '')
            item_errors = {}
            _name_date(item, planned, item_errors)
            _message, _level, detected = _inputs(item, scale, item_errors)
            if item_errors:
                first_field = next(iter(item_errors))
                field = 'uv_per_bit' if first_field == 'uv_per_bit' else 'from_dir'
                errors.setdefault(field, f'{child.name}: {item_errors[first_field]}')
                break
            planned[child.name] = {'date': item['date']}
            if scale is None or not (raws or (detected and Path(detected).is_file())):
                sorting_only += 1
        if errors:
            return InitValidation(errors, 'Resolve the highlighted fields to import these recordings.', 'error')
        count = len(children)
        message = f'{count} recording' + ('' if count == 1 else 's') + ' ready to import.'
        if sorting_only:
            message += f' {sorting_only} will use sorting data only; raw voltage will be excluded.'
        else:
            message += ' Raw voltage will be included for every recording.'
        return InitValidation(errors, message, 'warning' if sorting_only else 'info')
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        errors['from_dir'] = str(exc)
        return InitValidation(errors, 'The import folder could not be inspected.', 'error')


def validate_init(root: Path, values: dict, mode: str = 'manual') -> InitValidation:
    """Return field errors and a raw-data preview without creating or changing anything."""
    values = init_values(values, mode)
    errors: dict = {}
    recorded = _project(Path(root), values, errors)
    scale = _scale(values['uv_per_bit'], errors)
    if mode == 'import':
        return _import(values, recorded, scale, errors)
    _name_date(values, recorded, errors)
    try:
        message, level, detected = _inputs(values, scale, errors)
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        errors['sort'] = str(exc)
        message, level, detected = 'The recording paths could not be inspected.', 'error', None
    return InitValidation(errors, message, level, detected)
