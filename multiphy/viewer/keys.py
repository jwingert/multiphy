"""The viewer's keyboard shortcuts: one key per action, saved per user.

``ACTIONS`` holds one ``(id, label, default, hover)`` per action. A key the user sets for action ``id`` is saved
as key ``shortcuts/<id>`` of the user's settings file ``~/.config/multiphy/multiphy.ini`` (``theme.settings()``);
only ``save_binding`` and ``reset_bindings`` write those keys. ``bindings()`` reads them and never raises on a
saved value: a missing, malformed or invalid key (not a string, or not one key) is the action's default, an empty
one leaves the action unbound. Explicit custom bindings take priority over defaults; otherwise the earlier
action wins a clash. Saved old defaults follow the revised track-first defaults without modifying settings. A save or
a reset emits ``notifier().changed``, and every window of this process rebinds.

Two processes saving shortcuts at once: the last save wins, a clash falls back on read, and only the saving
process rebinds at once; the others rebind on their next start.
"""
from multiphy.viewer import theme
from multiphy.viewer.qt import QtCore, QtGui

# (id, label, default key, hover); within custom bindings or defaults, earlier actions win clashes.
ACTIONS = (
    ('up', 'up', 'W',
     "the unit above in the active day; in the picked unit's day it becomes the pick, or replaces the extra last "
     'clicked'),
    ('down', 'down', 'S',
     "the unit below in the active day; in the picked unit's day it becomes the pick, or replaces the extra last "
     'clicked'),
    ('left', 'previous day', 'A', 'the day to the left becomes the active day'),
    ('right', 'next day', 'D', 'the day to the right becomes the active day; its top unit is strung when none is'),
    ('track', 'string track', 'F', "strings the picked unit's current track"),
    ('clear', 'clear string', 'Esc', 'back to the picked unit alone'),
    ('merge', 'match selected tracks', 'Q',
     'joins the whole tracks of the selected units; changed membership needs review again'),
    ('accept_track', 'accept track', 'E', "approves the picked unit's current track for export; does not advance"),
    ('reject_track', 'reject track', 'R', "rejects the picked unit's association and releases its units as unmatched"),
    ('unmatch', 'unmatch unit', 'X', 'detaches the active unit; changed groups need review again'),
    ('skip', 'drop day from comparison', 'Shift+X',
     "drops the active day's selected unit from the comparison only; does not change its track"),
    ('accept', 'pair same (advanced)', 'Alt+E',
     'Pair review (advanced) only: saves a same-neuron pair verdict, then steps down; does not approve a track'),
    ('reject', 'pair different (advanced)', 'Alt+R',
     'Pair review (advanced) only: saves a different-neuron pair verdict, then steps down; does not unmatch units'),
    ('unsure', 'pair unsure (advanced)', 'Alt+U',
     'Pair review (advanced) only: saves an unsure pair verdict, then steps down; does not change export eligibility'),
    ('undo', 'undo', 'Ctrl+Z', 'undoes the last curation action'),
    ('redo', 'redo', 'Ctrl+Shift+Z', 'redoes the last undone action'),
)

# Before the track-first workflow, these defaults overlapped the new primary controls. Values written by
# this version are tagged separately so deliberately assigning a former default remains possible.
LEGACY_DEFAULTS = {'skip': 'X', 'accept': 'E', 'reject': 'R', 'unsure': 'U',
                   'accept_track': 'Shift+E', 'reject_track': 'Shift+R', 'unmatch': 'Shift+X'}

_notifier = None        # the one KeyNotifier


def normalized(key: str) -> str:
    """``key`` as one key in Qt's portable text (``'l'`` → ``'L'``, ``'shift+e'`` → ``'Shift+E'``, ``'Escape'``
    → ``'Esc'``); ``''`` for a key that is not a ``str`` (``QKeySequence(76)`` would read as ``L``), and for an
    empty key, a sequence of two or more keys or a key Qt cannot name (``'Foo+'``, ``'12'``, ``'Ctrl+'``)."""
    if not isinstance(key, str):
        return ''
    sequence = QtGui.QKeySequence(key)
    text = sequence.toString(QtGui.QKeySequence.PortableText)
    return text if sequence.count() == 1 and text else ''


def bindings() -> dict:
    """Action id → its key, in ``ACTIONS`` order, from the saved ``shortcuts/<id>`` keys.

    No saved value, or one that is not a ``str`` (a saved list included), is the default; ``''`` is ``''``
    (unbound); a value ``normalized`` reads as ``''`` is the default. Old saved defaults follow the new
    defaults, unless deliberately saved by this version. Custom values take priority over defaults; a later
    conflicting action in either group is unbound. No settings are written here.
    """
    s = theme.settings()
    candidates, custom = {}, []
    for aid, _label, default, _hover in ACTIONS:
        value = s.value('shortcuts/' + aid)
        if not isinstance(value, str):
            key = default
        elif value == '':
            key = ''
        else:
            key = normalized(value) or default
        legacy = (key == LEGACY_DEFAULTS.get(aid) and
                  s.value('shortcut_versions/' + aid) not in (2, '2'))
        if legacy:
            key = default
        elif isinstance(value, str) and normalized(value) and key != default:
            custom.append(aid)
        candidates[aid] = key
    out, taken = {}, set()
    for aid in custom + [aid for aid in candidates if aid not in custom]:
        key = candidates[aid]
        if key in taken:
            key = ''
        out[aid] = key
        if key:
            taken.add(key)
    return {aid: out[aid] for aid in candidates}


def _save_tag(s: QtCore.QSettings) -> str:
    """``''``, or ``shortcuts not saved · <file>`` when ``s`` failed to write."""
    if s.status() != QtCore.QSettings.NoError:
        return f'shortcuts not saved · {s.fileName()}'
    return ''


def save_binding(action_id: str, key: str) -> str:
    """Save ``key`` for ``action_id`` (``''`` unbinds it), then emit ``notifier().changed``; ``''``, or a tag
    naming the file when the save fails.

    Refused with ``ValueError``, in this order: an unknown action (``<id> · no such action``), a key that is not
    one key (``<key> · not one key``), a key another action holds (``<key> · taken by <label>``). A refusal saves
    and emits nothing.
    """
    labels = {aid: label for aid, label, _default, _hover in ACTIONS}
    if action_id not in labels:
        raise ValueError(f'{action_id} · no such action')
    if key == '':
        k = ''
    else:
        k = normalized(key)
        if k == '':
            raise ValueError(f'{key} · not one key')
        holder = next((aid for aid, bound in bindings().items() if bound == k and aid != action_id), None)
        if holder is not None:
            raise ValueError(f'{k} · taken by {labels[holder]}')
    s = theme.settings()
    s.setValue('shortcuts/' + action_id, k)
    s.setValue('shortcut_versions/' + action_id, 2)
    s.sync()
    notifier().changed.emit()
    return _save_tag(s)


def reset_bindings() -> str:
    """Remove every saved key, so each action takes its default, then emit ``notifier().changed``; ``''``, or a
    tag naming the file when the save fails."""
    s = theme.settings()
    s.remove('shortcuts')
    s.remove('shortcut_versions')
    s.sync()
    notifier().changed.emit()
    return _save_tag(s)


class KeyNotifier(QtCore.QObject):
    """Emits ``changed`` after a save or a reset of the shortcuts."""
    changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)


def notifier() -> KeyNotifier:
    """The one ``KeyNotifier``, made on first call."""
    global _notifier
    if _notifier is None:
        _notifier = KeyNotifier()
    return _notifier
