"""The viewer's keyboard shortcuts: one key per action, saved per user.

``ACTIONS`` holds one ``(id, label, default, hover)`` per action. A key the user sets for action ``id`` is saved
as key ``shortcuts/<id>`` of the user's settings file ``~/.config/multiphy/multiphy.ini`` (``theme.settings()``);
only ``save_binding`` and ``reset_bindings`` write those keys. ``bindings()`` reads them and never raises on a
saved value: a missing, malformed or invalid key (not a string, or not one key) is the action's default, an empty
one leaves the action unbound, and a key an earlier action already holds leaves the later action unbound. A save or
a reset emits ``notifier().changed``, and every window of this process rebinds.

Two processes saving shortcuts at once: the last save wins, a clash falls back on read, and only the saving
process rebinds at once; the others rebind on their next start.
"""
from multiphy.viewer import theme
from multiphy.viewer.qt import QtCore, QtGui

# (id, label, default key, hover), in the order bindings() reads them and an earlier action wins a clash
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
    ('skip', 'drop day', 'X', "drops the active day's strung unit; days beyond sort to the day before"),
    ('accept', 'accept pair', 'E', 'saves accept for the shown pair, then steps down'),
    ('reject', 'reject pair', 'R', 'saves reject for the shown pair, then steps down'),
    ('unsure', 'unsure pair', 'U', 'saves unsure for the shown pair, then steps down'),
    ('merge', 'merge', 'Q', 'joins the tracks of the selected units'),
    ('accept_track', 'accept track', 'Shift+E', "accepts the picked unit's track"),
    ('reject_track', 'reject track', 'Shift+R', "rejects the picked unit's track"),
    ('unmatch', 'unmatch unit', 'Shift+X', 'takes the active unit out of its track'),
    ('undo', 'undo', 'Ctrl+Z', 'undoes the last curation action'),
    ('redo', 'redo', 'Ctrl+Shift+Z', 'redoes the last undone action'),
)

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
    (unbound); a value ``normalized`` reads as ``''`` is the default; else the normalized key. A key an earlier
    action already holds is ``''``. Never raises on a saved value.
    """
    s = theme.settings()
    out, taken = {}, set()
    for aid, _label, default, _hover in ACTIONS:
        value = s.value('shortcuts/' + aid)
        if not isinstance(value, str):
            key = default
        elif value == '':
            key = ''
        else:
            key = normalized(value) or default
        if key in taken:
            key = ''
        out[aid] = key
        if key:
            taken.add(key)
    return out


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
    s.sync()
    notifier().changed.emit()
    return _save_tag(s)


def reset_bindings() -> str:
    """Remove every saved key, so each action takes its default, then emit ``notifier().changed``; ``''``, or a
    tag naming the file when the save fails."""
    s = theme.settings()
    s.remove('shortcuts')
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
