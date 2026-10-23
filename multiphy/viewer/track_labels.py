"""Qt-free labels separating original backend assignment, current track and review."""
from multiphy.clusters import cluster_id


def backend_assignments(result) -> dict:
    """Original multi-unit track id for each assigned unit; unmatched units are absent."""
    return {str(unit): cluster_id(track)
            for track in (getattr(result, 'tracks', None) or ()) if len(track) > 1
            for unit in track}


def short_track_id(track_id: str) -> str:
    """Stable compact track label; callers retain the full identifier in a tooltip."""
    return str(track_id).removeprefix('uc_')[:8]


def review_label(track: dict) -> str:
    """A track's reviewer decision, independent of its membership and backend origin."""
    return 'accepted' if track.get('status') == 'accepted' else 'unreviewed'
