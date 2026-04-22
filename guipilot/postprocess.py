"""
Postprocessing filters for screen-inconsistency predictions.

These are unsupervised adaptations of the RQ1 filters: instead of using ground-
truth y_true to locate the "anchor" mutated widget, we derive the anchor from the
predictions themselves (missing / excess entries) and use spatial overlap to
suppress noise in nearby self-matched widget pairs.
"""

import copy

from guipilot.entities import Inconsistency, Screen


def _overlaps(b1, b2) -> bool:
    """Return True if two [xmin, ymin, xmax, ymax] bboxes intersect."""
    return not (b1[2] < b2[0] or b2[2] < b1[0] or b1[3] < b2[1] or b2[3] < b1[1])


def filter_overlapping_noise(y_pred: set, s1: Screen, s2: Screen) -> set:
    """Remove self-matched inconsistency flags caused by nearby missing/excess widgets.

    When a widget is missing from real (or excess in real), the widget detector
    and matcher may shift widget assignments for spatially adjacent widgets.
    This produces spurious BBOX / TEXT / COLOR flags on self-matched pairs
    (A, A, TYPE) whose bounding box overlaps with the anchor missing/excess widget.

    Adapts RQ1's filter_overlap_predictions to work without y_true: the
    "mutated" anchor IDs are inferred from (id, None) and (None, id) entries in
    y_pred rather than read from ground truth.
    """
    y_pred = copy.deepcopy(y_pred)

    # Anchor widgets: those reported as missing (in s1) or excess (in s2)
    missing_ids = {item[0] for item in y_pred if item[1] is None}   # id1 in s1
    excess_ids  = {item[1] for item in y_pred if item[0] is None}   # id2 in s2

    noisy_ids: set[int] = set()

    for item in y_pred:
        id1, id2 = item[0], item[1]
        if id1 is None or id2 is None or id1 != id2:
            continue  # only consider self-matched pairs
        if id1 in missing_ids:
            continue  # this is the anchor itself, not noise

        # Check overlap with missing anchors in s1
        for mid in missing_ids:
            if _overlaps(s1.widgets[id1].bbox, s1.widgets[mid].bbox):
                noisy_ids.add(id1)
                break

        if id1 in noisy_ids:
            continue

        # Check overlap with excess anchors in s2
        for eid in excess_ids:
            if id2 in s2.widgets and eid in s2.widgets:
                if _overlaps(s2.widgets[id2].bbox, s2.widgets[eid].bbox):
                    noisy_ids.add(id2)
                    break

    for nid in noisy_ids:
        y_pred.discard((nid, nid, Inconsistency.BBOX))
        y_pred.discard((nid, nid, Inconsistency.TEXT))
        y_pred.discard((nid, nid, Inconsistency.COLOR))

    return y_pred


def consolidate_swaps(y_pred: set, s1: Screen, s2: Screen) -> set:
    """Consolidate (A, None) + (None, B) pairs into swap entries when A and B overlap.

    A swap appears in raw predictions as a missing widget on one side and an
    excess on the other, often accompanied by a self-BBOX flag on the "shifted"
    widget.  When the missing widget's bbox in s1 overlaps with the excess
    widget's bbox in s2, the pair is almost certainly a positional swap rather
    than a true insertion + deletion.

    Adapts the swap-consolidation logic from RQ1's filter_swapped_predictions,
    but discovers swap candidates from spatial overlap instead of y_true.
    """
    y_pred = copy.deepcopy(y_pred)

    missing = [(item[0],) for item in y_pred if item[1] is None]   # [(id1,), ...]
    excess  = [(item[1],) for item in y_pred if item[0] is None]   # [(id2,), ...]

    for (mid,) in missing:
        for (eid,) in excess:
            if mid not in s1.widgets or eid not in s2.widgets:
                continue
            if not _overlaps(s1.widgets[mid].bbox, s2.widgets[eid].bbox):
                continue

            # Spatial overlap → treat as swap
            y_pred.discard((mid, None))
            y_pred.discard((None, eid))
            # Also remove any self-BBOX flags that the checker emitted for
            # either participant (same pattern as RQ1's alternative1/alternative2)
            y_pred.discard((mid, mid, Inconsistency.BBOX))
            y_pred.discard((eid, eid, Inconsistency.BBOX))

            # Only emit the (mock_id, real_id) direction — id1 is always a mock
            # widget index, id2 is always a real widget index throughout run_checks.
            y_pred.add((mid, eid, Inconsistency.BBOX))

    return y_pred


def postprocess(y_pred: set, s1: Screen, s2: Screen) -> set:
    """Apply all postprocessing filters in order."""
    y_pred = consolidate_swaps(y_pred, s1, s2)
    y_pred = filter_overlapping_noise(y_pred, s1, s2)
    return y_pred
