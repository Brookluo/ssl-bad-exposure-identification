"""CCD anomaly scoring and top-K selection for multi-scale pipeline."""
import numpy as np
from decam_qa.info import ccdname2num

_CENTER_CCD = ccdname2num.get("S1", 25)
_EDGE_CCDS = {ccdname2num.get("S31", 3), ccdname2num.get("N31", 62)}


def compute_anomaly_scores(ccd_images, ccd_rows):
    """Compute cheap anomaly scores for each CCD from image statistics.

    Scores combine: background deviation, scatter, extreme pixel fraction,
    and non-finite fraction. Higher score = more anomalous.

    Parameters
    ----------
    ccd_images : list of np.ndarray
        List of CCD image arrays.
    ccd_rows : list of dict
        Metadata rows matching ccd_images. Must have 'ccdnum' key.

    Returns
    -------
    np.ndarray
        Score per CCD, shape (n_ccds,). Higher = more anomalous.
    """
    n = len(ccd_images)
    backgrounds = np.zeros(n)
    scatters = np.zeros(n)
    extreme_fracs = np.zeros(n)
    nonfinite_fracs = np.zeros(n)

    for i, img in enumerate(ccd_images):
        finite = img[np.isfinite(img)]
        if len(finite) == 0:
            backgrounds[i] = 0
            scatters[i] = 0
            extreme_fracs[i] = 0
            nonfinite_fracs[i] = 1.0
            continue

        backgrounds[i] = np.median(finite)
        scatters[i] = 1.4826 * np.median(np.abs(finite - backgrounds[i]))
        p99 = np.percentile(finite, 99.5)
        p01 = np.percentile(finite, 0.5)
        extreme_fracs[i] = np.mean((finite > p99) | (finite < p01))
        nonfinite_fracs[i] = 1.0 - len(finite) / img.size

    # Compute robust z-scores within the exposure
    def robust_z(values):
        med = np.median(values)
        mad = 1.4826 * np.median(np.abs(values - med))
        if mad == 0:
            return np.zeros_like(values)
        return (values - med) / mad

    scores = np.zeros(n)
    scores += np.clip(np.abs(robust_z(backgrounds)), 0, 5)
    scores += np.clip(np.abs(robust_z(scatters)), 0, 5)
    scores += np.clip(np.abs(robust_z(extreme_fracs)), 0, 5)
    scores += np.clip(np.abs(robust_z(nonfinite_fracs)), 0, 5)

    return scores


def select_top_k_ccds(scores, ccd_rows, k=8,
                      include_center_fallback=True,
                      include_edge_fallback=True):
    """Select top-K CCDs by anomaly score with optional fallback CCDs.

    Parameters
    ----------
    scores : np.ndarray
        Anomaly scores per CCD.
    ccd_rows : list of dict
        Metadata rows. Must have 'ccdnum' key.
    k : int
        Maximum number of CCDs to select.
    include_center_fallback : bool
        If True, always include one central CCD if not already selected.
    include_edge_fallback : bool
        If True, always include one edge CCD if not already selected.

    Returns
    -------
    list of dict
        Selected CCD rows, sorted by descending score.
    """
    if len(scores) == 0:
        return []

    # Sort by descending score
    order = np.argsort(scores)[::-1]
    effective_k = min(k, len(order))

    selected_indices = set(order[:effective_k].tolist())
    protected = set()

    # Protect the highest-scoring CCD from ever being evicted
    protected.add(order[0])

    def _lowest_evictable():
        """Return index of lowest-scoring selected item not in protected."""
        scored = [(scores[i], i) for i in selected_indices if i not in protected]
        if not scored:
            return None
        scored.sort()
        return scored[0][1]

    # Add center fallback
    if include_center_fallback and len(selected_indices) > 0:
        has_center = any(ccd_rows[i]["ccdnum"] == _CENTER_CCD for i in selected_indices)
        if not has_center:
            for i, row in enumerate(ccd_rows):
                if row["ccdnum"] == _CENTER_CCD and i not in selected_indices:
                    victim = _lowest_evictable()
                    if victim is not None:
                        selected_indices.discard(victim)
                    selected_indices.add(i)
                    protected.add(i)
                    break

    # Add edge fallback
    if include_edge_fallback and len(selected_indices) > 0:
        has_edge = any(ccd_rows[i]["ccdnum"] in _EDGE_CCDS for i in selected_indices)
        if not has_edge:
            for i, row in enumerate(ccd_rows):
                if row["ccdnum"] in _EDGE_CCDS and i not in selected_indices:
                    victim = _lowest_evictable()
                    if victim is not None:
                        selected_indices.discard(victim)
                    selected_indices.add(i)
                    protected.add(i)
                    break

    result = []
    for i in order:
        if i in selected_indices:
            entry = dict(ccd_rows[i])
            entry["selection_score"] = float(scores[i])
            result.append(entry)
            selected_indices.discard(i)

    return result
