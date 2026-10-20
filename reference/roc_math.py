"""Verbatim ROC functions extracted from the original paper analysis script."""
import numpy as np

def area(x, y):
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2.0))

def roc_binary(positive, scores):
    """All unique score thresholds, with ties grouped and an initial (0,0) vertex."""
    positive = np.asarray(positive, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind='stable')
    sorted_scores, truth = scores[order], positive[order]
    last_at_threshold = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores)-1]
    tp = np.cumsum(truth, dtype=np.int64)[last_at_threshold]
    fp = last_at_threshold + 1 - tp
    fpr = np.r_[0.0, fp / np.count_nonzero(~positive)]
    tpr = np.r_[0.0, tp / np.count_nonzero(positive)]
    threshold = np.r_[np.inf, sorted_scores[last_at_threshold]]
    auc_value = area(fpr, tpr)
    # Independent Mann-Whitney concordance check, including half-credit for ties.
    pos = scores[positive, None]
    neg = scores[~positive][None, :]
    concordance = float((np.count_nonzero(pos > neg) + .5*np.count_nonzero(pos == neg)) / (pos.size*neg.size))
    assert abs(auc_value-concordance) < 1e-12, (auc_value, concordance)
    assert np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0)
    assert np.array_equal([fpr[0], tpr[0], fpr[-1], tpr[-1]], [0, 0, 1, 1])
    return fpr, tpr, threshold, auc_value

def limits_on_grid(x, y, grid):
    """Exact left/right values of an empirical piecewise-linear ROC, preserving vertical jumps."""
    unique_x, first = np.unique(x, return_index=True)
    last = np.r_[first[1:]-1, len(x)-1]
    left, right = y[first], y[last]
    result_left, result_right = [], []
    for gx in grid:
        index = np.searchsorted(unique_x, gx)
        if index < len(unique_x) and unique_x[index] == gx:
            result_left.append(left[index])
            result_right.append(right[index])
        else:
            assert 0 < index < len(unique_x)
            fraction = (gx-unique_x[index-1]) / (unique_x[index]-unique_x[index-1])
            value = right[index-1] + fraction*(left[index]-right[index-1])
            result_left.append(value)
            result_right.append(value)
    return np.column_stack([result_left, result_right]).ravel()

def macro_curve(class_curves):
    grid = np.unique(np.concatenate([curve[0] for curve in class_curves]))
    x = np.repeat(grid, 2)
    y = np.mean([limits_on_grid(curve[0], curve[1], grid) for curve in class_curves], axis=0)
    expected_auc = float(np.mean([curve[3] for curve in class_curves]))
    assert abs(area(x, y)-expected_auc) < 1e-12
    return x, y, expected_auc
