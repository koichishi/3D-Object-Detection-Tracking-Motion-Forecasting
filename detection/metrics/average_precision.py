from collections import defaultdict
from dataclasses import dataclass
from pickle import NONE
from typing import List

import torch

from detection.metrics.types import EvaluationFrame


@dataclass
class PRCurve:
    """A precision/recall curve.

    Attributes:
        precision: [N] vector of precision values, where N is the total number of detections.
            The element at index n denotes the precision of the top n detections when ordered by
            decreasing detection scores.
        recall: [N] vector of recall values, where N is the total number of detections.
            The element at index n denotes the recall of the top n detections when ordered by
            decreasing detection scores.
    """

    precision: torch.Tensor
    recall: torch.Tensor


@dataclass
class AveragePrecisionMetric:
    """Stores average precision and its associate precision-recall curve."""

    ap: float
    pr_curve: PRCurve


def compute_precision_recall_curve(
    frames: List[EvaluationFrame], threshold: float
) -> PRCurve:
    """Compute a precision/recall curve over a batch of evaluation frames.

    The PR curve plots the trade-off between precision and recall when sweeping
    across different score thresholds for your detections. To compute precision
    and recall for a score threshold s_i, consider the set of detections with
    scores greater than or equal to s_i. A detection is a true positive if it
    matches a ground truth label; it is a false positive if it does not.

    With this, we define precision = TP / (TP + FP) and recall = TP / (TP + FN),
    where TP is the number of true positive detections, FP is the number of false
    positive detections, and FN is the number of false negative labels (i.e. the
    number of ground truth labels that did not match any detections). By varying
    the score threshold s_i over all detection scores, we have the PR curve.

    What does it mean for a detection to match a ground truth label? In this assignment, we use
    the following definition: A detection matches a ground truth label if: (1) the Euclidean
    distance between their centers is at most `threshold`; and (2) no higher scoring detection
    satisfies condition (1) with respect to the same label.

    Args:
        frames: A batch of evaluation frames, each containing a detection/label pair.
        threshold: Two bounding boxes match if their bird's eye view
            center-to-center distance is strictly less than `threshold`.

    Returns:
        A precision/recall curve.
    """
    # TODO: Replace this stub code.
    # return PRCurve(torch.zeros(0), torch.zeros(0))


    num_labels = 0 # for sanity check

    total_tp = total_fn = total_fp = total_scores = None

    for frame in frames:

        matched = set()
        D, L = len(frame.detections), len(frame.labels)
        tp = fp = torch.zeros(D)
        fn = torch.ones(L)

        num_labels += L

        # Sort the detections in decreasing order of scores
        d_score_order = torch.argsort(frame.detections.scores, descending=True)

        # L by D dist matrix (euc_dist[:, d] = dist of labels from detection d)
        euc_dist = torch.cdist(frame.labels.centroids, frame.detections.centroids[d_score_order])
        # print(euc_dist)

        # for each detect, find closest label (compute tp and fp)
        for detect_idx in d_score_order:
            label_idx = torch.argmax(euc_dist[:, detect_idx]) # only check the closest label per detect
            if euc_dist[label_idx, detect_idx] > threshold or label_idx in matched:
                fp[detect_idx] = 1
            else:
                tp[detect_idx] = 1
                matched.add(label_idx)

        # update fn 
        for i in matched:
            fn[i] = 0

        if not total_tp:
            total_tp = torch.cat((total_tp, tp), dim=0)
            total_fn = torch.cat((total_fn, fn), dim=0)
            total_fp = torch.cat((total_fp, fp), dim=0)
            total_scores =  torch.cat((total_scores, frame.detections.scores), dim=0)
        else:
            total_tp, total_fn, total_fp, total_scores = tp, fn, fp, frame.detections.scores

    tp_p_fp = torch.sum(total_tp) + torch.sum(total_fp)  # See discussion board (not sure if fn is right)
    tp_p_fn = torch.sum(total_tp) + torch.sum(total_fn)

    d_score_order = torch.argsort(total_scores, descending=True)

    cum_tp = torch.cumsum(total_tp[d_score_order])
    cum_tp_fp = torch.cumsum((total_tp + total_fp)[d_score_order])

    # check if number of tp_p_fn == number of labels within all the frames
    assert tp_p_fn == num_labels

    r_val = cum_tp / tp_p_fn
    p_val = cum_tp / cum_tp_fp # divide by the number of detection SO FAR

    return PRCurve(p_val, r_val)

    '''
    precisions = torch.zeros(len(frames))
    recalls = torch.zeros(len(frames))
    for f in range(len(frames)):
        D, L = len(frames[i].detections), len(frames[i].labels)
        euc_dist = torch.zeros(D, L)
        for i in range(D):
            euc_dist[i] = torch.cdist(frames[i].labels.centroids_x() - torch.tensor([frames[i].detections.centroids_x()[i]]).expand(L), 
                                        frames[i].labels.centroids_y() - torch.tensor([frames[i].detections.centroids_y()[i]]).expand(L))
        nearest_dist_mask = euc_dist  == torch.max(euc_dist, dim = 0).expand(-1, L)
        euc_dist[not nearest_dist_mask] = float('inf')
        dist_mask = euc_dist <= threshold
        score_mask = frames[i].detections.scores.expand(-1, L) == torch.max(frames[i].detections.scores.expand(-1, L) * dist_mask, dim = 1).expand(D, -1)

        precisions[i] = float(torch.count_nonzero(score_mask) / D)
        recalls[i] = float(torch.count_nonzero(score_mask) / L)
    
    return PRCurve(precisions, recalls)
    '''


def compute_area_under_curve(curve: PRCurve) -> float:
    """Return the area under the given curve.

    Given a `PRCurve` curve, this function computes the area under the curve as:
        AP = \sum_{i = 1}^{n} (r_i - r_{i - 1}) * p_i
    where r_i (resp. p_i) is the recall (resp. precision) of the top i detections,
    n is the total number of detections, and we set r_0 = 0.0. Intuitively, this
    is computing the integral of the step function defined by the PRCurve.

    Args:
        curve: The precision/recall curve.

    Returns:
        The area under the curve, as defined above.
    """
    # TODO: Replace this stub code.
    # return torch.sum(curve.recall).item() * 0.0
    rec_m1 = torch.roll(curve.recall, 1)
    rec_m1[0] = 0
    return float(torch.sum(curve.precision * (curve.recall - rec_m1)))

    

def compute_average_precision(
    frames: List[EvaluationFrame], threshold: float
) -> AveragePrecisionMetric:
    """Compute average precision over a batch of evaluation frames.

    Args:
        frames: A batch of evaluation frames, each containing a detection/label pair.
        threshold: Two bounding boxes match if their bird's eye view
            center-to-center distance is strictly less than `threshold`.

    Returns:
        A dataclass consisting of a PRCurve and its average precision.
    """
    # TODO: Replace this stub code.
    # return AveragePrecisionMetric(0.0, PRCurve(torch.zeros(0), torch.zeros(0)))
    prc = compute_precision_recall_curve(frames, threshold)
    return AveragePrecisionMetric(float(torch.mean(prc.precision)), prc)
