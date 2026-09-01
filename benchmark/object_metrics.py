"""Additional object-level metrics and full-resolution LAS export.

This module intentionally does not implement AP. Mask3D's original AP
evaluator remains in :mod:`benchmark.evaluate_semantic_instance`.
"""

from pathlib import Path

import numpy as np


def _safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def counts_to_metrics(tp, fp, fn):
    tp, fp, fn = int(tp), int(fp), int(fn)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def match_instances(
    prediction,
    gt_masks,
    gt_classes,
    score_threshold=0.5,
    iou_threshold=0.5,
    valid_class_ids=(1,),
):
    """Greedily match one scene's final predictions to ground truth.

    Candidate pairs are ordered by descending mask IoU, descending prediction
    confidence, then fixed prediction and GT indices. This gives deterministic
    one-to-one matching including exact ties.
    """

    pred_masks = np.asarray(prediction["pred_masks"]).astype(bool)
    pred_scores = np.asarray(prediction["pred_scores"], dtype=np.float64)
    pred_classes = np.asarray(prediction["pred_classes"], dtype=np.int64)
    gt_masks = np.asarray(gt_masks).astype(bool)
    gt_classes = np.asarray(gt_classes, dtype=np.int64)

    if pred_masks.ndim != 2:
        raise ValueError("pred_masks must have shape [num_points, num_predictions]")
    if gt_masks.ndim != 2:
        raise ValueError("gt_masks must have shape [num_gt, num_points]")
    if pred_masks.shape[0] != gt_masks.shape[1]:
        raise ValueError("prediction and GT masks must contain the same points")
    if pred_masks.shape[1] != len(pred_scores) or len(pred_scores) != len(pred_classes):
        raise ValueError("prediction mask, score and class counts do not match")
    if gt_masks.shape[0] != len(gt_classes):
        raise ValueError("GT mask and class counts do not match")

    valid_class_ids = tuple(int(class_id) for class_id in valid_class_ids)
    keep_pred = np.flatnonzero(
        (pred_scores >= float(score_threshold))
        & np.isin(pred_classes, valid_class_ids)
    )
    keep_gt = np.flatnonzero(np.isin(gt_classes, valid_class_ids))

    candidates = []
    for pred_idx in keep_pred:
        same_class_gt = keep_gt[gt_classes[keep_gt] == pred_classes[pred_idx]]
        pred_mask = pred_masks[:, pred_idx]
        for gt_idx in same_class_gt:
            intersection = np.count_nonzero(pred_mask & gt_masks[gt_idx])
            union = np.count_nonzero(pred_mask | gt_masks[gt_idx])
            iou = _safe_divide(intersection, union)
            if iou >= float(iou_threshold):
                candidates.append(
                    (
                        -iou,
                        -float(pred_scores[pred_idx]),
                        int(pred_idx),
                        int(gt_idx),
                    )
                )

    candidates.sort()
    matched_pred = set()
    matched_gt = set()
    matches = []
    for negative_iou, _, pred_idx, gt_idx in candidates:
        if pred_idx in matched_pred or gt_idx in matched_gt:
            continue
        matched_pred.add(pred_idx)
        matched_gt.add(gt_idx)
        matches.append((pred_idx, gt_idx, -negative_iou))

    metrics = counts_to_metrics(
        len(matches),
        len(keep_pred) - len(matches),
        len(keep_gt) - len(matches),
    )
    metrics["matches"] = matches
    metrics["per_class"] = {}
    for class_id in valid_class_ids:
        class_tp = sum(
            pred_classes[pred_idx] == class_id for pred_idx, _, _ in matches
        )
        class_pred = np.count_nonzero(pred_classes[keep_pred] == class_id)
        class_gt = np.count_nonzero(gt_classes[keep_gt] == class_id)
        metrics["per_class"][class_id] = counts_to_metrics(
            class_tp, class_pred - class_tp, class_gt - class_tp
        )
    return metrics


def aggregate_scene_metrics(scene_metrics, valid_class_ids=(1,)):
    total = counts_to_metrics(
        sum(result["tp"] for result in scene_metrics.values()),
        sum(result["fp"] for result in scene_metrics.values()),
        sum(result["fn"] for result in scene_metrics.values()),
    )
    total["per_class"] = {}
    for class_id in valid_class_ids:
        total["per_class"][class_id] = counts_to_metrics(
            sum(result["per_class"][class_id]["tp"] for result in scene_metrics.values()),
            sum(result["per_class"][class_id]["fp"] for result in scene_metrics.values()),
            sum(result["per_class"][class_id]["fn"] for result in scene_metrics.values()),
        )
    return total


def make_point_predictions(prediction, num_points, valid_class_ids=(1,)):
    """Resolve final overlapping masks by confidence for LAS point fields."""

    pred_masks = np.asarray(prediction["pred_masks"]).astype(bool)
    pred_scores = np.asarray(prediction["pred_scores"], dtype=np.float64)
    pred_classes = np.asarray(prediction["pred_classes"], dtype=np.int64)
    if pred_masks.shape != (num_points, len(pred_scores)):
        raise ValueError("full-resolution prediction mask shape is inconsistent")

    semantic_pred = np.zeros(num_points, dtype=np.int16)
    instance_pred = np.full(num_points, -100, dtype=np.int32)
    instance_score = np.zeros(num_points, dtype=np.float32)
    unassigned = np.ones(num_points, dtype=bool)

    finite_scores = np.where(np.isfinite(pred_scores), pred_scores, -np.inf)
    # lexsort uses its final key as primary: descending score, then index.
    order = np.lexsort((np.arange(len(pred_scores)), -finite_scores))
    next_instance_id = 1
    for pred_idx in order:
        class_id = int(pred_classes[pred_idx])
        if class_id not in valid_class_ids:
            continue
        assigned = pred_masks[:, pred_idx] & unassigned
        if not assigned.any():
            continue
        semantic_pred[assigned] = class_id
        instance_pred[assigned] = next_instance_id
        instance_score[assigned] = float(pred_scores[pred_idx])
        unassigned[assigned] = False
        next_instance_id += 1
    return semantic_pred, instance_pred, instance_score


def write_prediction_las(raw_filepath, prediction, output_filepath):
    """Write every original TXT point and five requested prediction fields."""

    try:
        import laspy
    except ImportError as error:
        raise ImportError(
            "LAS export requires laspy. Install it in the existing Mask3D "
            "environment with: pip install laspy"
        ) from error

    raw = np.loadtxt(raw_filepath, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[None, :]
    if raw.shape[1] != 8:
        raise ValueError(f"{raw_filepath}: expected 8 raw TXT columns")

    semantic_gt = np.rint(raw[:, 6]).astype(np.int16)
    instance_gt = np.rint(raw[:, 7]).astype(np.int32)
    semantic_pred, instance_pred, instance_score = make_point_predictions(
        prediction, len(raw)
    )

    header = laspy.LasHeader(point_format=3, version="1.2")
    # Raw TXT coordinates are written with six decimal places. Per-plot
    # offsets keep the LAS int32 representation in range at this exact scale.
    header.scales = np.array([0.000001, 0.000001, 0.000001])
    header.offsets = np.floor(raw[:, :3].min(axis=0))
    for name, data_type in (
        ("semantic_gt", "i2"),
        ("semantic_pred", "i2"),
        ("instance_gt", "i4"),
        ("instance_pred", "i4"),
        ("instance_score", "f4"),
    ):
        header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=data_type))

    las = laspy.LasData(header)
    las.x, las.y, las.z = raw[:, 0], raw[:, 1], raw[:, 2]
    rgb = np.clip(np.rint(raw[:, 3:6]), 0, 255).astype(np.uint16) * 257
    las.red, las.green, las.blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    las.semantic_gt = semantic_gt
    las.semantic_pred = semantic_pred
    las.instance_gt = instance_gt
    las.instance_pred = instance_pred
    las.instance_score = instance_score

    output_filepath = Path(output_filepath)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    las.write(output_filepath)
    return output_filepath
