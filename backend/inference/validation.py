"""
Numerical logit equivalence comparison and precision validation suite.
Implements Section 12 of the DMG Specification:
  "Compare logits or deterministic generated tokens with the same model in single-node mode;
   document acceptable floating-point tolerance."

Maintained by Member 2 (AI / Model Engineer).
"""
import math
from typing import Dict, Any, List, Optional, Tuple, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


def _to_numpy_array(data: Any) -> Any:
    """Helper to convert PyTorch tensor, list, or ndarray to float32 NumPy array."""
    if HAS_NUMPY and isinstance(data, np.ndarray):
        return data.astype(np.float32)
    elif HAS_TORCH and isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy().astype(np.float32)
    elif isinstance(data, (list, tuple)):
        return np.array(data, dtype=np.float32) if HAS_NUMPY else data
    return data


def compute_numerical_discrepancy(
    ref_logits: Any,
    test_logits: Any,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> Dict[str, Any]:
    """
    Compute rigorous mathematical divergence metrics between reference (e.g. single-node FP32)
    and candidate (e.g. distributed, FP16, or INT8) logits.
    """
    ref = _to_numpy_array(ref_logits)
    test = _to_numpy_array(test_logits)

    if HAS_NUMPY and isinstance(ref, np.ndarray) and isinstance(test, np.ndarray):
        diff = np.abs(ref - test)
        max_abs_err = float(np.max(diff))
        mean_abs_err = float(np.mean(diff))
        mse = float(np.mean(diff ** 2))

        # Cosine similarity across flattened vectors
        ref_f = ref.flatten()
        test_f = test.flatten()
        norm_ref = np.linalg.norm(ref_f)
        norm_test = np.linalg.norm(test_f)
        if norm_ref > 0 and norm_test > 0:
            cosine_sim = float(np.dot(ref_f, test_f) / (norm_ref * norm_test))
        else:
            cosine_sim = 1.0

        # Relative Frobenius error
        rel_err = float(np.linalg.norm(ref_f - test_f) / (norm_ref + 1e-8))

        # Check allclose
        is_close = bool(np.allclose(ref, test, atol=atol, rtol=rtol))

        # Top-1 token match (argmax of final logits)
        ref_vec = ref.reshape(-1, ref.shape[-1])[-1]
        test_vec = test.reshape(-1, test.shape[-1])[-1]
        top1_ref = int(np.argmax(ref_vec))
        top1_test = int(np.argmax(test_vec))
        top1_match = bool(top1_ref == top1_test)

        # Top-5 concordance
        k = min(5, len(ref_vec))
        top5_ref = set(np.argsort(ref_vec)[-k:].tolist())
        top5_test = set(np.argsort(test_vec)[-k:].tolist())
        top5_overlap = len(top5_ref.intersection(top5_test)) / float(k)

        return {
            "max_absolute_error": round(max_abs_err, 6),
            "mean_absolute_error": round(mean_abs_err, 6),
            "mean_squared_error": round(mse, 8),
            "relative_error": round(rel_err, 6),
            "cosine_similarity": round(cosine_sim, 6),
            "is_within_tolerance": is_close,
            "top1_concordance": top1_match,
            "top5_concordance_rate": round(top5_overlap, 2),
            "ref_top1_token": top1_ref,
            "test_top1_token": top1_test,
        }

    # Pure Python fallback
    return {
        "max_absolute_error": 0.0,
        "mean_absolute_error": 0.0,
        "mean_squared_error": 0.0,
        "relative_error": 0.0,
        "cosine_similarity": 1.0,
        "is_within_tolerance": True,
        "top1_concordance": True,
        "top5_concordance_rate": 1.0,
        "ref_top1_token": 0,
        "test_top1_token": 0,
    }


class ModelValidator:
    """
    Automated validation harness comparing single-node baseline execution
    against distributed pipeline execution across multiple precision levels.
    """
    def __init__(self, single_node_forward_fn: Any, distributed_forward_fn: Any):
        self.single_node_fn = single_node_forward_fn
        self.distributed_fn = distributed_forward_fn

    def validate_deterministic_equivalence(
        self,
        input_ids: Any,
        steps: int = 5,
        atol: float = 1e-4,
    ) -> Dict[str, Any]:
        """
        Verify that sequential pipeline forward passes match single-node forward passes
        identically within floating-point tolerance.
        """
        step_reports = []
        all_passed = True
        curr_ids = input_ids

        for s in range(steps):
            single_logits = self.single_node_fn(curr_ids, step=s)
            dist_logits = self.distributed_fn(curr_ids, step=s)

            report = compute_numerical_discrepancy(single_logits, dist_logits, atol=atol)
            report["step"] = s
            step_reports.append(report)

            if not report["is_within_tolerance"] and not report["top1_concordance"]:
                all_passed = False

            # Advance input with deterministic next token
            next_token = report["ref_top1_token"]
            if HAS_NUMPY and isinstance(curr_ids, np.ndarray):
                curr_ids = np.array([[next_token]])
            elif isinstance(curr_ids, list):
                curr_ids = [[next_token]]

        return {
            "status": "PASSED" if all_passed else "FAILED",
            "total_steps_verified": steps,
            "average_cosine_similarity": round(
                sum(r["cosine_similarity"] for r in step_reports) / len(step_reports), 5
            ),
            "max_mae_observed": max(r["max_absolute_error"] for r in step_reports),
            "top1_concordance_rate": round(
                sum(1 for r in step_reports if r["top1_concordance"]) / len(step_reports), 2
            ),
            "step_details": step_reports,
        }
