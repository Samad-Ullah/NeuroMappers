"""Sprint 2 Task 2 — baseline rigid registration of post-op CT to pre-op MRI.

Direction follows the project proposal: pre-op T1w MRI is the **fixed** reference,
post-op CT is the **moving** image. Output is the CT resampled into MRI space,
plus the rigid (6-DOF) transform that produced it.

Built on ANTs (antspyx) to match the rest of the Sprint 2 pipeline.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from time import perf_counter
from typing import Iterable

import ants
import numpy as np
import pandas as pd

from src.config import PATIENT_IDS, PROCESSED_DIR, RESULTS_DIR
from src.utils import ensure_dir, get_logger

logger = get_logger("registration")

REGISTRATION_LOG_CSV: Path = RESULTS_DIR / "sprint2_registration_log.csv"

DEFAULT_TRANSFORM_TYPE: str = "Rigid"   # 6 DOF (3 rotations + 3 translations)
DEFAULT_METRIC: str = "mattes"          # Mattes mutual information
DEFAULT_FIXED: str = "T1w"
DEFAULT_MOVING: str = "CT"

# Sprint 2 Task 3 — defaults validated by results/sprint2_metric_comparison.csv
# on sub-15454. The SimpleITK knobs (learning rate, registration-time
# interpolator, RANDOM/REGULAR sampling) do not have direct ANTs equivalents
# and so are not transferred here; ANTs uses its own optimizer machinery with
# RANDOM sampling internally.
DEFAULT_BINS: int = 50                  # numberOfHistogramBins -> aff_sampling
DEFAULT_SAMPLING_RATE: float = 0.30     # SamplingPercentage  -> aff_random_sampling_rate
# Sweep showed iter=200 at the final pyramid level beat iter=100 by ~8% on the
# Mattes-MI yardstick and ran faster (early convergence). We give the final
# level 200 iters; coarser levels keep the standard ANTs Rigid budget.
DEFAULT_ITERATIONS: tuple[int, ...] = (2100, 1200, 200, 0)


# ---------------------------------------------------------------------------
# data class
# ---------------------------------------------------------------------------

@dataclass
class RegistrationResult:
    subject_id: str
    fixed_modality: str
    moving_modality: str
    transform_type: str
    metric: str
    fixed_path: str
    moving_path: str
    warped_path: str
    transform_path: str
    final_metric: float
    runtime_s: float
    flag: str


# ---------------------------------------------------------------------------
# path helpers
# ---------------------------------------------------------------------------

def preprocessed_path(
    subject_id: str,
    modality: str,
    processed_root: Path = PROCESSED_DIR,
) -> Path:
    """Canonical path to a preprocessed NIfTI under data/processed/."""
    ses = "ses-postop" if modality == "CT" else "ses-preop"
    return (
        processed_root
        / f"sub-{subject_id}"
        / ses
        / "anat"
        / f"sub-{subject_id}_space-RAS_{modality}.nii.gz"
    )


def registration_dir(subject_id: str, processed_root: Path = PROCESSED_DIR) -> Path:
    return processed_root / f"sub-{subject_id}" / "registration"


# ---------------------------------------------------------------------------
# similarity for the log
# ---------------------------------------------------------------------------

def _final_similarity(fixed: "ants.ANTsImage", warped: "ants.ANTsImage") -> float:
    """Mattes MI between fixed and warped (lower = better in ANTs)."""
    try:
        return float(
            ants.image_similarity(fixed, warped, metric_type="MattesMutualInformation")
        )
    except Exception as exc:
        logger.warning("similarity calc failed: %s", exc)
        return float("nan")


# ---------------------------------------------------------------------------
# single-subject registration
# ---------------------------------------------------------------------------

def register_subject(
    subject_id: str,
    fixed_modality: str = DEFAULT_FIXED,
    moving_modality: str = DEFAULT_MOVING,
    transform_type: str = DEFAULT_TRANSFORM_TYPE,
    metric: str = DEFAULT_METRIC,
    bins: int = DEFAULT_BINS,
    sampling_rate: float = DEFAULT_SAMPLING_RATE,
    iterations: tuple[int, ...] = DEFAULT_ITERATIONS,
    processed_root: Path = PROCESSED_DIR,
) -> RegistrationResult | None:
    """Rigid-register `moving` onto `fixed` and save the warped image + transform."""
    fixed_path = preprocessed_path(subject_id, fixed_modality, processed_root)
    moving_path = preprocessed_path(subject_id, moving_modality, processed_root)
    if not fixed_path.exists() or not moving_path.exists():
        logger.warning(
            "sub-%s: missing preprocessed input(s) (fixed=%s, moving=%s); skipping",
            subject_id, fixed_path.exists(), moving_path.exists(),
        )
        return None

    out_dir = registration_dir(subject_id, processed_root)
    ensure_dir(out_dir)
    warped_path = out_dir / f"sub-{subject_id}_space-{fixed_modality}_{moving_modality}.nii.gz"
    transform_path = out_dir / f"sub-{subject_id}_to-{fixed_modality}_xfm.mat"

    fixed = ants.image_read(str(fixed_path))
    moving = ants.image_read(str(moving_path))

    t0 = perf_counter()
    flag = "ok"
    final_metric = float("nan")
    try:
        reg = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform=transform_type,
            aff_metric=metric,
            aff_sampling=bins,
            aff_random_sampling_rate=sampling_rate,
            aff_iterations=iterations,
        )
        warped = reg["warpedmovout"]
        ants.image_write(warped, str(warped_path))

        if reg.get("fwdtransforms"):
            src_xfm = Path(reg["fwdtransforms"][0])
            if src_xfm.exists():
                shutil.copy2(src_xfm, transform_path)

        final_metric = _final_similarity(fixed, warped)
    except Exception as exc:
        logger.error("sub-%s registration FAILED: %s", subject_id, exc)
        flag = f"exception:{type(exc).__name__}"
        warped_path = Path("")
        transform_path = Path("")

    runtime = perf_counter() - t0
    result = RegistrationResult(
        subject_id=subject_id,
        fixed_modality=fixed_modality,
        moving_modality=moving_modality,
        transform_type=transform_type,
        metric=metric,
        fixed_path=str(fixed_path),
        moving_path=str(moving_path),
        warped_path=str(warped_path),
        transform_path=str(transform_path),
        final_metric=round(final_metric, 6) if not np.isnan(final_metric) else float("nan"),
        runtime_s=round(runtime, 3),
        flag=flag,
    )
    logger.info(
        "sub-%s %s->%s done in %.2fs (Mattes MI=%s)",
        subject_id, moving_modality, fixed_modality, runtime,
        f"{final_metric:.4f}" if not np.isnan(final_metric) else "n/a",
    )
    return result


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------

def register_all(
    subject_ids: Iterable[str] = PATIENT_IDS,
    fixed_modality: str = DEFAULT_FIXED,
    moving_modality: str = DEFAULT_MOVING,
    transform_type: str = DEFAULT_TRANSFORM_TYPE,
    metric: str = DEFAULT_METRIC,
    bins: int = DEFAULT_BINS,
    sampling_rate: float = DEFAULT_SAMPLING_RATE,
    iterations: tuple[int, ...] = DEFAULT_ITERATIONS,
    processed_root: Path = PROCESSED_DIR,
    log_csv: Path = REGISTRATION_LOG_CSV,
) -> pd.DataFrame:
    """Run register_subject over every ID; write log; return as DataFrame."""
    rows: list[RegistrationResult] = []
    for sid in subject_ids:
        result = register_subject(
            sid, fixed_modality, moving_modality, transform_type, metric,
            bins, sampling_rate, iterations, processed_root,
        )
        if result is not None:
            rows.append(result)
    df = pd.DataFrame([asdict(r) for r in rows])
    ensure_dir(log_csv.parent)
    df.to_csv(log_csv, index=False)
    logger.info("Registration log -> %s (%d rows)", log_csv, len(df))
    return df
