"""Sprint 2 Task 1 — preprocessing for Lead-Tutor MRI / CT volumes.

Standardizes raw native-space NIfTI volumes into registration-ready form:
reorient to RAS+ -> resample to 1 mm isotropic -> intensity normalization
(HU clip for CT, z-score within foreground for MRI) -> crop to content ->
per-volume quality check.

Built on ANTs (antspyx) to stay consistent with Catarina's prototype.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import ants
import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu

from src.config import LEADTUTOR_DIR, PATIENT_IDS, PROCESSED_DIR, RESULTS_DIR
from src.utils import ensure_dir, get_logger

logger = get_logger("preprocessing")

PREPROCESSING_QC_CSV: Path = RESULTS_DIR / "sprint2_preprocessing_qc.csv"

CT_HU_MIN: float = -1024.0
CT_HU_MAX: float = 3500.0
ISO_SPACING: tuple[float, float, float] = (1.0, 1.0, 1.0)
INTERP_LINEAR: int = 0
INTERP_NN: int = 1


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def find_subject_files(subject_id: str, root: Path = LEADTUTOR_DIR) -> dict[str, Path]:
    """Return {modality: path} for one subject under rawdata/sub-XXXXX/."""
    subj_dir = root / "rawdata" / f"sub-{subject_id}"
    if not subj_dir.exists():
        return {}
    out: dict[str, Path] = {}
    for path in sorted(subj_dir.rglob("*.nii.gz")):
        name = path.name.lower()
        if "_ct" in name and "CT" not in out:
            out["CT"] = path
        elif "t1w" in name and "T1w" not in out:
            out["T1w"] = path
        elif "t2w" in name and "T2w" not in out:
            out["T2w"] = path
        elif "flair" in name and "FLAIR" not in out:
            out["FLAIR"] = path
        elif "swi" in name and "SWI" not in out:
            out["SWI"] = path
    return out


# ---------------------------------------------------------------------------
# per-step transforms
# ---------------------------------------------------------------------------

def reorient_to_ras(img: "ants.ANTsImage") -> "ants.ANTsImage":
    """Reorient to RAS+. Idempotent for already-RAS volumes."""
    return ants.reorient_image2(img, "RAS")


def resample_isotropic(
    img: "ants.ANTsImage",
    spacing: tuple[float, float, float] = ISO_SPACING,
    interp: int = INTERP_LINEAR,
) -> "ants.ANTsImage":
    """Resample to a target physical voxel size (mm). Linear interp by default."""
    return ants.resample_image(img, spacing, use_voxels=False, interp_type=interp)


def replace_nans(img: "ants.ANTsImage", fill: float) -> "ants.ANTsImage":
    arr = img.numpy()
    if not np.isnan(arr).any():
        return img
    arr = np.nan_to_num(arr, nan=fill)
    return img.new_image_like(arr.astype(np.float32))


def clip_ct(
    img: "ants.ANTsImage",
    lo: float = CT_HU_MIN,
    hi: float = CT_HU_MAX,
) -> "ants.ANTsImage":
    """NaN -> air HU, then clip to a sensible HU window."""
    img = replace_nans(img, fill=lo)
    arr = np.clip(img.numpy(), lo, hi).astype(np.float32)
    return img.new_image_like(arr)


def normalize_mri_zscore(img: "ants.ANTsImage") -> "ants.ANTsImage":
    """Z-score the volume using the foreground (Otsu) mean/std."""
    arr = np.nan_to_num(img.numpy().astype(np.float32), nan=0.0)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return img
    try:
        thr = float(threshold_otsu(finite))
    except ValueError:
        thr = float(finite.mean())
    fg = arr[arr > thr]
    if fg.size < 100:
        return img
    mean, std = float(fg.mean()), float(fg.std())
    if std < 1e-6:
        return img
    return img.new_image_like((arr - mean) / std)


def crop_to_content(
    img: "ants.ANTsImage",
    threshold: float | None = None,
    margin: int = 3,
) -> "ants.ANTsImage":
    """Crop to bounding box of voxels above threshold (default: above min + eps)."""
    arr = img.numpy()
    if threshold is None:
        threshold = float(arr.min()) + 1e-3
    mask = arr > threshold
    if not mask.any():
        return img
    coords = np.array(np.where(mask))
    lo = np.maximum(coords.min(axis=1) - margin, 0)
    hi = np.minimum(coords.max(axis=1) + margin + 1, np.array(arr.shape))
    return ants.crop_indices(
        img,
        tuple(int(x) for x in lo),
        tuple(int(x) for x in hi),
    )


# ---------------------------------------------------------------------------
# QC
# ---------------------------------------------------------------------------

@dataclass
class QCRow:
    subject_id: str
    modality: str
    stage: str
    shape: str
    spacing_mm: str
    orientation: str
    finite_fraction: float
    nan_count: int
    min_intensity: float
    max_intensity: float
    foreground_fraction: float
    flag: str


def _orientation_str(img: "ants.ANTsImage") -> str:
    try:
        return img.orientation
    except Exception:
        return ""


def quality_check(
    img: "ants.ANTsImage",
    subject_id: str,
    modality: str,
    stage: str,
) -> QCRow:
    arr = img.numpy()
    total = int(arr.size)
    finite_mask = np.isfinite(arr)
    finite_count = int(finite_mask.sum())
    nan_count = int(np.isnan(arr).sum())
    finite_arr = arr[finite_mask] if finite_count else np.array([0.0])
    fg_threshold = float(finite_arr.min()) + 1e-3
    foreground = float((finite_arr > fg_threshold).sum() / max(total, 1))

    flags: list[str] = []
    if nan_count > 0:
        flags.append(f"nan={nan_count}")
    if finite_count == 0:
        flags.append("all-nonfinite")
    if foreground < 0.05:
        flags.append("low-foreground")

    return QCRow(
        subject_id=subject_id,
        modality=modality,
        stage=stage,
        shape=str(tuple(arr.shape)),
        spacing_mm=str(tuple(round(float(s), 3) for s in img.spacing)),
        orientation=_orientation_str(img),
        finite_fraction=round(finite_count / max(total, 1), 6),
        nan_count=nan_count,
        min_intensity=round(float(finite_arr.min()), 3) if finite_count else float("nan"),
        max_intensity=round(float(finite_arr.max()), 3) if finite_count else float("nan"),
        foreground_fraction=round(foreground, 6),
        flag=";".join(flags) if flags else "ok",
    )


# ---------------------------------------------------------------------------
# per-modality pipeline
# ---------------------------------------------------------------------------

def preprocess_one(
    path: Path,
    subject_id: str,
    modality: str,
    out_dir: Path,
) -> tuple[Path, list[QCRow]]:
    """Run the full preprocessing chain on a single NIfTI file."""
    rows: list[QCRow] = []
    img = ants.image_read(str(path))
    rows.append(quality_check(img, subject_id, modality, "raw"))

    img = reorient_to_ras(img)
    img = resample_isotropic(img, ISO_SPACING, INTERP_LINEAR)
    if modality == "CT":
        img = clip_ct(img)
    else:
        img = normalize_mri_zscore(img)
    img = crop_to_content(img)

    rows.append(quality_check(img, subject_id, modality, "preprocessed"))

    ensure_dir(out_dir)
    out_path = out_dir / f"sub-{subject_id}_space-RAS_{modality}.nii.gz"
    ants.image_write(img, str(out_path))
    return out_path, rows


# ---------------------------------------------------------------------------
# subject-level orchestrator
# ---------------------------------------------------------------------------

def preprocess_subject(
    subject_id: str,
    out_root: Path = PROCESSED_DIR,
) -> tuple[dict[str, Path], list[QCRow]]:
    """Preprocess every modality of one subject; return {modality: out_path}, qc rows."""
    files = find_subject_files(subject_id)
    if not files:
        logger.warning("No NIfTI files found for sub-%s", subject_id)
        return {}, []

    saved: dict[str, Path] = {}
    qc_rows: list[QCRow] = []
    subj_root = out_root / f"sub-{subject_id}"

    for modality, path in files.items():
        ses = "ses-postop" if modality == "CT" else "ses-preop"
        out_dir = subj_root / ses / "anat"
        try:
            out_path, rows = preprocess_one(path, subject_id, modality, out_dir)
            saved[modality] = out_path
            qc_rows.extend(rows)
            logger.info("sub-%s %s -> %s", subject_id, modality, out_path.name)
        except Exception as exc:
            logger.error("sub-%s %s FAILED: %s", subject_id, modality, exc)
            qc_rows.append(QCRow(
                subject_id=subject_id, modality=modality, stage="error",
                shape="", spacing_mm="", orientation="",
                finite_fraction=0.0, nan_count=0,
                min_intensity=float("nan"), max_intensity=float("nan"),
                foreground_fraction=0.0,
                flag=f"exception:{type(exc).__name__}",
            ))
    return saved, qc_rows


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------

def preprocess_all(
    subject_ids: Iterable[str] = PATIENT_IDS,
    out_root: Path = PROCESSED_DIR,
    qc_csv: Path = PREPROCESSING_QC_CSV,
) -> pd.DataFrame:
    """Run preprocess_subject over every ID; write QC table; return as DataFrame."""
    all_rows: list[QCRow] = []
    for sid in subject_ids:
        _, rows = preprocess_subject(sid, out_root=out_root)
        all_rows.extend(rows)
    df = pd.DataFrame([asdict(r) for r in all_rows])
    ensure_dir(qc_csv.parent)
    df.to_csv(qc_csv, index=False)
    logger.info("Preprocessing QC -> %s (%d rows)", qc_csv, len(df))
    return df
