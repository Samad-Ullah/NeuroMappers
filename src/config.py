"""Pipeline configuration: paths, dataset constants, modality catalogue."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_ROOT: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_ROOT / "raw"
PROCESSED_DIR: Path = DATA_ROOT / "processed"
RESULTS_DIR: Path = PROJECT_ROOT / "results"

LEADTUTOR_DIR: Path = RAW_DIR / "LeadTutor"

MRI_MODALITIES: tuple[str, ...] = ("T1w", "T2w", "FLAIR")
CT_MODALITY: str = "CT"
ALL_MODALITIES: tuple[str, ...] = MRI_MODALITIES + (CT_MODALITY,)

PATIENT_IDS: tuple[str, ...] = (
    "15454", "29781", "33544", "39468", "57245",
    "76325", "78754", "80206", "84257", "93127",
)

NIFTI_EXTENSIONS: tuple[str, ...] = (".nii", ".nii.gz")
