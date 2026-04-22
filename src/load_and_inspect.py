"""Load MRI and CT volumes and inspect dimensions, spacing, and orientation."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd


@dataclass
class VolumeInfo:
    path: str
    modality_guess: str
    format: str
    shape: tuple[int, ...]
    voxel_spacing_mm: tuple[float, ...]
    orientation: tuple[str, str, str]
    dtype: str
    min_intensity: float
    max_intensity: float
    nan_count: int


def load_volume(path: str | Path) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    """Load a NIfTI volume as (data, affine, nibabel image)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Volume not found: {path}")
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    return data, img.affine, img


def _guess_modality(path: Path) -> str:
    name = path.name.lower()
    if "_ct" in name or name.endswith("_ct.nii.gz") or name.endswith("_ct.nii"):
        return "CT"
    if "t1w" in name or "_t1" in name:
        return "T1w"
    if "t2w" in name or "_t2" in name:
        return "T2w"
    if "flair" in name:
        return "FLAIR"
    if "swi" in name:
        return "SWI"
    return "unknown"


def _nifti_format(img: nib.spatialimages.SpatialImage) -> str:
    if isinstance(img, nib.Nifti2Image):
        return "NIfTI-2"
    if isinstance(img, nib.Nifti1Image):
        return "NIfTI-1"
    return type(img).__name__


def inspect_volume(path: str | Path) -> VolumeInfo:
    """Read header metadata and intensity range for a single NIfTI file."""
    path = Path(path)
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    nan_count = int(np.isnan(data).sum()) if np.issubdtype(data.dtype, np.floating) else 0
    return VolumeInfo(
        path=str(path),
        modality_guess=_guess_modality(path),
        format=_nifti_format(img),
        shape=tuple(int(x) for x in img.shape),
        voxel_spacing_mm=tuple(float(x) for x in img.header.get_zooms()[:3]),
        orientation=nib.aff2axcodes(img.affine),
        dtype=str(data.dtype),
        min_intensity=float(np.nanmin(data)),
        max_intensity=float(np.nanmax(data)),
        nan_count=nan_count,
    )


def inspect_many(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Inspect several volumes and return a summary DataFrame."""
    rows = [asdict(inspect_volume(p)) for p in paths]
    return pd.DataFrame(rows)


def find_nifti_files(root: str | Path) -> list[Path]:
    """Recursively list all .nii / .nii.gz files under a directory."""
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix == ".nii" or p.name.endswith(".nii.gz"))
