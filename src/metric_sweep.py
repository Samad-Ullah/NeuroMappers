"""Sprint 2 Task 3 — empirical comparison of similarity metrics & parameters.

We sweep a small grid of (metric, bins, sampling strategy, sampling %, learning
rate, iterations, interpolator) on a single sample subject and record the
results in a CSV. The chosen winner is then locked into the ANTs-based pipeline
in src/registration.py.

Why SimpleITK and not ANTs for the sweep:
- The acceptance criteria reference SimpleITK metric names by API method
  (SetMetricAsMattesMutualInformation, SetMetricAsJointHistogramMutualInformation,
  SetMetricAsCorrelation). ANTs only exposes Mattes MI cleanly via aff_metric.
- SimpleITK lets us vary learning rate, registration-time interpolator, and
  sampling strategy explicitly. ANTs hides those behind its own optimizer.
- Bins / sampling rate / iterations DO map back to ANTs kwargs and become the
  defaults we lock into src/registration.py.

The fixed image is the preprocessed pre-op T1 MRI; the moving image is the
preprocessed post-op CT (same direction as Task 2).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RESULTS_DIR
from src.utils import ensure_dir, get_logger

logger = get_logger("metric_sweep")

METRIC_SWEEP_CSV: Path = RESULTS_DIR / "sprint2_metric_comparison.csv"

# Sample subject for the sweep (same one used throughout Sprint 2 demos).
SAMPLE_ID_DEFAULT: str = "15454"


# ---------------------------------------------------------------------------
# config dataclass
# ---------------------------------------------------------------------------

@dataclass
class SweepConfig:
    """One row of the metric-comparison sweep."""
    label: str
    metric: str               # "mattes" | "joint_histogram" | "correlation"
    bins: int | None          # None for correlation (no histogram)
    sampling: str             # "RANDOM" | "REGULAR"
    sampling_pct: float       # fraction in (0, 1]
    learning_rate: float
    iterations: int
    interpolator: str         # "Linear" | "BSpline"


# Ten focused configs covering every parameter dimension required by the
# acceptance criteria. Row 2 is the baseline; subsequent rows vary one knob
# at a time so we can attribute differences to that knob.
SWEEP_GRID: tuple[SweepConfig, ...] = (
    SweepConfig("mattes_bins32",   "mattes",          32,  "RANDOM",  0.30, 1.0, 100, "Linear"),
    SweepConfig("mattes_bins50",   "mattes",          50,  "RANDOM",  0.30, 1.0, 100, "Linear"),
    SweepConfig("mattes_bins64",   "mattes",          64,  "RANDOM",  0.30, 1.0, 100, "Linear"),
    SweepConfig("mattes_regular",  "mattes",          50,  "REGULAR", 0.30, 1.0, 100, "Linear"),
    SweepConfig("mattes_lr0p5",    "mattes",          50,  "RANDOM",  0.30, 0.5, 100, "Linear"),
    SweepConfig("mattes_iter200",  "mattes",          50,  "RANDOM",  0.30, 1.0, 200, "Linear"),
    SweepConfig("mattes_bspline",  "mattes",          50,  "RANDOM",  0.30, 1.0, 100, "BSpline"),
    SweepConfig("joint_hist_50",   "joint_histogram", 50,  "RANDOM",  0.30, 1.0, 100, "Linear"),
    SweepConfig("correlation_neg", "correlation",     None, "RANDOM", 0.30, 1.0, 100, "Linear"),
)


# ---------------------------------------------------------------------------
# path helpers (mirror src/registration.py)
# ---------------------------------------------------------------------------

def _preprocessed_path(subject_id: str, modality: str, processed_root: Path) -> Path:
    ses = "ses-postop" if modality == "CT" else "ses-preop"
    return (
        processed_root
        / f"sub-{subject_id}"
        / ses
        / "anat"
        / f"sub-{subject_id}_space-RAS_{modality}.nii.gz"
    )


# ---------------------------------------------------------------------------
# SimpleITK runner for one config
# ---------------------------------------------------------------------------

def _build_registration(cfg: SweepConfig, fixed, moving, seed: int = 42):
    """Configure a SimpleITK ImageRegistrationMethod from a SweepConfig."""
    import SimpleITK as sitk

    R = sitk.ImageRegistrationMethod()

    # Metric
    if cfg.metric == "mattes":
        R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=cfg.bins)
    elif cfg.metric == "joint_histogram":
        R.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=cfg.bins)
    elif cfg.metric == "correlation":
        R.SetMetricAsCorrelation()
    else:
        raise ValueError(f"unknown metric: {cfg.metric!r}")

    # Sampling
    if cfg.sampling == "RANDOM":
        R.SetMetricSamplingStrategy(R.RANDOM)
    elif cfg.sampling == "REGULAR":
        R.SetMetricSamplingStrategy(R.REGULAR)
    else:
        raise ValueError(f"unknown sampling: {cfg.sampling!r}")
    R.SetMetricSamplingPercentage(cfg.sampling_pct, seed=seed)

    # Optimizer (gradient descent)
    R.SetOptimizerAsGradientDescent(
        learningRate=cfg.learning_rate,
        numberOfIterations=cfg.iterations,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    R.SetOptimizerScalesFromPhysicalShift()

    # Interpolator (registration-time)
    if cfg.interpolator == "Linear":
        R.SetInterpolator(sitk.sitkLinear)
    elif cfg.interpolator == "BSpline":
        R.SetInterpolator(sitk.sitkBSpline)
    else:
        raise ValueError(f"unknown interpolator: {cfg.interpolator!r}")

    # Multi-resolution pyramid (kept fixed across all configs so it doesn't
    # confound the comparison).
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Initial transform: align centers of mass (geometry-based, modality-agnostic).
    initial = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    R.SetInitialTransform(initial, inPlace=False)

    return R, initial


def _mattes_mi_score(fixed, warped, bins: int = 50, samples: float = 0.3, seed: int = 42) -> float:
    """Uniform yardstick: Mattes MI between fixed and warped, regardless of
    which metric was used during optimization. Lower = better, matching ANTs'
    reporting convention so values are directly comparable to Task 2's log."""
    import SimpleITK as sitk
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=bins)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(samples, seed=seed)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetInitialTransform(sitk.Euler3DTransform())  # identity
    return float(R.MetricEvaluate(fixed, warped))


def run_one_config(
    cfg: SweepConfig,
    fixed,
    moving,
    seed: int = 42,
) -> dict:
    """Run a single sweep configuration and return a dict row."""
    import SimpleITK as sitk

    t0 = perf_counter()
    final_metric_value = float("nan")
    flag = "ok"
    warped = None
    try:
        R, initial = _build_registration(cfg, fixed, moving, seed=seed)
        final_transform = R.Execute(fixed, moving)
        # Compose initial + learned transform for resampling.
        composite = sitk.CompositeTransform(final_transform)
        warped = sitk.Resample(
            moving, fixed, composite, sitk.sitkLinear, 0.0, moving.GetPixelID()
        )
        final_metric_value = float(R.GetMetricValue())
    except Exception as exc:
        logger.error("config %s FAILED: %s", cfg.label, exc)
        flag = f"exception:{type(exc).__name__}"

    runtime_s = perf_counter() - t0
    mattes_yard = _mattes_mi_score(fixed, warped) if warped is not None else float("nan")

    row = {
        "label": cfg.label,
        "metric": cfg.metric,
        "bins": cfg.bins if cfg.bins is not None else "",
        "sampling": cfg.sampling,
        "sampling_pct": cfg.sampling_pct,
        "learning_rate": cfg.learning_rate,
        "iterations": cfg.iterations,
        "interpolator": cfg.interpolator,
        "final_metric_value": round(final_metric_value, 6) if not np.isnan(final_metric_value) else float("nan"),
        "mattes_mi_final": round(mattes_yard, 6) if not np.isnan(mattes_yard) else float("nan"),
        "runtime_s": round(runtime_s, 2),
        "flag": flag,
        "visual_alignment_ok": "",   # filled in manually after Cell 47 figure
    }
    logger.info(
        "%s done in %.1fs | optim_metric=%.4f | mattes_yardstick=%.4f",
        cfg.label, runtime_s,
        final_metric_value if not np.isnan(final_metric_value) else float("nan"),
        mattes_yard if not np.isnan(mattes_yard) else float("nan"),
    )
    return row, warped, final_transform if flag == "ok" else None


# ---------------------------------------------------------------------------
# top-level sweep
# ---------------------------------------------------------------------------

def _load_pair(subject_id: str, processed_root: Path):
    """Read fixed (T1) and moving (CT) preprocessed volumes as SimpleITK images."""
    import SimpleITK as sitk

    fixed_path = _preprocessed_path(subject_id, "T1w", processed_root)
    moving_path = _preprocessed_path(subject_id, "CT",  processed_root)
    if not fixed_path.exists() or not moving_path.exists():
        raise FileNotFoundError(
            f"Preprocessed inputs missing for sub-{subject_id}: "
            f"T1w={fixed_path.exists()} CT={moving_path.exists()}"
        )
    fixed = sitk.ReadImage(str(fixed_path), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)
    return fixed, moving


def run_metric_sweep(
    subject_id: str = SAMPLE_ID_DEFAULT,
    grid: Iterable[SweepConfig] = SWEEP_GRID,
    processed_root: Path = PROCESSED_DIR,
    out_csv: Path = METRIC_SWEEP_CSV,
) -> pd.DataFrame:
    """Run every config in `grid` against `subject_id`'s preprocessed pair and
    save the comparison table."""
    fixed, moving = _load_pair(subject_id, processed_root)
    rows: list[dict] = []
    for cfg in grid:
        row, _warped, _xfm = run_one_config(cfg, fixed, moving)
        row["subject_id"] = subject_id
        rows.append(row)

    df = pd.DataFrame(rows)
    # Order columns to match the acceptance-criteria spec, then trailing extras.
    col_order = [
        "label", "metric", "bins", "sampling", "sampling_pct",
        "learning_rate", "iterations", "interpolator",
        "final_metric_value", "mattes_mi_final", "runtime_s",
        "visual_alignment_ok", "flag", "subject_id",
    ]
    df = df[col_order]
    ensure_dir(out_csv.parent)
    df.to_csv(out_csv, index=False)
    logger.info("Metric-comparison table -> %s (%d rows)", out_csv, len(df))
    return df


# ---------------------------------------------------------------------------
# Cell 47 helper: visual comparison of two configs
# ---------------------------------------------------------------------------

def _config_by_label(label: str) -> SweepConfig:
    for cfg in SWEEP_GRID:
        if cfg.label == label:
            return cfg
    raise KeyError(f"no SweepConfig with label={label!r}")


def compare_two_configs(
    subject_id: str = SAMPLE_ID_DEFAULT,
    config_a: str = "mattes_bins50",
    config_b: str = "correlation_neg",
    processed_root: Path = PROCESSED_DIR,
):
    """Render a 2x3 figure: rows = (axial, coronal, sagittal); columns =
    (config_a warped over T1) vs (config_b warped over T1).

    Used in the notebook to show that the negative-control metric (Correlation)
    visibly fails on multimodal MRI<->CT while the chosen MI metric succeeds."""
    import SimpleITK as sitk
    import matplotlib.pyplot as plt

    fixed, moving = _load_pair(subject_id, processed_root)

    cfg_a = _config_by_label(config_a)
    cfg_b = _config_by_label(config_b)

    _, warped_a, _ = run_one_config(cfg_a, fixed, moving)
    _, warped_b, _ = run_one_config(cfg_b, fixed, moving)

    t1 = sitk.GetArrayFromImage(fixed)            # (z, y, x) in sitk
    ct_a = sitk.GetArrayFromImage(warped_a) if warped_a is not None else np.zeros_like(t1)
    ct_b = sitk.GetArrayFromImage(warped_b) if warped_b is not None else np.zeros_like(t1)

    cz, cy, cx = (s // 2 for s in t1.shape)

    def _slices(vol):
        return [
            ("Axial",    np.rot90(vol[cz, :, :])),
            ("Coronal",  np.rot90(vol[:, cy, :])),
            ("Sagittal", np.rot90(vol[:, :, cx])),
        ]

    fig, axes = plt.subplots(3, 2, figsize=(9, 12))
    cols = [(config_a, ct_a), (config_b, ct_b)]
    t1_views = _slices(t1)

    for col_idx, (label, ct_vol) in enumerate(cols):
        ct_views = _slices(ct_vol)
        for row_idx, ((view_name, t1_view), (_, ct_view)) in enumerate(zip(t1_views, ct_views)):
            ax = axes[row_idx, col_idx]
            ax.imshow(t1_view, cmap="gray")
            # Mask near-air CT values so the bone overlay reads cleanly.
            ct_overlay = np.where(ct_view > 100, ct_view, np.nan)
            ax.imshow(ct_overlay, cmap="hot", alpha=0.45)
            if row_idx == 0:
                ax.set_title(label, fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(view_name, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f"sub-{subject_id} — warped CT overlaid on T1 (left: {config_a}, right: {config_b})",
        fontsize=12,
    )
    fig.tight_layout()
    plt.show()
    return fig
