---
title: "NeuroMappers — Sprint 2 Closure"
author: "Samad Ullah, Catarina Souto"
date: "2026-04-30"
---

# NeuroMappers — Sprint 2 Closure

**Sprint window:** 2026-04-22 to 2026-04-30
**Branch:** `feature/NeuroMapper_development`
**Authors:** Samad Ullah (USC) — preprocessing, registration, metric sweep, sweep-driven defaults
**Co-reviewer:** Catarina Souto (University of Porto) — visualization feedback round, rationale co-sign (pending)
**Supervisor:** Prof. João Paulo Cunha
**Dataset:** Lead-Tutor — Madan et al., *Aperture Neuro* (2025), DOI 10.52294/001c.129658

---

## 1. Sprint goal

Turn the **raw, heterogeneous Lead-Tutor volumes** (10 patients, 31 NIfTI files, 4 distinct orientations, anisotropic spacing) into a **registration-ready dataset** and produce a **baseline rigid registration of post-op CT to pre-op T1 MRI** for every patient — with empirical evidence justifying the chosen similarity metric and parameters, not just textbook citations.

Deliverable: a fully reproducible notebook (`notebooks/exploration.ipynb`) and Python modules (`src/preprocessing.py`, `src/registration.py`, `src/metric_sweep.py`) producing standardized volumes, warped CTs, transforms, and per-subject quality logs on disk.

---

## 2. Tasks delivered

### Task 1 — Preprocessing pipeline

**Goal:** make every raw volume registration-ready (consistent orientation, isotropic spacing, sane HU range), preserving full provenance.

**Acceptance criteria — all met:**

- All 31 raw volumes reoriented to **RAS** (handles original L,A,S / R,A,S / P,S,R / R,S,A orientations)
- All resampled to **1 mm isotropic** voxel size
- CT volumes clipped to soft-tissue HU range `[-1000, 3000]`
- Processed outputs written to `data/processed/sub-XXXXX/ses-{preop,postop}/anat/sub-XXXXX_space-RAS_<MOD>.nii.gz`
- QC log per file: `results/sprint2_preprocessing_qc.csv` (62 rows, all `flag="ok"`)
- Per-step demo on sub-15454 + 3-view ortho display for all 10 patients in the notebook

**Code:** `src/preprocessing.py` — functions `find_subject_files`, `reorient_to_ras`, `resample_isotropic`, `clip_ct_hu`, `quality_check`, `preprocess_subject`, `preprocess_all`.

**Library:** ANTs (antspyx). NIfTI I/O via `ants.image_read` / `ants.image_write`.

### Task 2 — Baseline rigid registration

**Goal:** register every patient's post-op CT to their pre-op T1 MRI using a 6-DOF rigid transform.

**Acceptance criteria — all met:**

- Direction follows project proposal: T1 = **fixed**, CT = **moving**
- Rigid transform (3 rotations + 3 translations, no scaling/shearing)
- Mattes Mutual Information as the optimization metric (validated empirically in Task 3)
- Warped CTs + transforms saved per subject under `data/processed/sub-XXXXX/registration/`
- Registration log written to `results/sprint2_registration_log.csv` with `final_metric` (Mattes MI) and `runtime_s` columns
- 3×3 ortho overlay (T1 / warped CT / fused) rendered for every subject in the notebook

**Code:** `src/registration.py` — `register_subject`, `register_all`, `RegistrationResult` dataclass.

**Library:** ANTs (antspyx). `ants.registration(type_of_transform="Rigid", aff_metric="mattes", ...)`.

**Per-subject Mattes-MI summary** (lower is better; ANTs convention):

| Subject | Mattes MI | Runtime (s) | Notes |
|---|---|---|---|
| 15454 | -0.1010 | ~25 | Demo subject — good alignment |
| 39468 | -0.2886 | ~22 | NaN-fill artefact in raw CT (~21 % of volume); excluded from cohort by team agreement |
| 7 others | -0.07 to -0.11 | 18–28 | All converged cleanly |

### Task 3 — Metric & parameter selection

**Goal:** run an empirical sweep of similarity metrics and registration parameters on a sample subject (sub-15454), record evidence in a CSV, and lock the validated configuration into `src/registration.py`.

**Acceptance criteria — all met:**

- At least three metrics tested: Mattes MI, Joint Histogram MI, Correlation (negative control)
- Parameters varied across 9 configs: bins (32 / 50 / 64), sampling strategy (RANDOM / REGULAR), learning rate (0.5 / 1.0), iterations (100 / 200), interpolator (Linear / BSpline)
- Comparison table saved to `results/sprint2_metric_comparison.csv` (9 rows, criteria-spec columns + uniform-yardstick column)
- Notebook markdown cell explaining why MI is appropriate for multimodal MRI ↔ CT
- Notebook figure comparing best Mattes config vs Correlation negative control (3-view × 2-config overlay)
- Chosen config locked into `src/registration.py` as default arguments to `register_subject` / `register_all`
- Co-sign line in cell 48 reserved for Catarina (pending as of 2026-04-30)

**Code:** `src/metric_sweep.py` — SimpleITK-based runner (`run_metric_sweep`, `compare_two_configs`, `SweepConfig`, `SWEEP_GRID`).

**Library choice:** SimpleITK (the acceptance-criteria metric names — `SetMetricAsMattesMutualInformation`, `SetMetricAsJointHistogramMutualInformation`, `SetMetricAsCorrelation` — are SimpleITK API methods; ANTs only exposes Mattes cleanly). Validated knobs that map back to ANTs (`bins → aff_sampling`, `sampling % → aff_random_sampling_rate`, `iterations → aff_iterations`) were transferred to `src/registration.py`. SimpleITK-only knobs (learning rate, registration-time interpolator) are documented in cell 48 but not transferred — ANTs uses its own optimizer machinery.

---

## 3. Mid-sprint Catarina-feedback round (commit `11f2aa0`)

After the first review pass, Catarina flagged two issues over WhatsApp:

| Issue | Root cause | Fix |
|---|---|---|
| **Slice alignment** — lateral ventricles visible in CT slice but not MRI slice when shown side-by-side | Pre-op MRI and post-op CT come from different scanners with **non-overlapping world coordinate frames** (sub-15454: only 65 mm Z-overlap, 0 mm Y-overlap). The previous "identity overlay" column was trying to display them in a shared frame *before* registration — physically impossible. | Notebook cell rewritten to a **2-column native-grid view** with a printout of per-axis world-coordinate ranges and overlap. Misleading overlay column removed. The post-registration overlay already shows correct alignment after rigid registration. |
| **Mattes MI not visible in code** | Metric was applied via the function default (`DEFAULT_METRIC="mattes"`) but never named at the call site in the notebook | Cells updated to explicit `register_subject(..., metric="mattes")` and `register_all(..., metric="mattes")` so the metric is loud-and-clear in the demo. Mattes MI value now logged per subject. |

Both fixes shipped in commit `11f2aa0` on 2026-04-29. No code-behavior change — only diagnostic transparency.

---

## 4. Empirical findings worth presenting

From `results/sprint2_metric_comparison.csv` (9 configs on sub-15454):

1. **Iterations matter more than bins.** Varying bins 32 / 50 / 64 moved the Mattes-MI yardstick by < 0.002 (a tie). Doubling iterations 100 → 200 moved it 0.007 (~8 %), *and* ran faster (17 s vs 28 s) because the optimizer hits its early-stopping criterion before the budget runs out.
2. **Joint Histogram MI underperformed at our parameters** — yardstick -0.029 vs Mattes' -0.094. Theoretically equivalent, empirically worse without higher bins or more iterations. **Mattes MI is the safer default.**
3. **Correlation as a negative control behaved exactly as predicted** — visually clearly mis-aligned (CT skull off-axis vs T1) despite a numerical score (-0.075) that *looks* superficially comparable to MI scores. **Lesson: visual checks matter when comparing across metric families with different native scales.**
4. **BSpline interpolator at registration time is a strict net loss** here — equivalent quality at 4× runtime (74 s vs 17 s). Linear is the right choice.

**Locked-in defaults in `src/registration.py`** (committed):

```
DEFAULT_BINS = 50
DEFAULT_SAMPLING_RATE = 0.30
DEFAULT_ITERATIONS = (2100, 1200, 200, 0)   # final pyramid level bumped to 200
DEFAULT_METRIC = "mattes"
DEFAULT_TRANSFORM_TYPE = "Rigid"
```

---

## 5. Joint deliverables (both team members can demo)

- Reproducible end-to-end notebook (`notebooks/exploration.ipynb`, 49 cells, runs Run-All without error on sub-15454 + the 9 valid subjects)
- Three Python modules under `src/` documenting every step (`preprocessing.py`, `registration.py`, `metric_sweep.py`)
- Three machine-readable result tables under `results/` (preprocessing QC, registration log, metric comparison)
- Per-subject 3×3 ortho overlay figures rendered in-notebook for the full cohort
- Cell 48 rationale paragraph drafted in plain English for Catarina co-sign (pending)

---

## 6. Retrospective

**What went well:**

- Sprint 2 hit all three task acceptance criteria within the planned window, including the mid-sprint Catarina feedback round.
- Empirical-evidence-driven defaults (Task 3) are now baked into the code — future sprints inherit a validated configuration without re-debating it.
- The `# per agreement` convention introduced in earlier sprints continues to keep team-binding decisions visible at the code level.

**What was harder than expected:**

- The "pre-registration overlay" concept didn't survive contact with the data. Pre-op MRI and post-op CT from different scanners simply do not share a world frame — any "before-alignment" visualization that pretends they do is misleading. The fix (native-grid 2-column view) is more honest but less visually striking.
- ANTs silently converts NaN voxels to -1024 on load, which made our QC `nan_count` flag never fire for sub-39468 even though nibabel confirmed 7.4 M NaN voxels in the raw file. Caught only when Mattes MI came back as a 3× outlier (-0.29 vs -0.10 cohort norm). **Lesson:** add a raw-NIfTI nibabel-based pre-flight check before ANTs touches the file.
- Joint Histogram MI's underperformance was unexpected and currently unexplained at the chosen parameters. We chose Mattes empirically, but a deeper Sprint 4 pass should re-test joint-histogram with iter=200 + 64 bins.

**What we'll change:**

- Stage 1 of Sprint 3 will add an early **raw-data sanity sweep with nibabel** (NaN counts, finite fraction, zero-fraction) before any ANTs preprocessing — closes the QC blind spot above.
- All subsequent batch operations should default to `VALID_IDS` (cohort excluding sub-39468), not `PATIENT_IDS`. To be applied once Catarina formally co-signs the rationale and the cohort decision.

---

## 7. Open items / carry-over to Sprint 3

| # | Item | Owner | Status |
|---|---|---|---|
| 1 | Catarina co-sign on Task 3 rationale paragraph (cell 48) | Catarina | pending |
| 2 | Apply `VALID_IDS = [pid for pid in PATIENT_IDS if pid != "39468"]` to `src/config.py` and notebook batch calls | Samad | blocked on item #1 |
| 3 | Re-run `preprocess_all(VALID_IDS)` + `register_all(VALID_IDS)` after item #2 to refresh CSVs | Samad | blocked on item #2 |
| 4 | Confirm with Prof. Cunha which "final images" Lead-DBS expects (raw BIDS / preprocessed only / rigidly registered) | Samad | open question |
| 5 | Push Sprint 2 commits to origin (`6fa24b2` + `11f2aa0` + Task 3 commit) | Samad | ready locally |
| 6 | Add raw-NIfTI nibabel pre-flight QC step to preprocessing pipeline | Samad / joint | Sprint 3 backlog |
| 7 | Re-test Joint Histogram MI with iter=200 + 64 bins as a Sprint 4 deep-dive | Joint | Sprint 4 backlog |

---

## 8. File index

```
NeuroMappers/
├── notebooks/
│   └── exploration.ipynb              ← 49 cells, full Sprint 1 + Sprint 2 demo
├── src/
│   ├── config.py                      ← paths, PATIENT_IDS, modality catalogue
│   ├── utils.py                       ← logger + ensure_dir
│   ├── load_and_inspect.py            ← Sprint 1 helpers
│   ├── preprocessing.py               ← Sprint 2 Task 1
│   ├── registration.py                ← Sprint 2 Task 2 + locked-in Task 3 defaults
│   └── metric_sweep.py                ← Sprint 2 Task 3 (NEW)
└── results/
    ├── sprint1_data_inspection.csv    ← (Sprint 1)
    ├── sprint2_preprocessing_qc.csv   ← Task 1 output (62 rows)
    ├── sprint2_registration_log.csv   ← Task 2 output (10 rows)
    └── sprint2_metric_comparison.csv  ← Task 3 output (9 rows, NEW)
```

---

## 9. Sprint 2 status

| Task | Status |
|---|---|
| Task 1 — Preprocessing | done, batch-run on all 10 subjects |
| Task 2 — Baseline rigid registration | done, all 10 subjects aligned, Mattes-MI logged |
| Task 3 — Metric/parameter selection | code + evidence + locked defaults; pending Catarina co-sign |
| Mid-sprint Catarina feedback round | shipped in commit `11f2aa0` |

**Sprint 2 is functionally complete pending Catarina's sign-off and the GitHub push.**
