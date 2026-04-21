# NeuroMappers

**NeuroMappers** is a biomedical image analysis project focused on the precise localization of **Deep Brain Stimulation (DBS)** electrodes through the fusion of **pre-operative MRI** and **post-operative CT** neuroimaging data.

The project is being developed in the context of the **Biomedical Image Analysis (BMIA)** course and aims to build a technical pipeline for **multimodal registration, image fusion, electrode localization, and anatomical interpretation**.

---

## Project Overview

Deep Brain Stimulation (DBS) is a surgical treatment used for neurological disorders such as **Parkinson’s disease**, **essential tremor**, and **dystonia**. The clinical effectiveness of DBS depends strongly on the **accurate placement of electrodes** within small subcortical targets.

This project combines:

- **Pre-operative MRI**, which provides detailed anatomical information and target visualization
- **Post-operative CT**, which clearly shows the implanted DBS electrodes

Since these modalities provide complementary information, they must be **registered and fused** in order to localize the electrode accurately within anatomical brain space.

---

## Main Objective

The main objective of this repository is to develop a reproducible pipeline for:

1. Loading and inspecting pre-operative MRI and post-operative CT data
2. Preprocessing both modalities
3. Performing multimodal MRI-CT registration
4. Generating fused visualizations for quality control
5. Detecting DBS electrodes in CT
6. Estimating electrode trajectory and contact positions
7. Mapping electrode localization into MRI anatomical space
8. Evaluating registration and localization quality

---

## Technical Workflow

<p align="center">
  <img src="assets/dbs_workflow_a4_spacious.png" alt="Initial Technical Workflow for DBS Electrode Localization" width="700">
</p>

<p align="center">
  <em>Figure 1. Proposed native-space technical workflow for the DBS electrode localization project.</em>
</p>

## Technical Workflow

The current implementation plan follows this workflow:

1. **Data Loading and Inspection**
2. **Preprocessing**
3. **MRI-CT Multimodal Registration**
4. **Fusion and Quality Control**
5. **Electrode Detection from CT**
6. **Trajectory and Contact Localization**
7. **MRI-space Anatomical Mapping**
8. **Evaluation and Final Outputs**

---

## Repository Structure

```text
NeuroMappers/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── exploration.ipynb
│
├── results/
│   ├── figures/
│   ├── masks/
│   ├── metrics/
│   ├── qc/
│   └── transforms/
│
├── src/
│   ├── load_and_inspect.py
│   ├── visualization.py
│   ├── preprocessing.py
│   ├── registration.py
│   ├── quality_control.py
│   ├── electrode_detection.py
│   ├── trajectory.py
│   ├── contact_localization.py
│   ├── anatomical_visualization.py
│   ├── evaluation.py
│   ├── utils.py
│   └── config.py
│
├── main.py
├── requirements.txt
└── README.md
