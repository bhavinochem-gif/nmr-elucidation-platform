# nmr-elucidation-platform
# 🧪 NMR Elucidation Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![RDKit](https://img.shields.io/badge/Cheminformatics-RDKit-green?style=flat-square)](https://www.rdkit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)
[![Tests](https://img.shields.io/badge/Tests-Passing-success?style=flat-square&logo=github-actions)](https://github.com/)

An automated **Computer-Assisted Structure Elucidation (CASE)** engine designed to extract, correlate, and solve molecular structures directly from raw 1D and 2D NMR experimental data.

---

## ⚡ Key Features

* **Multi-Format Ingestion:** Native parsing of Bruker, Varian/Agilent, and JCAMP-DX raw datasets via `nmrglue`.
* **Spectral Decomposition:** Automated phase correction, baseline flattening, peak picking, and multiplet deconvolution.
* **2D Correlation Engine:** Cross-validates spin networks across $^1\text{H}$-$^1\text{H}$ COSY, $^1\text{H}$-$^{13}\text{C}$ HSQC, and $^1\text{H}$-$^{13}\text{C}$ HMBC.
* **Structure Generation:** Graph-based molecular fragment assembly and candidate ranking using **RDKit** and chemical shift prediction models.
* **Interactive Visualization:** Export annotated assignments directly to interactive HTML or publication-ready vector figures.

---

## 🏗️ Workflow Architecture

```text
  Raw NMR FIDs (1D/2D)
           │
           ▼
  ┌─────────────────────────────────┐
  │ Preprocessing & Peak Extraction │ (nmrglue / scipy)
  └─────────────────────────────────┘
           │
           ▼
  ┌─────────────────────────────────┐
  │  Spin System & Graph Assembly   │ (COSY / HSQC / HMBC correlation)
  └─────────────────────────────────┘
           │
           ▼
  ┌─────────────────────────────────┐
  │  Candidate Ranking & Scoring    │ (RDKit / Shift Predictor)
  └─────────────────────────────────┘
           │
           ▼
  Validated SMILES & Mol Assignments
