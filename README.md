# Alfalfa Causal Inference & GWAS Pipeline

This repository contains a Python-based pipeline for analyzing Alfalfa genotype-phenotype data, as described in our paper: "[Disentangling direct and pleiotropic SNP effects in alfalfa (Medicago sativa L.) using causal graph learning](https://www.nature.com/articles/s41598-026-35876-w)" 

The workflow integrates Double Machine Learning (DoubleML) for GWAS, ensemble feature selection, and Causal Discovery algorithms (PC Algorithm) to construct and compare causal graphs. It includes a comprehensive suite of visualization tools for generating publication-quality figures (LaTeX/PGF).

## 📂 File Structure & Description

### 1. Feature Selection & GWAS
* **`feature_select_lasso_pca.py`**: Performs a "DoubleML GWAS Full Scan". Regresses Phenotypes against SNPs while controlling for population structure (PCs) using LassoCV.
* **`util_selection.py`**: Toolkit for ensemble feature selection, including Variance Thresholding, Collinearity filtering, PCA, SelectKBest, RFE, and Random Forest importance.
* **`verify_single_snp.py`**: A diagnostic tool to verify DoubleML results for a specific SNP. It performs a detailed fit, reports coefficients, and calculates $R^2$ metrics to validate if Principal Components effectively control for confounding.
* **`visualize_feature_select.py`**: Generates publication-ready GWAS figures using a LaTeX-compatible PGF backend.
    * *Outputs*: Manhattan Plots, QQ Plots (with $\lambda_{GC}$), Volcano Plots, and Effect Size plots.
* **`analyze_feature_selection.py`**: A simpler alternative script for basic GWAS result visualization (Manhattan, QQ, Volcano).

### 2. Causal Discovery (Graph Construction)
* **`causal_dis.py`**: The main driver for Causal Discovery. Loads top SNPs, runs the PC Algorithm (using `causal-learn`), prunes the graph to trait ancestors, and quantifies edge weights.
* **`alfafa_causal.py`**: An alternative implementation using `gpucsl` for GPU-accelerated causal discovery.
* **`load_data.py`**: Helper module to load `geno.csv` and `pheno.csv`, supporting data pruning based on feature selection scores.

### 3. Graph Visualization & Analysis
* **`visualize_cg_sub.py`**: The main entry point for visualizing causal graphs. It supports Venn diagram generation to compare feature selection overlap and renders graphs using advanced layouts.
* **`util_vis.py`**: Visualization library supporting `visualize_cg_sub.py`.
    * *Key Feature*: **Sunflower Layout** (`vis_cg_sunflower`)—a spiral layout algorithm organizing SNPs around the trait based on connection strength, with chromosome-specific coloring.
* **`cg_compare.py`**: Compares causal mechanisms between two traits (e.g., Stem Color vs. Strength) to identify pleiotropy and shared genetic drivers.
* **`cg_util.py`**: Utility library for graph manipulation, including converting CausalLearn graphs to NetworkX and assigning edge weights via CIT.
* **`util_cg.py`**: Additional utilities for trimming ancestors and saving NetworkX graphs.

### 4. Utilities
* **`test_eps_nx.py`**: A test script for validating graph visualization export to PGF/LaTeX formats.

---

## 🚀 Workflow

### Step 1: Run DoubleML GWAS
Perform the association study to generate P-values and Coefficients.
```bash
python feature_select_lasso_pca.py --trait stem_color --output feature_stem_color_100it.csv
