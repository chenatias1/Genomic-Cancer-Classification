# Genomic Cancer Classification (LUSC vs HNSC)

Machine learning project for classifying squamous cell lung cancer (LUSC) and head & neck squamous cell carcinoma (HNSC) using genomic mutation and methylation profiles.

## Project Overview

This project builds a Random Forest classifier to distinguish between two cancer types based on genomic features derived from:

- Somatic mutation profiles (100 cancer-related genes)
- DNA methylation data
- Statistical and biologically-informed feature engineering

The pipeline includes cross-validation, greedy feature selection, and final model training on selected features.

---

## Feature Engineering

### Mutation-Based Features
- Total mutation burden
- Variant type counts
- Gene–variant combinations
- High-impact mutation indicators (TP53, MUC16, MUC4)
- Gene mutation entropy (Shannon entropy)
- Mutation dispersion statistics (std, variance)
- Transversion ratios (C>A, G>T, A>T)
- Chromosome-level mutation counts

### Methylation-Based Features
- Mean β-value per gene
- Standard deviation per gene
- CpG density
- Gene methylation entropy
- Hyper/Hypo-methylation proportions

---

## Model

- Algorithm: Random Forest
- Feature Selection: Greedy selection based on validation error
- Evaluation: 5-fold cross validation
- Final model selected based on minimal validation-test error gap

---

## Technologies

- Python 3.9+
- Pandas
- NumPy
- SciPy
- Scikit-learn
- Matplotlib

---
