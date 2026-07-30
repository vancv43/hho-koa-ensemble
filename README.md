# HHO-KOA-Ensemble

### Heterogeneous Multi-Backbone Ensemble with Harris Hawks Optimization for Knee Osteoarthritis Severity Grading

**Repository:** https://github.com/vancv43/hho-koa-ensemble

Reference implementation accompanying the manuscript:

> **Heterogeneous Multi-Backbone Ensemble with Harris Hawks Optimization of an Automated Knee Osteoarthritis Severity Grading System: A Dual-Dataset Study on Radiographic and Clinical Data**
> Van Van Chau, Minh Tuan Nguyen, Phayung Meesad
> *Journal of Data Science and Intelligent Systems (JDSIS)*, 2026 (under review)

This repository provides a clean, documented reference implementation of the method described in the paper: four heterogeneous CNN backbones fine-tuned with a combined CORN + Boundary-Sensitive Focal Loss (BSFL), an ensemble soft-voting weight search via Harris Hawks Optimization (benchmarked against PSO and GA), test-time augmentation, and an HHO-based feature-selection branch for a tabular clinical pathway.

**This is a methodology reference, not a copy of the lab notebook used for the submission.** It is written to match the equations, algorithms, and hyperparameters reported in the paper (Sections 3-4) so that the described method can be understood and re-implemented; it is not a byte-for-byte reproduction of the exact training run. See [Notes on scope](#notes-on-scope) below.

## Clone

```bash
git clone https://github.com/vancv43/hho-koa-ensemble.git
cd hho-koa-ensemble
```

---

## Repository structure

```
.
├── config.py                       # hyperparameters (match manuscript Table 3)
├── train.py                        # Phase 1: single-backbone fine-tuning + LLRD
├── ensemble_optimize.py            # Phase 2: HHO/PSO/GA ensemble weight search
├── evaluate.py                     # test metrics + bootstrap significance test
├── src/
│   ├── models/backbones.py         # CORN head wrapper around timm backbones
│   ├── losses/corn_bsfl.py         # combined CORN + BSFL loss
│   ├── optim/{hho,pso,ga}.py       # ensemble weight optimizers
│   ├── data/dataset.py             # OAI DataLoader + augmentation + oversampling
│   ├── clinical/feature_selection.py  # HHO wrapper feature selection + SVM
│   └── metrics.py                  # QWK, balanced accuracy, macro-AUC
└── results/tables/                 # published aggregate results (CSV), matching the paper's tables
```

## Setup

```bash
pip install -r requirements.txt
```

## Data

**Not included in this repository.**

- **Radiographic data**: this study uses the Osteoarthritis Initiative (OAI) cohort. OAI imaging data is distributed under its own data use agreement and must be requested directly from the OAI (https://nda.nih.gov/oai/). Point `KOA_DATA_ROOT` (or `--data-root`) at a local export organized as `train/`, `val/`, `test/`, each with one subfolder per KL grade (`0`-`4`).
- **Clinical Data-2** (160-patient tabular cohort): this is confidential patient data and is not distributed with this repository (see the manuscript's Ethical Statement). `src/clinical/feature_selection.py` operates on a generic `(X, y)` array so it can be run against any feature matrix of the same shape.
- **Model checkpoints and cached predictions are not included** in this repository.

## Usage

```bash
# Phase 1 - fine-tune each backbone
python train.py --backbone densenet121      --data-root /path/to/oai
python train.py --backbone efficientnetv2_s --data-root /path/to/oai
python train.py --backbone resnet50         --data-root /path/to/oai
python train.py --backbone convnext_tiny    --data-root /path/to/oai   # LLRD applied automatically

# Phase 2 - ensemble weight search (expects one cached <backbone>.npy
# softmax-probability array per model on the validation set)
python ensemble_optimize.py --probs-dir ./cached_val_probs --labels val_labels.npy

# Evaluation - full metrics + bootstrap significance test on the test set
python evaluate.py --probs-dir ./cached_test_probs --labels test_labels.npy \
    --weights 0.307 0.231 0.238 0.224
```

## Method summary

| Component | Details |
|:----------|:--------|
| Backbones | DenseNet-121, EfficientNetV2-S, ResNet-50, ConvNeXt-Tiny (ImageNet-1K pretrained) |
| Ordinal head | CORN (Conditional Ordinal Regression for Neural Networks) |
| Loss | 60% CORN + 40% Boundary-Sensitive Focal Loss, 4x KL-1 class weight |
| Fine-tuning | AdamW, cosine schedule; LLRD (6 groups, decay 0.65) for ConvNeXt-Tiny only |
| Ensemble weights | Harris Hawks Optimization on validation QWK (benchmarked against PSO, GA) |
| Test-time augmentation | 8 deterministic geometric/brightness variants |
| Clinical branch | HHO binary feature selection (67 -> 12 features) + RBF-SVM |

## Paper ↔ code mapping

| Manuscript | Location in this repository |
|:-----------|:-----------------------------|
| Eq. (1)-(2): CORN ordinal head & decoding rule | `src/models/backbones.py` |
| Eq. (3)-(5): combined CORN + BSFL loss | `src/losses/corn_bsfl.py` |
| Section 3.4, Eq. (8): LLRD for ConvNeXt-Tiny | `train.py::build_llrd_param_groups` |
| Algorithm 1, Eq. (6)-(7): HHO ensemble weight search | `src/optim/hho.py`, `ensemble_optimize.py` |
| Section 4.4 (PSO/GA baselines) | `src/optim/pso.py`, `src/optim/ga.py` |
| Section 3.6: clinical HHO feature selection + SVM | `src/clinical/feature_selection.py` |
| Table 3: hyperparameters | `config.py` |
| Table 5: full evaluation (backbones vs. ensemble vs. SOTA) | `results/tables/main_results.csv` |
| Table 6: HHO/PSO/GA optimiser comparison | `results/tables/optimizer_comparison.csv` |
| Table 7: bootstrap significance test | `results/tables/bootstrap_validation.csv`, `evaluate.py::bootstrap_qwk_gain` |
| Table 8: ablation study | `results/tables/ablation.csv` |
| Table 9: clinical branch (HHO+SVM vs. full-feature SVM) | `results/tables/clinical_branch.csv` |

## Published results (see `results/tables/`)

Headline test-set result: **QWK = 0.8279, macro-AUC = 0.8909** (HHO-Ensemble + TTA, OAI test set, strict subject-level separation). Full tables, including the ablation study and the paired-bootstrap significance test for the HHO-vs-uniform-weighting gain, are in `results/tables/`.

## Citation

If you use this code, please cite the paper (citation details to be updated upon publication) and, optionally, this repository:

```bibtex
@article{chau2026koahho,
  title   = {Heterogeneous Multi-Backbone Ensemble with Harris Hawks Optimization of an Automated
             Knee Osteoarthritis Severity Grading System: A Dual-Dataset Study on Radiographic and
             Clinical Data},
  author  = {Chau, Van Van and Nguyen, Minh Tuan and Meesad, Phayung},
  journal = {Journal of Data Science and Intelligent Systems},
  year    = {2026}
}

@misc{hhokoaensemble2026,
  title  = {HHO-KOA-Ensemble: reference implementation},
  author = {Chau, Van Van and Nguyen, Minh Tuan and Meesad, Phayung},
  year   = {2026},
  url    = {https://github.com/vancv43/hho-koa-ensemble}
}
```

## Notes on scope

- The HHO implementation here covers the exploration phase and soft-besiege exploitation; the hard-besiege and Levy-flight dive variants used in the full experimental run are omitted for brevity.
- `train.py` illustrates the exact loss, optimizer, and LLRD schedule from the paper, but omits some incidental engineering details (checkpoint management, mixed precision, logging) that do not affect the reported method.
- Raw per-image predictions, trained weights, and the confidential clinical dataset are not released. Aggregate results already published in the manuscript are provided in `results/tables/` for reference.

## License

TBD by the authors.
