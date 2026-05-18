# Predicting Clonal Expansion of Tumor-Infiltrating T Cells from scRNA-seq

**Camryn Franke**

**Deep Learning in Genomics Final Project** 

**May 18 2026**

## Question
Can gene expression alone predict whether a T cell belongs to an *expanded* TCR clonotype (≥2 cells sharing the same clonotype) versus a singleton?

## Hypothesis
Expanded tumor-infiltrating T cells occupy distinct transcriptional states, enriched for cytotoxic, exhausted, activated, or tissue-resident programs. A model trained on gene expression should be able to recover these programs without ever seeing the TCR sequence.

## Data
**GSE139555**: Wu et al., *Nature* 2020, "Peripheral T cell expansion predicts tumour infiltration and clinical response."

- 14 treatment-naive cancer patients
- 4 cancer types: lung adenocarcinoma, endometrial adenocarcinoma, colorectal adenocarcinoma, renal clear cell carcinoma
- Tissue: tumor, normal-adjacent (NAT), peripheral blood
- 141,623 T cells with paired scRNA-seq + TCR-seq

Files used (downloaded automatically in nb01):
- `GSE139555_RAW.tar` : per-sample cellranger MTX matrices (download directly from GEO)
- `GSE139555_tcell_metadata.txt.gz` : per-cell metadata including TCR clonotype, cell-state cluster (`ident`), tissue (`source`), patient

## Pipeline

| Notebook | Step | Output |
|---|---|---|
| `01_download_and_explore.ipynb` | Download from GEO, load MTX per sample, merge with TCR metadata | `data/raw_combined.h5ad` |
| `02_preprocessing.ipynb` | QC, normalize, select 2,000 HVGs, build `expanded` label, patient-stratified train/val/test split | `data/processed.h5ad` |
| `03_train_models.ipynb` | Train + evaluate models: LogReg, MLP, supervised autoencoder (SAE), XGBoost. Includes HP sweep and the additional SAE experiments that were tried. | `results/predictions.pkl`, `results/models/{mlp,sae}.pt`, comparison CSVs |
| `04_multitask_model.ipynb` | Multitask deep model: shared encoder with three heads (expanded / clone-size bin / cell state). Adds the multitask row to `predictions.pkl`. | `results/multitask_predictions.pkl`, `results/models/multitask.pt` |
| `05_evaluation_interpretation.ipynb` | All evaluation figures: model comparison, ROC/PR, stratified performance, clone-size calibration, SAE + multitask latent UMAPs, gene importance (standardized LogReg + input×gradient + XGBoost gain), cross-model consensus, biology overlap. | `results/figures/*.png`, gene importance CSVs |

## Models compared

| Model | Type | Why |
|---|---|---|
| **Logistic regression** | Linear, sklearn | Reference for how much non-linearity buys |
| **MLP** | Single-task DL | Off-the-shelf neural baseline |
| **Supervised autoencoder** | DL: shared encoder + classifier + reconstruction | Tests whether unsupervised reconstruction regularizes the latent usefully |
| **XGBoost** | Gradient-boosted trees | Often the strongest tabular model for genomics |
| **Multitask model** | DL: shared encoder + 3 supervised heads | Biologically grounded multitask regularization (expansion + clone-size + cell state) |

Each DL model has a hyperparameter sweep selected on **val AUROC**; test set is touched only when running the winning config.

## Repo layout

```
final_project/
├── README.md
├── requirements.txt
├── data/                                 
├── notebooks/
│   ├── 01_download_and_explore.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_train_models.ipynb
│   ├── 04_evaluation_interpretation.ipynb
│   └── 05_multitask_model.ipynb
├── src/                            # reusable functions+modules imported by notebooks
│   ├── data.py                           # GEO download, MTX loading, label/split construction, QC, HVG
│   ├── models.py                         # MLPClassifier, SupervisedAutoencoder, MultiTaskModel
│   ├── train.py                          # training loops, predict_*, encode_latents
│   └── evaluate.py                       # metrics, ROC/PR/UMAP plots, gradient importance
└── results/                              # figures, model checkpoints, metrics
```

## Notes on Reproducibility

- Random seeds fixed in `src/train.py:set_seed`.
- Train/val/test splits assigned **by patient** (not per cell) to avoid clonotype leakage.
- Hyperparameter sweeps select on val AUROC; test set is only touched for final reported metrics.
- All model configs, predictions, and split assignments saved into `results/` so plots can be regenerated without retraining.

## Key references

- Wu, T.D., Madireddi, S., de Almeida, P.E. et al. *Peripheral T cell expansion predicts tumour infiltration and clinical response.* Nature 579, 274–278 (2020). https://doi.org/10.1038/s41586-020-2056-8
- Wolf, F.A., Angerer, P., Theis, F.J. *SCANPY: large-scale single-cell gene expression data analysis.* Genome Biology 19, 15 (2018).
- Chen, T., Guestrin, C. *XGBoost: A Scalable Tree Boosting System.* KDD 2016.
