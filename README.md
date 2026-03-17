# Predicting FDA Medical Device Adverse Events with Heterogeneous Graph Neural Networks

[![Medium](https://img.shields.io/badge/Medium-Blog_Post-black)](https://medium.com/@mfarring/predicting-fda-medical-device-adverse-events-with-heterogeneous-graph-neural-networks-f3b4fc46a941)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation, datasets, and experimental notebooks for predicting post-market medical device adverse events using Multimodal Heterogeneous Graph Neural Networks. This work was originally developed for Stanford CS224W (Machine Learning with Graphs).

## Abstract
Post-market surveillance of medical devices relies heavily on the FDA's Medical Device Reporting (MDR) ecosystem, which contains noisy metadata and unstructured clinical narratives. This project reformulates adverse event detection as a multimodal link prediction task on a heterogeneous graph. By integrating structured categorical data with Freedom of Information (FOI) text embeddings via Sentence-BERT, we benchmark models to predict *report-mentions-device* associations. Our results demonstrate that Heterogeneous Graph Transformers (HGT) successfully capture the complex topology of manufacturers and device product codes, significantly outperforming traditional GCN architectures.

## Graph Schema & Methodology
We construct a directed, heterogeneous graph utilizing openFDA MAUDE data:

**Nodes:**
* **Report:** Unique MDR reports.
* **Device:** Canonicalized composite keys.
* **Manufacturer:** Normalized manufacturer entities.
* **Event:** Unique device event codes.

**Edges (including explicit reverse relations):**
* `report` $\rightarrow$ `mentions` $\rightarrow$ `device`
* `manufacturer` $\rightarrow$ `makes` $\rightarrow$ `device`
* `event` $\rightarrow$ `involves` $\rightarrow$ `device`

**Multimodal Features:** Node features combine categorical indices (brand, generic name) with 384-dimensional Sentence-BERT (`all-MiniLM-L6-v2`) embeddings of clinical narratives.

## Results
The models were evaluated on a strict chronological split (January 2024 snapshot) to prevent temporal leakage. The HGT architecture achieves the highest predictive performance:

| Model | Test AUROC | Test F1-Score |
| :--- | :--- | :--- |
| Simple HAN | 0.299 | 0.000 |
| R-GCN | 0.482 | 0.062 |
| **HGT** | **0.704** | **0.811** |

## Repository layout 
```
.
├── notebooks/
│   └── cs224w_final_project_notebook.ipynb     
├── data/                              
├── outputs/                            
├── assets/                             
├── scripts/                         
├── requirements.txt
└── .gitignore
```

## Quickstart 
1. Open `notebooks/cs224w_final_project_notebook.ipynb` in Colab.
2. Enable GPU: Change runtime type → T4/A100 GPU
3. Run cells top-to-bottom.

### Versions tested
The notebook was validated on Colab with:
- **PyTorch** `2.3.1+cu121`
- **PyTorch Geometric** `2.6.1`
- CUDA available 

## Data
This project uses **FDA MAUDE MDR downloadable files** (pipe-delimited text in `.zip` archives), including:
- **Device data for 2024**: `device2024.zip`
- **Narrative text for 2024**: `foitext2024.zip`

These files are listed on FDA’s “MDR Data Files” page.  
See `data/README.md` for download instructions and where to place files.

## Experimental setup (as used in the notebook)
- **Window:** reports received **2024-01-01 to 2024-01-30** (January 2024 slice)
- **Split:** report-level chronological split (~70/15/15) to reduce temporal leakage
- **Objective:** binary classification on edges (positive report–device edges vs negative samples)
- **Metrics:** AUROC + F1 (threshold tuned on validation, applied to test)
- **Optimizer:** AdamW + early stopping on validation AUROC

Key default hyperparameters (see notebook constants):
- `EMB = 16`
- `HGT_HEADS = 1`
- `LR = 1e-3`
- `WD = 1e-4`
- `batch_size = 256`

## Reproducibility notes
- Downloaded FDA data is **not** committed. Place it under `data/raw/` (see `data/README.md`).
- Sentence-BERT embeddings are computed with `sentence-transformers` (model: `all-MiniLM-L6-v2`).
- Results may vary slightly by random seed and negative sampling.

## Citation
If you use this repo, please cite the blog post and/or this repository.

## License
MIT (see `LICENSE`).
