# Predicting FDA Medical Device Adverse Events with Heterogeneous GNNs (CS224W)

This repo contains my **Stanford CS224W final project**: building a heterogeneous graph from FDA MDR (MAUDE) device + narrative data and training heterogeneous GNNs to predict **report–device links**.

**Blog post (walkthrough):** <PUT_MEDIUM_LINK_HERE>

## What this project does
- Builds a heterogeneous graph with node types:
  - **Report** (MDR_REPORT_KEY)
  - **Device** (canonicalized composite key)
  - **Manufacturer** (normalized name)
  - **Event** (DEVICE_EVENT_KEY, when present)
- Adds edge types (and explicit reverse edges) such as:
  - report → mentions → device
  - manufacturer → makes → device
  - event → involves → device
- Trains and evaluates heterogeneous GNNs for **link prediction** on *(report, mentions, device)*:
  - **R-GCN**
  - **HAN** (simple/custom)
  - **HGT**

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
