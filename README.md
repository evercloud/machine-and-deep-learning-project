# CTG Classification Project

Minimal local setup guide to run the project notebook from scratch.

## Project Files

- `CTG_Classification_Project.ipynb`: main notebook (full pipeline)
- `CTG.csv`: dataset used by the notebook
- `Relazione_CTG_Classification.pdf`: project report

## Requirements

- Python 3.10+ (3.11 recommended)
- `pip`

## Quick Start

1. Open a terminal in this folder.
2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow shap notebook
```

4. Launch Jupyter Notebook:

```bash
jupyter notebook
```

5. Open `CTG_Classification_Project.ipynb` and run all cells from top to bottom.

## What You Should See

- Data exploration plots
- PCA analysis and dimensionality reduction
- Training and evaluation of 3 shallow classifiers
- Confusion matrices and ROC curves
- Deep learning training/evaluation
- SHAP explainability plots

Generated images are saved in `outputs/` (e.g. `outputs/01_eda.png`, `outputs/03_confusion_matrices.png`, `outputs/08_rf_feature_importance.png`, etc.).

## Troubleshooting

- If TensorFlow install fails on your machine, first run:

```bash
pip install --upgrade pip setuptools wheel
```

- If SHAP plots fail to render, ensure `matplotlib` is installed and rerun the SHAP cells.
- If kernel errors appear in Jupyter, restart kernel and run all cells again.

## Reproducibility Notes

- The notebook uses fixed random seeds where relevant.
- Keep the same execution order (top to bottom) to avoid missing variables/state.
