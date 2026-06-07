# SDG 3 Indicator Text Classification: Group 5

Multi-label text classification system for 27 SDG 3 health indicators, built for the Machine Learning Techniques I assignment at African Leadership University.

**Demo Video:** [Watch on Google Drive](https://drive.google.com/file/d/1-aWxfTLisQ6VlAcaox8NKTLLPZTkiXnc/view?usp=sharing)
**Report Link:** [Report](https://docs.google.com/document/d/1Sj_41-zrugwPv_A-4aDn5ny2ooQ3u1iDBubBlx3F6_w/edit?usp=sharing)

## Problem Statement

Given a text sample from development reports, tenders, humanitarian initiatives, or news articles, predict which of the 27 SDG 3 health indicators are relevant. This is a multi-label classification problem where a single document can belong to multiple indicators simultaneously.

**Evaluation metric:** Hamming Loss (lower is better)
**Best result:** 0.0459 using Logistic Regression (C=10.0, threshold=0.45)

## Repository Structure

```
SGD_Group_5/
├── notebooks/
│   └── SGD_Group_5.ipynb        # full pipeline: EDA, preprocessing, feature engineering, all experiments
├── inference/
│   └── best_model.ipynb         # standalone inference on Devex_test_questions.csv
│   └── best_model.py            # Python script version of inference pipeline
├── data/
│   └── Devex_train.csv          # training data (not committed, download from Canvas)
│   └── Devex_test_questions.csv # test data (not committed, download from Canvas)
└── README.md
```

## Experiments Summary

| # | Model | Configuration | Hamming Loss |
|---|-------|--------------|-------------|
| 1 | Logistic Regression | TF-IDF baseline, full vocab | 0.0557 |
| 2 | Logistic Regression | TF-IDF, stopwords retained | 0.0571 |
| 3 | Logistic Regression | Vocab cap + Type OHE | 0.0540 |
| 4 | Multinomial NB | Baseline, alpha=1.0 | 0.0661 |
| 5 | Multinomial NB | Alpha sweep, best alpha=0.1 | 0.0605 |
| 6 | Multinomial NB | Threshold tuning, t=0.60 | 0.0590 |
| 7 | Logistic Regression | C=1.0 baseline | 0.0531 |
| 8 | Logistic Regression | class_weight='balanced' | pending |
| 9 | Logistic Regression | C sweep, best C=10.0 | pending |
| 10 | Logistic Regression | Threshold tuning, t=0.45 | **0.0459** |
| 11 | GRU | Sequential model | pending |
| 12 | RNN | Baseline | pending |
| 13 | Linear SVC | OvR wrapper | pending |
| 14 | BERT | Fine-tuned, Experiment 7 | 0.1078 |

## Setup

All notebooks run on Google Colab. No local setup required. Open the notebook and mount your Google Drive when prompted.

**Dependencies** (pre-installed on Colab, or install via pip):

```
scikit-learn
pandas
numpy
scipy
nltk
beautifulsoup4
joblib
torch
transformers
```

Or install all at once:

```bash
pip install scikit-learn pandas numpy scipy nltk beautifulsoup4 joblib torch transformers
```

## Running Inference

To generate predictions on `Devex_test_questions.csv`:

1. Open the inference notebook on Colab: [best_model.ipynb](https://colab.research.google.com/drive/1rYdp2rxynxswMCTLWVZTKspoZ4mHTa8i?usp=sharing)
2. Mount your Google Drive
3. Update the four path constants at the top:

```python
CLASSICAL_PATH = '/content/drive/MyDrive/SGD/classical_model_train_test/'
TRAIN_CSV      = '/content/drive/MyDrive/SGD/Devex_train.csv'
TEST_CSV       = '/content/drive/MyDrive/SGD/Devex_test_questions.csv'
OUTPUT_CSV     = '/content/drive/MyDrive/SGD/test_predictions.csv'
```

4. Run all cells. Predictions are saved to `OUTPUT_CSV`.

**Required Drive artifacts:**
- `tfidf.pkl` (fitted TF-IDF vectorizer from the training pipeline)
- `Devex_train.csv` (original training data)

The script reconstructs the OHE and ROS balancing from scratch so no other saved artifacts are needed.

## Team Contributions

| Member | Contributions |
|--------|--------------|
| Winston | Preprocessing pipeline, feature engineering, Transformer and BERT experiments |
| Justine | Classical ML: Logistic Regression and Multinomial Naive Bayes (LR-1 to LR-4, MNB-1 to MNB-3) |
| David | Exploratory data analysis, GRU model |
| Kumi | Linear SVC, RNN model |
