# SemEval 2026 Task 2: Valence & Arousal Prediction

This repository contains our system submission for **SemEval 2026 Task 2**, focused on predicting **valence** and **arousal** from textual data. Our approach combines **transformer embeddings**, **lexical and statistical features**, **user embeddings**, and **ensemble modeling**.

**Final Paper (PDF):** [Link to final paper](./FinalPaper.pdf)


## Key Features
- RoBERTa-based encoder using CLS embeddings for text representation
- User embeddings to incorporate author-level behavioral signal  
- Ensemble of 5 diverse constituent models (A, B, D, G, H), combining continuous, categorical, and ordinal-regression approaches
- Joint regression heads for valence and arousal prediction  


## Repository Structure

- **ClassDefinition/**
  - `AffectClassifier.py` — Ensemble model logic (Version A/B)
  - `Dataset.py` — Data loading + lexical/user feature extraction
  - `Entry.py` — Wrapper class for individual samples
  - `Roberta.py` — Transformer embedding module

- **Main/**
  - `main.py` — Training + evaluation entry point
  - `predict.py` — Inference script
  - `utils/` — Logging, argument parsing, metrics

- **Data/**
  - `TRAIN_RELEASE_3SEP2025/`

- `requirements.txt`
- `README.md`


 
### Set Project Root
export SEMROOT=/path/to/Semeval2026Group8

### Install all unmet dependencies
./installDependencies.sh 

### Run main.py
python Main/main.py --model_version A --epochs 10 --batch_size 16



## Model Overview
<img width="677" height="356" alt="Screenshot 2025-12-15 at 9 51 42 AM" src="https://github.com/user-attachments/assets/eb1181ba-6864-4369-8009-105d7b1c14df" />

## Evaluation

Metrics used in this work include:

- **Pearson’s R (primary metric)**
- **MAE (Mean Absolute Error)**
- **Precision**
- **Recall**
- **Accuracy**
- **F1 Score**

The ensemble model outperforms all individual constituent models across metrics, including MAE and F1, and achieves the strongest **Pearson’s R** values for both valence and arousal.


## 👥 Team
- **Troy Arthur**
- **Sierra Reschke**
- **Aidan Kelley**

