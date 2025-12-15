# SemEval 2026 Task 2: Valence & Arousal Prediction

This repository contains our system submission for **SemEval 2026 Task 2**, focused on predicting **valence** and **arousal** from textual data. Our approach combines **transformer embeddings**, **lexical and statistical features**, **user embeddings**, and **ensemble modeling**.

**Final Paper (PDF):** [Link to final paper](./FinalPaper.pdf)

---

## Key Features
- RoBERTa-based text encoder for multilingual affect representation  
- Lexical + statistical feature extraction (e.g., affective lexicons, metadata)  
- User embeddings to incorporate author-level behavioral signal  
- Two ensemble model variants (`Version A` and `Version B`) implemented in a modular PyTorch architecture  
- Joint regression heads for valence and arousal prediction  
- Evaluation using **CCC (Concordance Correlation Coefficient)** per SemEval guidelines  

---

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
  - `train/`, `dev/`, `test/` *(not included)*

- `requirements.txt`
- `README.md`


-- 
### Set Project Root
export SEMROOT=/path/to/Semeval2026Group8

### To install all unmet dependencies
./installDependencies.sh 

### Set Project Root
python Main/main.py --model_version A --epochs 10 --batch_size 16

### Inference
python Main/predict.py --input input.jsonl --output predictions.json

--


## Model Overview
<img width="677" height="356" alt="Screenshot 2025-12-15 at 9 51 42 AM" src="https://github.com/user-attachments/assets/eb1181ba-6864-4369-8009-105d7b1c14df" />

## 📈 Evaluation

Metrics include:
- **CCC (Concordance Correlation Coefficient)**
- **MSE**

---

## 👥 Team
- **Troy Arthur**
- **Sierra Reschke**
- **Aidan Kelley**

