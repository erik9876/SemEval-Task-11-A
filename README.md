# SemEval-Task-11-A
This repository contains the implementation for **SemEval 2025 Task 11-A**, developed as part of the *Behind the Secrets of LLMs* module at **TU Dresden**.  The project focuses on **fine-tuning a RoBERTa-large model** for **emotion recognition in short text snippets**.

Developed by **S. Brinster, E. Gugumus, and A. Taher**.

## Contents
- `training.py` – Contains the code for training the transformer model on the English dataset (`track_a/train/eng.csv`).  
  Training was performed on the HPC of TU Dresden with CUDA-enabled GPUs, using the shell script for job submission.

- `explainability.py` – Implements SHAP-based explainability analysis for the trained model.

- `helper/` – Contains the translation script used to test whether a larger dataset would improve results.
