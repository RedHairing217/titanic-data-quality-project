# Titanic Data Quality Project

## Goal
Explore how data quality and preprocessing affect model performance on the Titanic - Machine Learning from Disaster competition. Gain foundational understanding for effective model training and refinement processes.

## Project Focus
This project compares a baseline model against a cleaner, better-preprocessed version of the dataset to understand how data quality decisions impact results.

## Workflow
1. Data audit
2. Baseline preprocessing
3. Baseline model
4. Improved preprocessing
5. Model comparison
6. Experiments and evaluation

## Dataset
Titanic competition dataset from Kaggle.
https://www.kaggle.com/competitions/titanic/overview

## Rough Model Results
- Accuracy: 0.8492
- Precision: 0.9038
- Recall: 0.6812

## Key Insight
The baseline model is strong overall, but recall is lower than precision, meaning the model misses a meaningful number of actual survivors.

## Repository Structure
- `src/` → code for audit, preprocessing, training, evaluation
- `outputs/` → notes and results
- `data/` → local data folders (raw data not tracked)

## Next Steps
- build cleaned model
- Cleaned model goal: recall >= 0.75, precision >= 0.85
- compare baseline vs improved preprocessing
- run experiments on feature choices and model behavior