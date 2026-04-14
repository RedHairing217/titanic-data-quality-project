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

## Modeling tool justification
I am using logistic regression because this is a binary classification problem (predicting survival vs. non-survival). It provides a simple and interpretable model for understanding how multiple input features influence the probability of survival.

As a baseline model, logistic regression offers a clear perspective on feature influence and allows for straightforward evaluation of how data quality and preprocessing decisions affect model performance.
## Rough Model Results
- Accuracy: 0.8492
- Precision: 0.9038
- Recall: 0.6812

## Key Insight
The rough model is strong overall, but recall is lower than precision, meaning the model misses a meaningful number of actual survivors.

## Cleaned model conclusions
 (4/13/26 5:24 PM PDT) Family size was collapsed from SibSp and ParCh, then solo travelers were isolated. This had no on affect model effectiveness, meaning that family size or whether someone was traveling alone are likely to be insignificant predictors for survival. Moving focus to age, sex, and Pclass.
## Repository Structure
- `src/` → code for audit, preprocessing, training, evaluation
- `outputs/` → notes and results
- `data/` → local data folders (raw data not tracked)

## Next Steps
- Cleaned model goal: recall >= 0.75, precision >= 0.85
- compare baseline vs improved preprocessing
- run experiments on feature choices and model behavior