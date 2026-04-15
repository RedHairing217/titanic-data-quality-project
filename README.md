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

## Initial modeling tool justification
I am using logistic regression because this is a binary classification problem (predicting survival vs. non-survival). It provides a simple and interpretable model for understanding how multiple input features influence the probability of survival.

As a baseline model, logistic regression offers a clear perspective on feature influence and allows for straightforward evaluation of how data quality and preprocessing decisions affect model performance.
## Logistic Model Results
- Accuracy: 0.8492
- Precision: 0.9038
- Recall: 0.6812
- Log Loss:  0.4303

## Logistic Model Insights
The rough model is strong overall, but recall is lower than precision, meaning the model misses a meaningful number of actual survivors.\
(4/13/26 5:24 PM PDT) Family size was collapsed from SibSp and ParCh, then solo travelers were isolated. This had no on affect model effectiveness, meaning that family size or whether someone was traveling alone are likely to be insignificant predictors for survival. Moving focus to age, sex, and Pclass.\
(4/14/26 2:18 PM PDT) Data cleaning and segmentation has proved inconsequential for improving accuracy, precision, or recall. While results are strong, more balance between precision and recall would be preferred. Moving to random forest modeling to add depth and deepen tuning options

## Secondary modeling tool justificaiton
Random Forest was introduced after logistic regression performance plateaued despite multiple feature engineering attempts. This suggested that the underlying relationships in the data were not fully captured by a linear model.

Random Forest was chosen because it can model non-linear relationships and feature interactions, leading to improved recall and overall model behavior.

## Random Forest Results
Uncalibrated
- Accuracy:  0.8436
- Precision: 0.8361
- Recall:    0.7391
- Log Loss:  0.6505

Calibrated
- Accuracy:  0.8547
- Precision: 0.8772
- Recall:    0.7246
- Log Loss:  0.4188
## Random Forest Insights
(4/15/2026 4:00 PM) Switching to Random Forest improved balance between precision and recall compared to logistic regression. However, this came at the cost of higher log loss (~+0.22), indicating less reliable probability estimates.
The next step is to apply probability calibration to improve log loss while maintaining balanced classification performance.\
(4/15/2026 4:15) Implementing basic calibration resulted in a significant ~-0.23 decrease in Log Loss, bringing it down below even that of logistic regression. This came at the cost of ~-0.015 recall, but consequentially gained ~+0.011 accuracy and ~+0.041 precision. This is significant progress towards project goals, final step is to tune recall above 0.75 while maintaining 0.85 precision and accuracy.

## Repository Structure
- `src/` → code for audit, preprocessing, training, evaluation
- `outputs/` → notes and results
- `data/` → local data folders (raw data not tracked)

## Next Steps
- Cleaned model goal: recall >= 0.75, precision >= 0.85
- compare baseline vs improved preprocessing
- run experiments on feature choices and model behavior