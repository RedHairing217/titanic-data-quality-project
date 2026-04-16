# Titanic Data Quality Project

## Goal
Explore how data quality and preprocessing affect model performance on the Titanic - Machine Learning from Disaster competition. Gain foundational understanding for effective model training and refinement processes.

## Project Focus
This project compares a baseline model against a cleaner, better-preprocessed version of the dataset to understand how data quality decisions impact results.
## Evaluation Targets
- Accuracy ≥ 0.85
- Precision ≥ 0.85
- Recall ≥ 0.75
- Log Loss ≤ 0.45
## Repository Structure
- `src/` → code for audit, preprocessing, training, evaluation
    - `tools.py`→ Shared tools imported across all scripts; consolidates cleaning, encoding, and training logic to ensure consistent behavior
    - `deep_tuner.py`→ Analyzes model performance across a range of hyperparameter and threshold configurations, outputting comparative tables to assist in tuning decisions
    - `final_model.py`→ Evaluates model performance on a local train/test split to validate tuning decisions before submission
    - `predictor.py`→ Trains on the full `train.csv` dataset and predicts survivors for `test.csv`, outputting results to `test_predictions.csv`
- `outputs/` → notes and results from data audit
- `data/` → local data folders (raw data not tracked)
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
I am using `LogisticRegression` because this is a binary classification problem (predicting survival vs. non-survival). It provides a simple and interpretable model for understanding how multiple input features influence the probability of survival.

As a baseline model, `LogisticRegression` offers a clear perspective on feature influence and allows for straightforward evaluation of how data quality and preprocessing decisions affect model performance.
## Logistic Model Results
- Accuracy: 0.8492
- Precision: 0.9038
- Recall: 0.6812
- Log Loss:  0.4303

## Logistic Model Insights
(4/13/26) The rough model is strong overall, but recall is lower than precision, meaning the model misses a meaningful number of actual survivors.

(4/13/26 5:24 PM PDT) Family size was collapsed from SibSp and ParCh, then solo travelers were isolated. This had no effect on model effectiveness, meaning that family size or whether someone was traveling alone are likely to be insignificant predictors for survival. Moving focus to age, sex, and Pclass.

(4/14/26 2:18 PM PDT) Data cleaning and segmentation has proved inconsequential for improving accuracy, precision, or recall. While results are strong, more balance between precision and recall would be preferred. Moving to `RandomForest` model to add depth and deepen tuning options

## Secondary modeling tool justification
`RandomForest` was introduced after `LogisticRegression` performance plateaued despite multiple feature engineering attempts. This suggested that the underlying relationships in the data were not fully captured by a linear model.

`RandomForest` was chosen because it can model non-linear relationships and feature interactions, leading to improved recall and overall model behavior.

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

## Threshold Tuning Results

| Threshold | Accuracy | Precision | Recall | Loss |
|-----------|----------|----------|--------|------|
| 0.35      | 0.8380   | 0.8125   | 0.7536 | 0.1620 |
| **0.375**     | **0.8436**   | **0.8254**   | **0.7536** | **0.1564** |
| 0.4       | 0.8436   | 0.8361   | 0.7391 | 0.1564 |
| 0.425     | 0.8492   | 0.8500   | 0.7391 | 0.1508 |
| 0.45      | 0.8492   | 0.8500   | 0.7391 | 0.1508 |
| **0.475**     | **0.8547**   | **0.8644**   | **0.7391** | **0.1453** |
| 0.5       | 0.8547   | 0.8772   | 0.7246 | 0.1453 |


## Random Forest Insights
**Initial Testing**\
Switching to `RandomForest` improved balance between precision and recall compared to logistic regression. However, this came at the cost of higher log loss (~+0.22), indicating less reliable probability estimates.\
The next step is to apply probability calibration to improve log loss while maintaining balanced classification performance.

**Calibration Stage** \
Implementing basic calibration resulted in a significant ~-0.23 decrease in Log Loss, bringing it down below even that of logistic regression. \
This came at the cost of ~-0.015 recall, but consequentially gained ~+0.011 accuracy and ~+0.041 precision. \
This is significant progress towards project goals, final step is to tune recall above 0.75 while maintaining 0.85 precision and accuracy.

**Tuning Stage**\
After tuning between 0.35 and 0.5 in 0.025 increments, I've found that the decision rests with either 0.475 or 0.375. \
Optimizing for finding survivors, 0.375 is the preferred option because it maximizes recall before loss spirals out of control. \
The weakness of 0.375 is the obvious increase in false positives, 0.475 is the superior option if minimizing false positives is a higher priority than identifying the most survivors possible.

## Final Model Results; Calibrated With Tuning Decision
**Threshold: 0.475**
- Accuracy:  0.8547
- Precision: 0.8644
- Recall:    0.7391
- Loss:      0.1453
- Log Loss:  0.4188

## Final Model Observations, Decision, and Justification
**Final Observations**\
Switching to `RandomForest` achieved the goal of increasing recall to acceptable levels with minimal impact to precision and accuracy.\
Implementing calibration tools even allowed for a decrease in Log Loss compared to Logistic Regression, which was an initial concern considering that ~0.65 Log Loss is unacceptable\
The calibration process also resulted in a minor decrease in Recall, with significant increases in Accuracy and Precision.\
Implementing tuning allowed me to recover the lost Recall with minimal impact to other factors.


While I was unable to find a tuning procedure that hit all three targets, I am satisfied with the final result and have learned a great deal throughout this process.

**Decision and Justification**\
The final decision of implementing a 0.375 or a 0.475 threshold lies with the models final goal (maximizing found survivors vs maximizing predictive accuracy).
For the goals of this project, I've decided to submit with 0.475 threshold to reduce false positives and maximize predictive accuracy.


- Kaggle Score: 0.76555

---

## Hyperparameter Tuning Stage

(4/15/26) With a significant divergence between kaggle score and predicted accuracy, I spent several more hours refining my tuning and feature engineering.\
Focus shifted to hyperparameter tuning via `GridSearchCV` to find the optimal `RandomForest` configuration. I conducted two rounds of grid search to find optimal parameters.

**Round 1 Results**\
`{'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'n_estimators': 200}` → CV Accuracy: 0.8287

**Round 2 Results**\
`{'max_depth': 12, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'n_estimators': 200}` → CV Accuracy: 0.8315

Constraining `max_depth` to 12 improved CV accuracy slightly, confirming that fully unconstrained trees were overfitting. `min_samples_leaf: 2` provides the primary regularization.

## Feature Engineering Stage

(4/15/26) Additional features were reintroduced to expose signal not captured by the baseline feature set:
- **Title** extracted from passenger Name field (Mr, Mrs, Miss, Master, Rare) — one of the strongest predictors on this dataset
- **FamilySize** and **IsAlone** consolidated from SibSp and Parch
- **AgeGroup** binned from continuous Age
- **Pclass_Sex** interaction term combining Pclass and Sex

## Threshold Tuning Results; Hyperparameter-Tuned Model

| Threshold | Accuracy | Precision | Recall | Loss   |
|-----------|----------|-----------|--------|--------|
| 0.25     | 0.8547   | 0.8209    | 0.7971 | 0.1453 |
| 0.275     | 0.8547   | 0.8308    | 0.7826 | 0.1453 |
| **0.3** | **0.8603** | **0.8438** | **0.7826** | **0.1397** |
| 0.325     | 0.8492   | 0.8387    | 0.7536 | 0.1508 |
| 0.35     | 0.8492   | 0.8621    | 0.7246 | 0.1508 |
| 0.375     | 0.8492   | 0.8621    | 0.7246 | 0.1508 |
| 0.4     | 0.8547   | 0.8772    | 0.7246 | 0.1453 |
| 0.425     | 0.8547   | 0.8772    | 0.7246 | 0.1453 |
| **0.45** | **0.8603** | **0.8929** | **0.7246** | **0.1397** |
| 0.475     | 0.8547   | 0.8909    | 0.7101 | 0.1453 |
| 0.5     | 0.8492   | 0.8889    | 0.6957 | 0.1508 |

## Tuned Model Insights

**Threshold Decision**\
Two thresholds tie on Accuracy and Loss: 0.3 and 0.45. The difference is the Precision/Recall trade-off, 0.3 achieves Recall of 0.7826 at the cost of Precision dropping to 0.8438, while 0.45 maximizes Precision at 0.8929 with Recall at 0.7246.\
Even though 0.3 comes closest to hitting project targets, submission at 0.3 produced a Kaggle score of 0.75358, significantly worse than 0.45's 0.77272

**Comparison to Previous Best**\
The tuned model improves accuracy by +0.0056 and precision by +0.0285 over the previous final model (threshold 0.475), at the cost of -0.0145 recall. Log Loss is reduced to 0.3948


## Hyperparameter Final Model Results
**Threshold: 0.45**
- Accuracy:  0.8603
- Precision: 0.8929
- Recall:    0.7246
- Loss:      0.1397
- Log Loss:  0.3948\
**Kaggle Score: 0.77272**

**Evaluation Targets Check**
- Accuracy: 0.8603 ✓ (target ≥ 0.85)
- Precision: 0.8929 ✓ (target ≥ 0.85)
- Recall: 0.7246 ✗ (target ≥ 0.75)
- Log Loss: 0.3948 ✓ (target ≤ 0.45)

Recall remains the one unmet target. While I was able to tune this model at 0.3 to get within 0.0062 Precision of my goal, that balanced model was inferior for competition purposes. 



## Project Retrospective

**(4/15/26)** Local metrics and Kaggle scores told different stories, and I was unable to close that gap. Local accuracy peaked at 0.8603 while my best Kaggle submission scored 0.77272.

This is likely due to the size of the data set, only 891 rows. Threshold tables based on ~178 test passengers are sensitive to which passengers land in the test fold, so cross-validation gave a more honest estimate of ~0.83.

Threshold 0.3 came within 0.01 precision of hitting all project targets locally, but scored 0.75358 on Kaggle, significantly worse than 0.45's 0.77272. While 0.3 may be more appropriate for real-world applications, like contacting shipwreck survivors, 0.45 is the correct threshold for competitive accuracy.
