# Loan Approval Risk Analytics Project

This project builds a machine learning pipeline for FinTech Innovations to support a more consistent, data-driven loan approval process.

The main deliverable is a classification model that predicts `LoanApproved` from applicant and application information available at decision time. The final selected model is a balanced logistic regression because it performs strongly and remains easier to explain to loan officers and compliance stakeholders than more opaque alternatives.

## Project Structure

- `src/train_loan_approval_model.py` trains, evaluates, and saves the model pipeline.
- `reports/loan_approval_crisp_dm_report.md` documents the work using the CRISP-DM framework.
- `outputs/metrics.json` stores evaluation results after training.
- `outputs/loan_approval_model.joblib` stores the fitted scikit-learn pipeline after training.
- `outputs/feature_importance.csv` stores permutation importances for interpretation.
- `outputs/*.png` contains evaluation plots.

## Reproduce the Model

```powershell
python .\src\train_loan_approval_model.py
```

The script defaults to:

```text
C:\Users\Dell_User\Documents\flatiron\module-4\First Summative Lab\financial_loan_data.csv
```

To train on a different file:

```powershell
python .\src\train_loan_approval_model.py --data-path "path\to\financial_loan_data.csv"
```

## Current Results

Using an 80/20 stratified train-test split, the selected balanced logistic regression achieved:

- Accuracy: `0.916`
- Precision for approvals: `0.765`
- Recall for approvals: `0.932`
- F1 for approvals: `0.841`
- ROC-AUC: `0.979`

The strongest model drivers were `TotalDebtToIncomeRatio`, `MonthlyIncome`, `LoanDuration`, `LoanAmount`, and `NetWorth`.
