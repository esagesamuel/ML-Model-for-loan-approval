# Loan Approval Model Report

## 1. Business Understanding

FinTech Innovations wants to help partner banks modernize loan approvals. The current manual process can be slow and inconsistent, so the business goal is to provide a repeatable decision-support model that helps standardize applicant screening.

Two modeling options were available:

- Regression: predict `RiskScore` so loan officers can use a numeric risk estimate.
- Classification: predict `LoanApproved` so the system can directly support approval screening.

This submission uses the classification approach as the primary model because it maps directly to the operational workflow: approve or deny a loan application. Since lending decisions can create fairness, compliance, and customer-impact risks, the model is framed as a decision-support tool rather than a fully autonomous final decision system.

## 2. Data Understanding

The dataset contains 20,000 historical loan applications and 35 columns. It includes demographic/application fields, credit indicators, financial balances, loan terms, historical payment behavior, approval outcomes, and a risk score.

The target variable is `LoanApproved`:

- Denied applications: 76.1%
- Approved applications: 23.9%

This class imbalance matters because a model could appear accurate by mostly predicting denials. For that reason, evaluation emphasizes approval-class precision, recall, F1, ROC-AUC, and the confusion matrix rather than accuracy alone.

Missing values were found in:

- `MaritalStatus`: 1,331 missing
- `EducationLevel`: 901 missing
- `SavingsAccountBalance`: 572 missing

## 3. Data Preparation

`AnnualIncome` was stored as currency text, so dollar signs and commas were removed and the column was converted to numeric.

The following columns were removed before modeling to reduce target leakage:

- `LoanApproved`: target variable
- `RiskScore`: alternate target
- `BaseInterestRate`: likely underwriting output
- `InterestRate`: likely underwriting output
- `MonthlyLoanPayment`: depends on loan pricing/terms finalized during approval

Numeric features were imputed with median values. Categorical features were imputed with the most frequent value and one-hot encoded. For logistic regression, numeric features were standardized.

The data was split into an 80% training set and a 20% test set using stratification so the approval/denial balance stayed consistent in both sets.

## 4. Modeling

Three classification models were compared:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.916 | 0.765 | 0.932 | 0.841 | 0.979 |
| Random Forest | 0.910 | 0.789 | 0.853 | 0.820 | 0.966 |
| Gradient Boosting | 0.918 | 0.779 | 0.917 | 0.842 | 0.976 |

The selected model is balanced logistic regression. It had the highest ROC-AUC and the strongest approval recall, while remaining more transparent than the ensemble models. That interpretability is useful in a regulated financial-services setting where stakeholders need to understand and challenge model behavior.

## 5. Evaluation

The selected model produced the following confusion matrix on the test set:

| | Predicted Denied | Predicted Approved |
|---|---:|---:|
| Actual Denied | 2,771 | 273 |
| Actual Approved | 65 | 891 |

The model correctly identified 891 of 956 approved applications in the test set. It missed 65 approved applications and incorrectly marked 273 denied applications as approved.

For a bank, the right operating threshold should depend on risk appetite. A stricter threshold can reduce false approvals, while a more inclusive threshold can reduce missed creditworthy applicants. The current default threshold favors high recall for approvals, which is useful when the model is used for screening or recommendation rather than automatic final approval.

Permutation importance showed the strongest model drivers:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `TotalDebtToIncomeRatio` | 0.185 |
| 2 | `MonthlyIncome` | 0.081 |
| 3 | `LoanDuration` | 0.024 |
| 4 | `LoanAmount` | 0.023 |
| 5 | `NetWorth` | 0.022 |
| 6 | `LengthOfCreditHistory` | 0.011 |
| 7 | `EducationLevel` | 0.008 |
| 8 | `BankruptcyHistory` | 0.008 |
| 9 | `PreviousLoanDefaults` | 0.005 |
| 10 | `EmploymentStatus` | 0.002 |

These drivers are broadly consistent with lending intuition: debt burden, income, loan size, loan duration, and borrower balance-sheet strength are highly relevant to approval outcomes.

## 6. Deployment Recommendations

The model should be deployed as a decision-support service that returns:

- Approval probability
- Recommended decision band, such as low, medium, or high review priority
- Top contributing features for the decision

Recommended governance steps before production:

- Validate fairness across protected or proxy demographic groups where legally and ethically appropriate.
- Calibrate the probability threshold with the bank's risk appetite and portfolio strategy.
- Review false approvals and false denials with loan officers.
- Monitor approval rates, default outcomes, and feature drift over time.
- Retrain on a scheduled basis once enough new labeled performance data is available.

## Conclusion

The balanced logistic regression model provides a strong, interpretable starting point for modernizing loan approvals. It achieved a ROC-AUC of 0.979 and approval recall of 0.932 on the held-out test set. The model should not replace human oversight immediately, but it can standardize screening, reduce review time, and help loan officers focus attention on applications that most need judgment.
