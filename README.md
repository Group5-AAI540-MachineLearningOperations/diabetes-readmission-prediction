# Diabetes Readmission Prediction

AAI-540 Group 5 project. We use the Diabetes 130-US Hospitals data to predict whether a patient will be readmitted within 30 days. The target is `readmitted_30d`: `1` means readmitted in less than 30 days, and `0` means otherwise. This is a class project and its predictions are not for clinical decisions.

## Project workflow

1. Load the hospital encounter data and ID mappings. Clean missing values, group diagnosis codes, and prepare the features.
2. Split patients into training (70%), validation (15%), and test (15%) sets. All encounters for one patient stay in the same set.
3. Train a logistic regression baseline and three XGBoost candidates. Choose the XGBoost model by validation PR-AUC, then choose a prediction threshold on validation data with a target of at least 70% recall.
4. Evaluate once on the test set. Check recall, precision, PR-AUC, ROC-AUC, the confusion matrix, and results by race, gender, and age.
5. Save the model and evaluation files. The training notebook registers a model package in SageMaker with `PendingManualApproval` status.
6. Run the local quality gate. After a model is approved in SageMaker, the deployment notebook can create a real-time endpoint. Separate notebooks cover data capture and monitoring.

## Repository files

| Path | Purpose |
| --- | --- |
| [`data/`](data/) | Hospital encounter data and `IDS_mapping.csv` lookup file. |
| [`notebooks/01_data_ingestion_eda.ipynb`](notebooks/01_data_ingestion_eda.ipynb) | Initial S3 ingestion, AWS Glue cataloging, and data exploration. |
| [`notebooks/00-models.ipynb`](notebooks/00-models.ipynb) | Majority-class benchmark. |
| [`notebooks/training/train-model.ipynb`](notebooks/training/train-model.ipynb) | Data preparation, model training, test evaluation, fairness tables, and SageMaker registration. |
| [`notebooks/training/artifacts/`](notebooks/training/artifacts/) | Saved processed data, model, preprocessor, and evaluation results from the training run. |
| [`notebooks/evaluation/automated-model-evaluation.ipynb`](notebooks/evaluation/automated-model-evaluation.ipynb) | Local model evaluation and pass/fail quality gates. |
| [`notebooks/evaluation/results/`](notebooks/evaluation/results/) | Saved quality gate report and comparison tables. |
| [`notebooks/infrastructure/deploy-model.ipynb`](notebooks/infrastructure/deploy-model.ipynb) | Deploy the latest approved model package to a SageMaker real-time endpoint. |
| [`notebooks/setup/`](notebooks/setup/) | SageMaker Feature Store setup and endpoint data drift checks. |
| [`notebooks/monitor/cloudwatch-monitoring.ipynb`](notebooks/monitor/cloudwatch-monitoring.ipynb) | Drift, outcome metrics, CloudWatch alarms, and dashboard workflow. |

## Run the notebooks

Use Python 3 and Jupyter. The local evaluation notebook needs `numpy`, `pandas`, `scikit-learn`, `xgboost`, `joblib`, and `matplotlib`. Its first code cell includes an optional install command. The saved preprocessor was created with **scikit-learn 1.4.2**, so use that version when loading it. The AWS notebooks also need AWS credentials, permissions for the services they call, `boto3`, and SageMaker SDK 2.x. They are intended for a SageMaker environment.

For a local check with the included artifacts, open [`notebooks/evaluation/automated-model-evaluation.ipynb`](notebooks/evaluation/automated-model-evaluation.ipynb) and run all cells. It finds the project folder, recreates the patient split, loads the saved model, calculates test metrics, and writes `quality_gate_report.json`, `quality_gates.csv`, and `model_comparison.csv` under `notebooks/evaluation/results/`. It stops with an error if a data check or quality gate fails.

To run this check from a terminal at the project root, install Jupyter and the packages above, then use:

```bash
jupyter nbconvert --to notebook --execute notebooks/evaluation/automated-model-evaluation.ipynb --output automated-model-evaluation-executed.ipynb --ExecutePreprocessor.timeout=600
```

For the full AWS workflow, start with the ingestion and Feature Store notebooks if those resources are needed in your account. Run the benchmark and training notebooks next. Review the saved test results and quality gate report before approving a model package. Then run the deployment and monitoring notebooks. Deployment creates a billable SageMaker endpoint; the monitoring setup notebook includes a cleanup section.

## Saved model results

The saved XGBoost model uses 500 trees, a maximum depth of 6, and a learning rate of 0.03. Its decision threshold is about 0.435. The included test report shows:

| Metric | Test result |
| --- | ---: |
| Recall | 0.7251 |
| Precision | 0.1614 |
| PR-AUC | 0.2287 |
| ROC-AUC | 0.6836 |
| F1 | 0.2641 |

The included quality gate report shows **PASS** for the example project thresholds: recall at least 0.70, precision at least 0.15, PR-AUC at least 0.20, ROC-AUC at least 0.65, and recall and PR-AUC above the majority-class benchmark. Precision is still low, so many positive predictions are false alarms. These thresholds are class-project checks, not clinical standards.

The deployment notebook uses one `ml.m5.large` instance for real-time predictions. Its model package must already have `Approved` status in SageMaker. The training notebook creates a package with `PendingManualApproval` status; it does not approve it automatically.

## Proposed improvements

The [system diagram](docs/images/ml-system-feature-enhancements.png) marks possible future work in blue. This includes newer hospital data, stronger data quality checks, more model tuning, expanded fairness checks, and monitoring with retraining when real outcomes become available. These are proposed improvements rather than completed features.
