import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import dagshub
import os

# === Setup MLflow DagsHub Tracking ===
dagshub.init(repo_owner='codedreamerD', repo_name='SMSML_Fadhilah-Nurrahmayanti', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/codedreamerD/SMSML_Fadhilah-Nurrahmayanti.mlflow")
mlflow.set_experiment("Experiment Student Performance")

# === Load Data ===
data = pd.read_csv("dataset_preprocessed/students_performance_preprocessed.csv")
X = data.drop(['math score', 'reading score', 'writing score', 'average_score', 'performance_level'], axis=1)
y = data['performance_level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# === Input example for logging model
input_example = X_train.iloc[:5]

# === Hyperparameter Ranges
n_estimators_range = np.linspace(100, 1000, 5, dtype=int)
max_depth_range = np.linspace(5, 50, 5, dtype=int)

best_accuracy = 0
best_params = {}

# === Manual Tuning Loop
for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:
        with mlflow.start_run(run_name=f"Tuning_n{n_estimators}_depth{max_depth}"):
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # === Manual Logging
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted")
            rec = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")

            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)

            # Simpan model terbaik
            if acc > best_accuracy:
                best_accuracy = acc
                best_params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth
                }

                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="best_model",
                    input_example=input_example
                )

print("Tuning selesai.")
print(f"Model terbaik: {best_params}, Akurasi: {best_accuracy:.4f}")