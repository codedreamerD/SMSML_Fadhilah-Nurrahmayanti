import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np
import os
import joblib
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
import eli5

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from eli5.sklearn import explain_weights

# === MLflow Tracking (DagsHub)
dagshub.init(repo_owner='codedreamerD', repo_name='SMSML_Fadhilah-Nurrahmayanti', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/codedreamerD/SMSML_Fadhilah-Nurrahmayanti.mlflow")
mlflow.set_experiment("Experiment Student Performance")

# === Load data
data = pd.read_csv("dataset_preprocessed/students_performance_preprocessed.csv")
X = data.drop(['math score', 'reading score', 'writing score', 'average_score', 'performance_level'], axis=1)
y = data['performance_level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
input_example = X_train.iloc[:5]

# === Hyperparameter tuning
n_estimators_range = np.linspace(100, 1000, 5, dtype=int)
max_depth_range = np.linspace(5, 50, 5, dtype=int)

best_accuracy = 0
best_params = {}

for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:
        with mlflow.start_run(run_name=f"Tuning_n{n_estimators}_depth{max_depth}"):
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted')
            rec = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)

            # Jika model ini yang terbaik, simpan artefak
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

                # === Simpan confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.tight_layout()
                os.makedirs("model", exist_ok=True)
                plt.savefig("model/training_confusion_matrix.png")
                mlflow.log_artifact("model/training_confusion_matrix.png")

                try:
                    html_exp = eli5.format_as_html(explain_weights(model, feature_names=list(X.columns)))
                    with open("model/estimator.html", "w", encoding="utf-8") as f:
                        f.write(html_exp)
                    mlflow.log_artifact("model/estimator.html")
                except Exception as e:
                    print(f"Gagal membuat estimator.html: {e}")

print("Tuning selesai.")
print(f"Model terbaik: {best_params}, Akurasi: {best_accuracy:.4f}")