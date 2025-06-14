import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import dagshub

dagshub.init(repo_owner='codedreamerD', repo_name='SMSML_Fadhilah-Nurrahmayanti', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/codedreamerD/SMSML_Fadhilah-Nurrahmayanti.mlflow")

# Create a new MLflow Experiment
mlflow.set_experiment("Experiment Student Performance")

data = pd.read_csv("dataset_preprocessed/students_performance_preprocessed.csv")

# Split fitur dan target
X = data.drop(['math score', 'reading score', 'writing score', 'average_score', 'performance_level'], axis=1)
y = data['performance_level']

# Split data train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Ambil contoh input untuk log model (harus DataFrame)
input_example = X_train.iloc[0:5]

with mlflow.start_run():
    # Set parameter model
    n_estimators = 500
    max_depth = 20

    # model Random Forest
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    # Train model
    model.fit(X_train, y_train)

    # Prediksi untuk evaluasi
    y_pred = model.predict(X_test)

    # Hitungan metrik
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Log parameter manual
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Log metrik manual
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # Simpan model dengan input_example
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )

    print("Model training dan logging selesai.")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-Score: {f1:.4f}")