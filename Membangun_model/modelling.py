import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Tracking ke lokal (localhost)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Experiment Student Performance")

# Load data
data = pd.read_csv("dataset_preprocessed/students_performance_preprocessed.csv")

# Fitur dan target
X = data.drop(['math score', 'reading score', 'writing score', 'average_score', 'performance_level'], axis=1)
y = data['performance_level']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Ambil contoh input untuk log model
input_example = X_train.iloc[:5]

# Mulai MLflow run
with mlflow.start_run():
    mlflow.autolog()  # autolog otomatis log param, metric, dan model

    model = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=42)
    model.fit(X_train, y_train)

    # Simpan model ke MLflow
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )

print("Training & Logging selesai.")