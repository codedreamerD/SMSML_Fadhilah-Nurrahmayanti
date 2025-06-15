import os
import time
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict
import mlflow
import mlflow.sklearn
import uvicorn
import random
from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest

# Create FastAPI app
app = FastAPI(title="Student Performance API", 
              description="Model serving API with Prometheus monitoring")

# Define Prometheus metrics
REQUESTS = Counter('student_requests_total', 'Total number of requests to the API', ['method', 'endpoint', 'status'])
PREDICTIONS = Counter('student_api_predictions_total', 'Total number of predictions made', ['class'])
PREDICTION_TIME = Histogram('student_prediction_seconds', 'Time spent processing prediction request')
MODEL_CONFIDENCE = Histogram('student_model_confidence', 'Confidence scores for predictions', ['class'])
REQUEST_LATENCY = Summary('student_request_latency_seconds', 'Request latency in seconds', ['endpoint'])
FEATURE_GAUGE = Gauge('student_feature_value', 'Feature values from requests', ['feature'])
PREDICTION_ERRORS = Counter('student_prediction_errors', 'Prediction errors', ['error_type'])
SYSTEM_MEMORY = Gauge('student_system_memory_bytes', 'System memory usage')
SYSTEM_CPU = Gauge('student_system_cpu_percent', 'System CPU usage percent')
MODEL_LOAD_TIME = Gauge('student_model_load_time_seconds', 'Time to load model')

# Track active requests
ACTIVE_REQUESTS = Gauge('student_active_requests', 'Number of currently active requests')

# Input data model
class PerformanceLevel(BaseModel):
    performance_level: str

# Output prediction model
class StudentPerformance(BaseModel):
    prediction: int
    class_name: str
    probability: Dict[str, float]
    processing_time: float

# Load the model
def load_model():
    start_time = time.time()
    try:
        # Try loading from MLflow
        try:
            model_uri = "models/best_model.pkl"
            model = mlflow.sklearn.load_model(model_uri)
        except Exception as e:
            # Fallback to local model file
            model_path = os.path.join(os.path.dirname(__file__), "models", "best_model.pkl")
            model = joblib.load(model_path)
        
        MODEL_LOAD_TIME.set(time.time() - start_time)
        return model
    except Exception as e:
        PREDICTION_ERRORS.labels(error_type="model_load_error").inc()
        print(f"Error loading model: {e}")
        # For demo purposes, return a dummy model
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier()

# Load the model
model = load_model()

# Class names mapping
class_names = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

# Middleware to track request latency
@app.middleware("http")
async def add_metrics_middleware(request: Request, call_next):
    ACTIVE_REQUESTS.inc()
    start_time = time.time()
    try:
        response = await call_next(request)
        REQUEST_LATENCY.labels(endpoint=request.url.path).observe(time.time() - start_time)
        REQUESTS.labels(method=request.method, endpoint=request.url.path, status=response.status_code).inc()
        return response
    except Exception as e:
        PREDICTION_ERRORS.labels(error_type="request_processing_error").inc()
        raise e
    finally:
        ACTIVE_REQUESTS.dec()

# Endpoint to expose metrics to Prometheus
@app.get("/metrics")
async def metrics():
    # Simulate system metrics for demo
    SYSTEM_MEMORY.set(random.randint(1000000, 2000000))
    SYSTEM_CPU.set(random.uniform(10, 90))
    
    return generate_latest()

# Root endpoint with updated HTML and styling
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Student Performance API</title>
            <style>
                body {
                    font-family: 'Arial', sans-serif;
                    margin: 0;
                    padding: 30px;
                    background-color: #e8f0f2;
                    color: #333;
                }
                .container {
                    max-width: 900px;
                    margin: 0 auto;
                    background-color: #ffffff;
                    padding: 25px;
                    border-radius: 8px;
                    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
                    border-top: 5px solid #3b8d99;
                }
                h1 {
                    font-size: 2.5em;
                    color: #3b8d99;
                    text-align: center;
                }
                p {
                    font-size: 1.1em;
                    line-height: 1.6;
                    text-align: center;
                    color: #555;
                }
                a {
                    color: #1e73b3;
                    text-decoration: none;
                    font-weight: bold;
                }
                a:hover {
                    color: #3b8d99;
                    text-decoration: underline;
                }
                .endpoint {
                    margin-top: 30px;
                    padding: 20px;
                    background-color: #f1f8ff;
                    border-left: 5px solid #3b8d99;
                    border-radius: 5px;
                }
                h3 {
                    font-size: 1.5em;
                    color: #3b8d99;
                    margin-bottom: 15px;
                }
                pre {
                    background-color: #f4f4f4;
                    padding: 10px;
                    border-radius: 5px;
                    font-size: 1.1em;
                    margin-top: 15px;
                }
                code {
                    background-color: #f4f4f4;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-size: 1em;
                    font-family: 'Courier New', Courier, monospace;
                }
                .footer {
                    margin-top: 50px;
                    text-align: center;
                    color: #777;
                    font-size: 0.9em;
                }
                .footer a {
                    font-weight: bold;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Welcome to the Student Performance API</h1>
                <p>This API predicts student performance levels based on input features. Use it to assess performance predictions in real-time!</p>
                
                <div class="endpoint">
                    <h3>Predict Endpoint</h3>
                    <p>To make predictions, send POST requests to <code>/predict</code> with the following JSON data format:</p>
                    <pre><code>{"performance_level": "A"}</code></pre>
                    <p>This will return a prediction of the student's performance level.</p>
                </div>
                
                <div class="endpoint">
                    <h3>Metrics Endpoint</h3>
                    <p>Monitor the health and status of the API using Prometheus metrics by accessing <a href="/metrics">/metrics</a>.</p>
                </div>
                
                <div class="endpoint">
                    <h3>Health Check</h3>
                    <p>Check the overall health of the API by visiting <a href="/health">/health</a>.</p>
                </div>
            </div>
        </body>
    </html>
    """


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# Prediction endpoint
@app.post("/predict", response_model=StudentPerformance)
def predict(features: PerformanceLevel):
    # Start timing
    with PREDICTION_TIME.time():
        start_time = time.time()
        
        try:
            # Log feature values
            feature_array = np.array([[
                features.performance_level
            ]])
            
            # Update feature gauges
            FEATURE_GAUGE.labels(feature="performance_level").set(features.performance_level)

            
            # Make prediction
            prediction = int(model.predict(feature_array)[0])
            PREDICTIONS.labels(class_name=class_names[prediction]).inc()
            
            # Get probabilities if available
            probabilities = {}
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(feature_array)[0]
                for i, p in enumerate(proba):
                    class_name = class_names[i]
                    probabilities[class_name] = float(p)
                    MODEL_CONFIDENCE.labels(class_name=class_name).observe(p)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Return prediction
            return {
                "prediction": prediction,
                "class_name": class_names[prediction],
                "probability": probabilities,
                "processing_time": processing_time
            }
            
        except Exception as e:
            PREDICTION_ERRORS.labels(error_type="prediction_error").inc()
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)