from prometheus_client import Counter, Gauge, Summary, Histogram, start_http_server, generate_latest, CONTENT_TYPE_LATEST
import time
import random
import http.server
import threading

# Create metrics
REQUEST_COUNT = Counter('student_api_requests_total', 'Total number of requests received')
REQUEST_LATENCY = Summary('student_api_request_latency_seconds', 'Request latency in seconds')
PREDICTION_COUNT = Counter('student_model_predictions_total', 'Total number of predictions made', ['class_name'])
MODEL_CONFIDENCE = Histogram('student_model_confidence', 'Confidence scores of model predictions', ['class_name'])
FEATURE_GAUGE = Gauge('student_feature_values', 'Feature values used for prediction', ['feature'])
SYSTEM_MEMORY = Gauge('system_memory_usage_percent', 'System memory usage percentage')
SYSTEM_CPU = Gauge('system_cpu_usage_percent', 'System CPU usage percentage')

# Class names
class_names = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

# Simulated data for Student Performance Level (this can be used as an input for the model)
student_performance_data = {
    "A": 0.2,
    "B": 0.5,
    "C": 0.1,
    "D": 0.1,
    "E": 0.1
}

# Simulate metrics for demo purposes
def simulate_metrics():
    while True:
        # Simulate requests
        REQUEST_COUNT.inc()
        
        # Simulate latency
        with REQUEST_LATENCY.time():
            # Simulate some processing time
            time.sleep(random.uniform(0.01, 0.1))
        
        # Simulate predictions based on performance level
        performance_level = random.choice(list(student_performance_data.keys()))  # Random performance level
        prediction = class_names.get(performance_level)
        PREDICTION_COUNT.labels(class_name=prediction).inc()
        
        # Simulate confidence scores
        for i in range(5):
            conf = random.random() if list(class_names.values())[i] == prediction else random.random() * 0.5
            MODEL_CONFIDENCE.labels(class_name=list(class_names.values())[i]).observe(conf)
        
        # Simulate feature values based on performance level
        FEATURE_GAUGE.labels(feature=performance_level).set(student_performance_data[performance_level])
        
        # Simulate system metrics
        SYSTEM_MEMORY.set(random.uniform(40, 90))
        SYSTEM_CPU.set(random.uniform(10, 80))
        
        time.sleep(1)

# Metrics HTTP handler
class MetricsHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', CONTENT_TYPE_LATEST)
        self.end_headers()
        self.wfile.write(generate_latest())
    
    def log_message(self, format, *args):
        # Suppress log messages
        return

# Start the exporter
if __name__ == '__main__':
    # Start metrics server
    start_http_server(5000)
    print("Prometheus metrics server started on port 5000")
    
    # Start metrics simulation in a background thread
    sim_thread = threading.Thread(target=simulate_metrics, daemon=True)
    sim_thread.start()
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
