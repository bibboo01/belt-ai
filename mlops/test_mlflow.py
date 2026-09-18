import mlflow
import time

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Default")

with mlflow.start_run(run_name="first-test"):

    mlflow.log_param("model", "YOLO11")
    mlflow.log_param("image_size", 1280)
    mlflow.log_param("epochs", 10)
    mlflow.log_param("batch", 8)

    mlflow.log_metric("precision", 0.8905)
    mlflow.log_metric("recall", 0.9272)
    mlflow.log_metric("mAP50", 0.9518)
    mlflow.log_metric("mAP50_95", 0.7669)

    mlflow.set_tag("project", "belt-ai")
    mlflow.set_tag("model_type", "object_detection")

    time.sleep(1)

print("MLflow test completed.")