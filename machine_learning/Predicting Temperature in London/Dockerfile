FROM python:3.9-slim
# RUN apt-get update && apt-get install -y python3 python3-pip
RUN mkdir /app
WORKDIR /app

COPY linear_regression_model_v1 /app/linear_regression_model_v1
COPY src/inference.py inference.py

# Install required Python packages
RUN pip install mlflow scikit-learn

CMD ["python", "inference.py"]