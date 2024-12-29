# Use the official Python image as the base
FROM python:3.11-slim

# Setting the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of application code
COPY . .

# command to run  script
CMD ["python", "onnx-detect.py", "--weights", "ShipDetectionClassifier.onnx", "--img", "640", "--conf", "0.25", "--source", "000019.bmp", "--exist-ok"]
