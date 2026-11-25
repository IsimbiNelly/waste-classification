# locustfile.py
# Simulates many visitors sending an image to the /predict endpoint.

import os
import random

from locust import HttpUser, task, between

# Use real test images from your project
IMAGE_DIR = os.path.join("data", "test", "plastic")
IMAGE_FILES = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)][:10]


class WasteUser(HttpUser):
    wait_time = between(1, 2)   # each simulated visitor waits 1 to 2 seconds between requests

    @task(5)
    def predict(self):
        path = random.choice(IMAGE_FILES)
        with open(path, "rb") as f:
            self.client.post("/predict", files={"file": (os.path.basename(path), f, "image/jpeg")})

    @task(1)
    def health(self):
        self.client.get("/health")