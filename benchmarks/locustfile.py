import os
import time

from locust import FastHttpUser, task

TEXT = "A cat running away from a mouse"
SERVER_URL = os.getenv("SERVER_URL", "http://127.0.0.1:50362")


class User(FastHttpUser):
    host = SERVER_URL

    @task
    def req(self):
        s = time.time()
        self.client.post("/predict", json={"text": TEXT})
        print(time.time() - s)
