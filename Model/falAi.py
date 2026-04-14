import os
from huggingface_hub import InferenceClient
from SafeExecution.safeExecution import safeExecution

class FalAi:
    def __init__(self, client):
        self.client = client

    @safeExecution
    def generate_image(self, prompt):
        image = self.client.text_to_image(
            prompt, model="stabilityai/stable-diffusion-3.5-large"
        )
        return image
