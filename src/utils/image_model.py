# -*- coding: utf-8 -*-
import base64
import os
import requests
from dotenv import load_dotenv
from utils.config import (
    ENDPOINT, IMAGE_MODEL, NEGATIVE_PROMPT, 
    HEIGHT, WIDTH, NUM_IMAGES, SEED, STEPS
)
# Credentials
load_dotenv()
TOGETHER_API_KEY = os.environ["TOGETHER_API_KEY"]


def generate_image(image_prompt: str, msg_id: str):
    """
    Generates and saves image(s) based on a text prompt
    """
    res = requests.post(
        ENDPOINT, 
        json={
        "model": IMAGE_MODEL,
        "prompt": image_prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "width": WIDTH,
        "height": HEIGHT,
        "steps": STEPS,
        "n": NUM_IMAGES,
        "seed": SEED
        }, 
        headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}"}
    )
    # Convert response to JSON
    res = res.json()
    # Get image from response
    images = res["choices"]
    # Create image directory (if not exists)
    img_dir = os.path.join(os.getcwd(), "..", "images")
    os.makedirs(img_dir, exist_ok=True)
    # Set file name and path
    img_file = f"{msg_id}.png"
    img_path = os.path.join(img_dir, img_file)
    # Save image in temporary directory
    for b64_img in images:
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(b64_img["image_base64"]))
    
    return img_path
            
            