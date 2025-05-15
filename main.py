
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerationRequest(BaseModel):
    prompt: str
    output_format: str = "webp"

@app.post("/generate")
async def generate(req: GenerationRequest):
    headers = {
        "Authorization": f"Bearer {os.getenv('STABILITY_API_KEY', '')}",
        "Accept": "image/*"
    }

    try:
        response = requests.post(
            "https://api.stability.ai/v2beta/stable-image/generate/ultra",
            headers=headers,
            files={"none": ""},
            data={
                "prompt": req.prompt,
                "output_format": req.output_format
            }
        )

        if response.status_code == 200:
            img_base64 = base64.b64encode(response.content).decode("utf-8")
            return { "image_base64": img_base64 }

        return {
            "error": f"Generation failed",
            "status_code": response.status_code,
            "body": response.json()
        }

    except Exception as e:
        return { "error": str(e) }
