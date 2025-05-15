
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
import json

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
    negative_prompt: str | None = None
    width: int = 512
    height: int = 512
    steps: int = 30
    cfg_scale: float = 7.0
    seed: int | None = None

@app.post("/generate")
async def generate(req: GenerationRequest):
    headers = {
        "Authorization": f"Bearer {os.getenv('STABILITY_API_KEY', '')}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    payload = {
        "model": "stable-diffusion-v1-5",
        "prompt": req.prompt,
        "negative_prompt": req.negative_prompt,
        "steps": req.steps,
        "cfg_scale": req.cfg_scale,
        "seed": req.seed,
        "width": req.width,
        "height": req.height,
        "samples": 1,
        "output_format": "base64"
    }

    print("\n📤 Sending payload to Stability AI:")
    print(json.dumps(payload, indent=2))

    try:
        response = requests.post(
            "https://api.stability.ai/v1/generation/stable-diffusion-v1-5/text-to-image",
            headers=headers,
            json=payload
        )

        print("\n📥 Response status:", response.status_code)
        print("📥 Response body:")
        print(response.text)

        response.raise_for_status()
        return response.json()

    except requests.exceptions.HTTPError as http_err:
        print("\n❌ HTTP error occurred:", http_err)
        return {
            "error": str(http_err),
            "status_code": response.status_code,
            "body": response.text
        }

    except Exception as e:
        print("\n❌ General error occurred:", str(e))
        return {"error": str(e)}
