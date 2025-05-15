
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
        "model": "stable-diffusion-xl-v1",
        "prompt": req.prompt,
        "steps": req.steps,
        "cfg_scale": req.cfg_scale,
        "width": req.width,
        "height": req.height,
        "samples": 1,
        "output_format": "base64"
    }

    if req.negative_prompt:
        payload["negative_prompt"] = req.negative_prompt
    if req.seed is not None:
        payload["seed"] = req.seed

    log_block = "\n========== Stability API v2beta Request =========="
    log_block += "\n📤 Payload:\n" + json.dumps(payload, indent=2)

    try:
        response = requests.post(
            "https://api.stability.ai/v2beta/image/generate",
            headers=headers,
            json=payload
        )
        log_block += f"\n📥 Status: {response.status_code}"
        log_block += f"\n📥 Response Body:\n{response.text}"
        response.raise_for_status()
        print(log_block)
        return response.json()

    except requests.exceptions.HTTPError as http_err:
        log_block += "\n❌ HTTP Error:\n" + str(http_err)
        log_block += "\n📥 Response Body:\n" + response.text
        print(log_block)
        return {
            "error": str(http_err),
            "status_code": response.status_code,
            "body": response.text
        }

    except Exception as e:
        log_block += "\n❌ General Error:\n" + str(e)
        print(log_block)
        return {"error": str(e)}
