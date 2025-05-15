
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os

app = FastAPI()

# CORS for frontend on Netlify or localhost
dev_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[dev_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = None
    steps: int = 30
    cfg_scale: float = 7.0
    seed: int = None
    width: int = 512
    height: int = 512

@app.post("/generate")
async def generate_image(req: GenerateRequest):
    api_key = os.getenv("STABILITY_API_KEY")
    if not api_key:
        return {"error": "API key missing"}

    payload = {
        "model": "stable-diffusion-xl-beta",
        "prompt": req.prompt,
        "negative_prompt": req.negative_prompt,
        "steps": req.steps,
        "cfg_scale": req.cfg_scale,
        "seed": req.seed,
        "width": req.width,
        "height": req.height,
        "output_format": "base64"
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.stability.ai/v2beta/stable-image/generate",
            json=payload,
            headers=headers
        )

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"{response.status_code} {response.text}"}

@app.get("/")
def root():
    return {"message": "Ghost AI Backend Ready"}
