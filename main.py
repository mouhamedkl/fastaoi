from fastapi import FastAPI, Form
from fastapi.responses import FileResponse
import torch
from diffusers import StableDiffusionPipeline

app = FastAPI(title="Stable Diffusion API")

# Détecter GPU si disponible
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Device utilisé: {device}")

# Charger le modèle
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype,
    safety_checker=None
).to(device)

# Optimisations pour VRAM / CPU faible
try:
    pipe.enable_xformers_memory_efficient_attention()
except:
    pass
pipe.enable_attention_slicing()

@app.post("/generate")
async def generate(prompt: str = Form(...)):
    """Génère une image à partir d'un prompt texte"""
    try:
        with torch.inference_mode():
            image = pipe(
                prompt,
                guidance_scale=7.5,
                num_inference_steps=15,   # CPU friendly
                height=512,
                width=512
            ).images[0]

        filename = "generated.png"
        image.save(filename)
        return FileResponse(filename)

    except Exception as e:
        return {"error": str(e)}
