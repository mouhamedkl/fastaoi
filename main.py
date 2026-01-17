from fastapi import FastAPI, Form
from fastapi.responses import FileResponse
import torch
from diffusers import StableDiffusionPipeline

app = FastAPI(title="Stable Diffusion API CPU")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32  # CPU ne supporte pas float16
print(f"Device utilisé: {device}")

# Charger un modèle léger + low_cpu_mem_usage
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-base",  # modèle plus léger
    torch_dtype=dtype,
    safety_checker=None,
    low_cpu_mem_usage=True
).to(device)

# Optimisations pour CPU
pipe.enable_attention_slicing()

@app.post("/generate")
async def generate(prompt: str = Form(...)):
    try:
        with torch.inference_mode():
            image = pipe(
                prompt,
                guidance_scale=5.0,
                num_inference_steps=10,  # peu de steps = moins de RAM
                height=256,
                width=256
            ).images[0]

        filename = "generated.png"
        image.save(filename)
        return FileResponse(filename)
    except Exception as e:
        return {"error": str(e)}
