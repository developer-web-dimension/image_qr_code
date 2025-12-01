import torch
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
from PIL import Image
from pyzbar.pyzbar import decode
import os

QR_PATH = r"inputs\qrcode.png"
STYLE_PATH = r"inputs\ram.png"
OUTPUT_PATH = r"outputs\final_qr_image.png"

device = "cuda"

# ----------------------------------------------------
# Load images (512x512 for SD15)
# ----------------------------------------------------
def load_img(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} missing: {path}")
    img = Image.open(path).convert("RGB").resize((512, 512))
    print(f"Loaded {name}")
    return img

qr_img = load_img(QR_PATH, "QR")
style_img = load_img(STYLE_PATH, "Style")


# ----------------------------------------------------
# Load ControlNet (SD15 QR version)
# ----------------------------------------------------
print("Loading ControlNet SD15 QR...")
controlnet = ControlNetModel.from_pretrained(
    "monster-labs/control_v1p_sd15_qrcode_monster",
    torch_dtype=torch.float16
)

# ----------------------------------------------------
# Load SD 1.5 Img2Img Pipeline
# ----------------------------------------------------
print("Loading SD15 Img2Img...")
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None
)

pipe.enable_model_cpu_offload()    
pipe.enable_xformers_memory_efficient_attention()

prompt = (
    "Blend the QR code into the image while keeping it readable, clean, artistic."
)

# ----------------------------------------------------
# QR validator
# ----------------------------------------------------
def is_qr_scannable(path):
    try:
        return len(decode(Image.open(path))) > 0
    except:
        return False

# ----------------------------------------------------
# Generate + Retry Logic
# ----------------------------------------------------
print("\nGenerating QR-image...")

strength_list = [0.55, 0.45, 0.35]

for strength in strength_list:
    print(f"Trying strength={strength}")

    result = pipe(
        prompt=prompt,
        image=style_img,
        controlnet_conditioning_image=qr_img,
        strength=strength,
        num_inference_steps=20,   # 🔥 Keep low for 4GB
        guidance_scale=7.0,
    ).images[0]

    result.save(OUTPUT_PATH)

    if is_qr_scannable(OUTPUT_PATH):
        print("SUCCESS: Scannable QR ✔")
        break

else:
    print("WARNING: QR not scannable after attempts.")

print("Saved output:", OUTPUT_PATH)
