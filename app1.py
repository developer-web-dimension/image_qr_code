import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline, AutoPipelineForImage2Image
from PIL import Image
from pyzbar.pyzbar import decode
import os

QR_PATH = r"inputs\qrcode.png"
STYLE_PATH = r"inputs\ram.png"
OUTPUT_PATH = r"outputs\final_qr_image.png"

device = "cuda" 

# ----------------------------------------------------
# File Loading and Preprocessing
# ----------------------------------------------------

def load_and_preprocess_image(path, name):
    """Loads, checks, converts, and resizes a single image."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Error: {name} image not found at: {path}")

    try:
        # Load the image
        img = Image.open(path)
        # Convert to RGB and resize to SDXL standard
        img = img.convert("RGB").resize((1024, 1024))
        print(f"Successfully loaded and preprocessed {name} image.")
        return img
    except Exception as e:
        # Catch and report the specific loading error
        raise RuntimeError(f"CRITICAL ERROR: Failed to load or process {name} image from {path}. Reason: {e}")


# Load the images. If either fails, the script will exit early with a clear error.
qr_img = load_and_preprocess_image(QR_PATH, "QR code")
style_img = load_and_preprocess_image(STYLE_PATH, "Style/Initial")


# ----------------------------------------------------
# Load ControlNet (QR-Code)
# ----------------------------------------------------
print("Loading ControlNet QR model...")
controlnet = ControlNetModel.from_pretrained(
    "monster-labs/control_v1p_sdxl_qrcode_monster",
    torch_dtype=torch.float16
)

# ----------------------------------------------------
# Load SDXL Base Model (Img2Img)
# ----------------------------------------------------
print("Loading SDXL base model...")
pipe = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16
)
pipe.controlnet = controlnet




pipe.to(device)
pipe.enable_xformers_memory_efficient_attention()

prompt = (
    "Blend the QR code into the image while keeping it readable. "
    "High quality, aesthetic, artistic, clean integration. "
    "Do not distort the QR alignment patterns."
)

# ----------------------------------------------------
# QR validator
# ----------------------------------------------------
def is_qr_scannable(path):
    # Ensure file exists before decoding
    if not os.path.exists(path):
        return False
    
    try:
        data = decode(Image.open(path))
        return len(data) > 0
    except Exception:
        # If the generated image is corrupted, it might fail to open here
        return False

# ----------------------------------------------------
# Generate + Retry Logic
# ----------------------------------------------------
print("\nGenerating QR-image...")

strength_values = [0.75, 0.65, 0.55]

# >>> DIAGNOSTIC CHECK <<<
print(f"DEBUG: Type of style_img (Image input) before pipe call: {type(style_img)}")


for strength in strength_values:
    print(f"Trying strength={strength}")
    
    result = pipe(
        prompt=prompt,
        image=style_img,
        controlnet_conditioning_image=qr_img,
        num_inference_steps=35,
        strength=strength,      
        guidance_scale=5.0,
    ).images[0]

    result.save(OUTPUT_PATH)
    print(f"Generated with strength={strength}, checking QR...")

    if is_qr_scannable(OUTPUT_PATH):
        print("SUCCESS: QR is scannable ✔")
        break

else:
    print("WARNING: QR not scannable after all attempts. Try adjusting scale or strength.")

print("Saved output to:", OUTPUT_PATH)