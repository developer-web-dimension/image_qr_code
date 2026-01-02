import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    AutoencoderKL,
)
from PIL import Image
import qrcode
import os


def generate_qr_condition_image(
    qr_content: str,
    canvas_size: int = 1024,
    module_px: int = 16,
    border_modules: int = 4,
    background_gray: int = 200,
) -> Image.Image:

    # High error correction = more room for art
    qr = qrcode.QRCode(
        version=None,  # let lib pick appropriate size
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=module_px // 2,  # we'll upscale later to get ~16px modules
        border=border_modules,
    )
    qr.add_data(qr_content)
    qr.make(fit=True)

    qr_img = qr.make_image(fill_color="black", back_color="white").convert("L")

    # Compute final QR size to get ~module_px per module
    # qr_img.width = (modules + 2*border) * box_size
    # We upscale to reach ~16px/module
    modules_plus_border = qr_img.width // (module_px // 2)
    target_size = modules_plus_border * module_px

    qr_img = qr_img.resize((target_size, target_size), resample=Image.NEAREST)

    # Create gray background canvas
    canvas = Image.new("L", (canvas_size, canvas_size), color=background_gray)

    # Center QR on canvas
    x = (canvas_size - target_size) // 2
    y = (canvas_size - target_size) // 2
    canvas.paste(qr_img, (x, y))

    # Convert to 3-channel RGB for ControlNet
    canvas_rgb = canvas.convert("RGB")
    return canvas_rgb


def load_sdxl_qr_pipeline(device: str = "cpu"):

    controlnet = ControlNetModel.from_pretrained(
        "monster-labs/control_v1p_sdxl_qrcode_monster",
        torch_dtype=torch.float32,
    )

    # Recommended FP16 VAE for SDXL ControlNet
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch.float32,
    )

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float32,
        variant="fp16",
        use_safetensors=True,
    )

    pipe.to(device)
    return pipe


def generate_qr_art(
    qr_content: str,
    prompt: str,
    negative_prompt: str = "low quality, blurry, distorted, messy, broken qr, unreadable, low contrast, noisy, artifacts",
    guidance_scale: float = 7.0,
    controlnet_conditioning_scale: float = 1.4,
    num_inference_steps: int = 30,
    seed: int | None = None,
    output_path: str = "qr_art_sdxl.png",
    device: str = "cpu",
):
    # 1) Build condition image (QR on gray background)
    cond_image = generate_qr_condition_image(qr_content)

    # 2) Load pipeline
    pipe = load_sdxl_qr_pipeline(device=device)

    # 3) Seed
    generator = torch.Generator(device=device)
    if seed is not None:
        generator = generator.manual_seed(seed)

    # 4) Run diffusion
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=cond_image,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        generator=generator,
        height=768,
        width=768,
    )

    image = result.images[0]

    # 5) Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    image.save(output_path)
    print(f"✅ Saved QR art to: {output_path}")

    return image


if __name__ == "__main__":
    qr_content = "https://example.com/naruto-qr-demo"

    prompt = "High-energy Naruto-style QR code art, QR code forming the entire backdrop with stylized chakra-infused finder patterns glowing brightly. A dramatic ninja chakra figure in front, arm extended as if unleashing a Rasengan-like burst of energy, intense orange and blue rim lighting, neon reflections on the figure, cinematic anime shading, explosive glow effects, clean QR modules preserved for readability."

    generate_qr_art(
        qr_content=qr_content,
        prompt=prompt,
        seed=1234567,
        output_path="outputs/naruto_qr_sdxl.png",
        guidance_scale=8.0,
        controlnet_conditioning_scale=1.2, 
        num_inference_steps=10,
    )
