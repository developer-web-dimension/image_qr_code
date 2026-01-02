import torch
from PIL import Image
import qrcode


from diffusers import (
    StableDiffusionXLControlNetImg2ImgPipeline,
    ControlNetModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DEISMultistepScheduler,
    HeunDiscreteScheduler,
    EulerDiscreteScheduler,
)

controlnet = ControlNetModel.from_pretrained(
    "monster-labs/control_v1p_sdxl_qrcode_monster",
    torch_dtype=torch.float32
).to("cuda")

pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float32,
).to("cuda")

# pipe = pipe.to(torch.float16)
pipe.enable_sequential_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()

def resize_for_condition_image(input_image: Image.Image, resolution: int):
    """Resize input image while keeping it divisible by 64."""
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    return input_image.resize((W, H), Image.LANCZOS)


SAMPLER_MAP = {
    "DPM++ Karras SDE": lambda cfg: DPMSolverMultistepScheduler.from_config(cfg, use_karras=True, algorithm_type="sde-dpmsolver++"),
    "DPM++ Karras": lambda cfg: DPMSolverMultistepScheduler.from_config(cfg, use_karras=True),
    "Heun": lambda cfg: HeunDiscreteScheduler.from_config(cfg),
    "Euler": lambda cfg: EulerDiscreteScheduler.from_config(cfg),
    "DDIM": lambda cfg: DDIMScheduler.from_config(cfg),
    "DEIS": lambda cfg: DEISMultistepScheduler.from_config(cfg),
}


def generate_qr_image(
    qr_code_content: str,
    prompt: str,
    negative_prompt: str = "ugly, low quality, blurry, distorted, nsfw",
    guidance_scale: float = 7.5,
    controlnet_conditioning_scale: float = 1.1,
    strength: float = 0.9,
    seed: int = 1526630701,
    sampler: str = "DPM++ Karras SDE",
    width: int =  512,
    height: int =  512,
    qrcode_image: Image.Image | None = None,
):

    # Set sampler
    pipe.scheduler = SAMPLER_MAP[sampler](pipe.scheduler.config)

    # Seed control
    generator = torch.Generator(device="cuda").manual_seed(seed)

    # Generate or use QR image
    if qrcode_image is None:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=4,
        )
        qr.add_data(qr_code_content)
        qr.make(fit=True)
        qrcode_image = qr.make_image(fill_color="black", back_color="white")

    # Resize QR
    qrcode_image = resize_for_condition_image(qrcode_image, 768)

    # Use QR as init + control image
    init_image = qrcode_image
    control_image = qrcode_image

    # Run Stable Diffusion
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        control_image=control_image,
        width=width,
        height=height,
        strength=strength,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        num_inference_steps=40,
        generator=generator
    )

    return result.images[0]


if __name__ == "__main__":

    output_path = r"SDXL/output.png"

    img = generate_qr_image(
        qr_code_content="https://britanniatreatnaruto.com",
        prompt="Anime-style Naruto scene. **Glowing QR code** with **clean, sharp modules** and unobstructed finder patterns. **Subtle Hidden Leaf Village swirl motifs** around the QR code. **Warm sunlight, soft gradients**. Minimalist composition, **maximum QR readability**. Small stylized **Naruto kunai emblem** in the safe central logo space.",
        negative_prompt="blurry, low contrast, messy QR, broken QR modules, extra QR codes, duplicate patterns, text, watermark, poorly drawn anatomy, washed-out colors, weak lighting, low detail",
        guidance_scale=8,
        controlnet_conditioning_scale=1.2,
        strength=0.85,
        seed=1234567,
        sampler="DPM++ Karras SDE",
    )

    img.save(output_path)                                                           
    print(f"Saved: {output_path}")