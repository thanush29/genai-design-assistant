# app.py
"""
GenAI Design Assistant — Stable Diffusion 2.1 (Crisp Text Edition)
Auto Type Detection + Pillow Overlay for Sharp Labels & Dimensions
"""

import os, re, torch, gradio as gr
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# -------------------------  MODEL  -------------------------
MODEL_ID = os.environ.get("MODEL_ID", "stabilityai/stable-diffusion-2-1")
HF_TOKEN = os.environ.get("HF_TOKEN", None)

def load_pipe():
    auth = {"use_auth_token": HF_TOKEN} if HF_TOKEN else {}
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=dtype, safety_checker=None, **auth
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.enable_attention_slicing()
    return pipe

pipe = load_pipe()

# -------------------------  HELPERS  -------------------------
NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, gibberish text, watermark, duplicate, noise, "
    "crooked lines, wrong perspective, extra walls, unrealistic proportions"
)

def detect_image_type(prompt: str):
    floor = ["floor", "layout", "blueprint", "plan", "bhk", "villa", "map", "room size"]
    mood = ["interior", "decor", "sofa", "lamp", "texture", "wall", "mood", "bedroom"]
    p = prompt.lower()
    if any(k in p for k in floor): return "Floorplan"
    if any(k in p for k in mood): return "Moodboard"
    return "Floorplan"

def refine_prompt(prompt: str, kind: str):
    p = prompt.strip().capitalize()
    if kind == "Floorplan":
        return (
            f"{p}, professional architectural 2D top-down floor plan, "
            "technical drawing, clean black lines, white background, "
            "precise proportions, schematic layout, no blur, no text"
        )
    return (
        f"{p}, high-end interior design moodboard collage, photorealistic, "
        "textures, furniture, warm lighting, 8k render, aesthetic composition"
    )

def extract_labels_dims(text: str):
    text = text.replace("×", "x").replace("X", "x")
    pairs = re.findall(r"([A-Za-z0-9_ ]{2,30})\s+(\d+\.?\d*\s*x\s*\d+\.?\d*)", text)
    labels, dims = [], []
    for l, d in pairs:
        labels.append(l.strip())
        dims.append(d.replace(" ", ""))
    return labels, dims

# -------------------------  POSTPROCESS  -------------------------
def load_font(size=32):
    try:
        return ImageFont.truetype("DejaVuSansMono-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()

def overlay_text(image: Image.Image, labels, dims):
    """Add ultra-sharp text labels/dimensions evenly spaced on image."""
    draw = ImageDraw.Draw(image)
    w, h = image.size
    n = max(len(labels), len(dims))
    step_y = max(80, h // (n + 2))
    x_left, x_right = int(w * 0.05), int(w * 0.55)
    font = load_font(28)
    for i in range(n):
        y = 50 + i * step_y
        label = labels[i] if i < len(labels) else ""
        dim = dims[i] if i < len(dims) else ""
        # Text with white stroke for clarity
        draw.text((x_left, y), f"{label}", fill=(0,0,0),
                  font=font, stroke_width=3, stroke_fill=(255,255,255))
        draw.text((x_right, y), f"{dim}", fill=(30,30,30),
                  font=font, stroke_width=2, stroke_fill=(255,255,255))
    return image

def enhance_image(image: Image.Image):
    image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=2))
    image = ImageEnhance.Sharpness(image).enhance(1.6)
    image = ImageEnhance.Contrast(image).enhance(1.15)
    image = ImageEnhance.Brightness(image).enhance(1.03)
    return image

# -------------------------  GENERATION  -------------------------
def generate(prompt, steps, scale, seed, aspect, show_text):
    kind = detect_image_type(prompt)
    refined = refine_prompt(prompt, kind)
    labels, dims = extract_labels_dims(prompt)
    width, height = (1024,768) if aspect=="Wide" else (768,1024) if aspect=="Tall" else (896,896)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=device).manual_seed(int(seed))
    result = pipe(prompt=refined, negative_prompt=NEGATIVE_PROMPT,
                  num_inference_steps=int(steps), guidance_scale=float(scale),
                  generator=gen, width=width, height=height)
    img = result.images[0].convert("RGB")
    img = enhance_image(img)
    if show_text and kind == "Floorplan" and (labels or dims):
        img = overlay_text(img, labels, dims)
    img = enhance_image(img)
    save = f"output_{kind.lower()}_{seed}.png"
    img.save(save)
    return img, save, kind

# -------------------------  UI  -------------------------
examples = [
    ["Modern 3BHK apartment living 4x5, kitchen 3x3, bed1 4x4", 40, 9.0, 42, "Wide", True],
    ["Scandinavian bedroom mood board with oak wood textures and white linen", 35, 8.5, 77, "Square", False],
]

css = """
body { background: linear-gradient(135deg,#edf2fb,#e3ebf9); }
.gradio-container { font-family:'Segoe UI',Roboto,sans-serif; }
h2 { color:#111;font-weight:600; }
"""

with gr.Blocks(title="Thanush GEN AI IMAGE", css=css) as demo:
    gr.Markdown("<h2>Thanush GenAI Design Assistant — Crisp Text Edition</h2>")
    gr.Markdown("Smartly generates floorplans or moodboards with perfectly readable dimensions and labels.")

    with gr.Row():
        with gr.Column(scale=3):
            prompt_in = gr.Textbox(lines=3, label="📝 Design Prompt",
                placeholder="e.g., 3BHK apartment layout: living 4x5, kitchen 3x3, bed1 4x4")
            aspect = gr.Radio(["Wide","Square","Tall"], value="Wide", label="Aspect Ratio")
            show_text = gr.Checkbox(True, label="Add Sharp Labels & Dimensions")
            steps = gr.Slider(20,75,40,1,label="Inference Steps")
            scale = gr.Slider(5,12,9.0,0.5,label="Guidance Scale")
            seed = gr.Number(42, label="Seed")
            btn = gr.Button("✨ Generate Crisp Image", variant="primary")
            gr.Examples(examples=examples, inputs=[prompt_in, steps, scale, seed, aspect, show_text])
        with gr.Column(scale=4):
            out_img = gr.Image(type="pil", label="Generated Image")
            out_kind = gr.Textbox(label="🧭 Detected Type", interactive=False)
            out_file = gr.File(label="Download PNG")

    btn.click(fn=generate, inputs=[prompt_in, steps, scale, seed, aspect, show_text],
              outputs=[out_img, out_file, out_kind])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)
