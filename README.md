# 🧠 GenAI Design Assistant — Stable Diffusion 2.1 (Crisp Text Edition)

> Auto Floorplan & Moodboard Generator with ultra-sharp text overlays powered by **Stable Diffusion 2.1** and **Gradio UI**.

---

## 🎯 Overview

This app intelligently distinguishes between **floorplans** and **moodboards** from a single text prompt — and generates crisp, professional visuals using the **Stable Diffusion 2.1** model.  

If it detects a floorplan, it overlays sharp, readable **labels and dimensions** automatically.  
If it detects a moodboard, it creates a **photorealistic, aesthetic collage** for interiors or decor concepts.

---

## 🧩 Features

- 🔍 **Auto-type detection:** Detects “Floorplan” or “Moodboard” from prompt context.  
- 🧱 **Sharp overlays:** Adds text labels & dimensions with Pillow and custom stroke rendering.  
- 🧠 **Refined prompt engineering:** Enhances inputs for realistic, clean results.  
- ⚡ **Optimized performance:** Uses DPMSolverMultistepScheduler + attention slicing for faster inference.  
- 🎨 **Interactive UI:** Built with Gradio Blocks, complete with sliders, checkboxes, and examples.  
- 💾 **Auto-save:** Saves every generated image locally as `output_<type>_<seed>.png`.

---

## 🛠️ Installation

### 1️⃣ Clone this repo
```bash
git clone https://github.com/thanush29/genai-design-assistant.git
cd genai-design-assistant


Demo link: https://2675514e9496685501.gradio.live/