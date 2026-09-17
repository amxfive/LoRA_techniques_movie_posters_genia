# LoRA_techniques_movie_posters_genia# 🎬 Movie Poster Generation via Fine-Tuned Stable Diffusion v1.5 (LoRA)

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)
![Diffusers](https://img.shields.io/badge/HuggingFace-Diffusers-yellow.svg)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-green.svg)
![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20T4%20(16GB)-76B900.svg)

---

## 📌 Authors
* **Álvaro Lorenzo Hidalgo**
* **Marcos Ortíz Durán**[cite: 4]

*Universidad Loyola (Sevilla, España)*[cite: 4]  
*Course: Generative Artificial Intelligence (Master's Degree in AI)*[cite: 4]

---

## 📸 Project Overview

This project focuses on the specialized generation of **movie posters** across three distinct genres: **Animation, Horror, and Romance**[cite: 4]. While base text-to-image models like Stable Diffusion v1.5 excel at general image synthesis, they struggle to capture the rigid structural composition and stylistic nuance required for film promotional material[cite: 4].

To solve this efficiently without full fine-tuning, we apply **LoRA (Low-Rank Adaptation)**[cite: 4] to inject low-rank trainable matrices into the U-Net attention layers[cite: 4]. We systematically evaluate the impact of **LoRA Rank ($r$)**, **Semantic Prompt Depth**, **Adapter Weights ($w$)**, and **Inference Steps ($t$)**[cite: 4].

---

## 🏗️ Architecture & Preprocessing

### 1️⃣ Base Architecture: Stable Diffusion v1.5
* **VAE (Variational Autoencoder)**: Compresses $512 \times 512$ pixel space into latent space and decodes generated latents back into image space[cite: 4].
* **U-Net (Denoising Network)**: Predicts and subtracts noise iteratively conditioned on text embeddings[cite: 4].
* **Text Encoder (CLIP)**: Converts text prompts into latent vectors to guide the U-Net[cite: 4].

### 2️⃣ Dataset & Preprocessing Pipeline
* **Source Dataset**: `skvarre/movie-posters` (~10,000 movie posters with metadata)[cite: 4].
* **Filtering & Stratification**: Sub-sampled balanced subset across **Horror, Romance, and Animation** to prevent class bias[cite: 4].
* **Visual Standardisation**: RGB conversion and resized uniformly to $512 \times 512$ pixels[cite: 4].
* **Automated Prompt Synthesizer**: Built synthetic prompts using two templates[cite: 4]:
  * *Style 1 (Genre only)*: `A cinematic movie poster in the style of [GENRES], high quality`[cite: 4]
  * *Style 2 (Genre + Plot Overview)*: `A cinematic movie poster in the style of [GENRES], about [OVERVIEW], high quality`[cite: 4]
* **Metadata Export**: Structured as JSONL for direct loading via HuggingFace `diffusers`[cite: 4].

---

## 🔬 Experimental Setup & Hyperparameters

Experiments were run on an **NVIDIA T4 GPU (16GB VRAM)** via Google Colab with seed fixed to `42` for exact noise reproducibility[cite: 4].

### General Training Configuration

| Parameter | Value | Parameter | Value |
| :--- | :--- | :--- | :--- |
| **Base Model** | Stable Diffusion v1.5[cite: 4] | **Batch Size** | 2[cite: 4] |
| **Resolution** | $512 \times 512$ px[cite: 4] | **Epochs** | 4[cite: 4] |
| **Learning Rate** | $1 \times 10^{-4}$[cite: 4] | **Precision** | `fp16`[cite: 4] |
| **Scheduler** | Cosine[cite: 4] | **Gradient Clipping** | Max norm 1.0[cite: 4] |
| **Augmentation** | Random Horizontal Flip[cite: 4] | **Tracking** | Weights & Biases (WandB)[cite: 4] |

### Experimental Variants

| Model Variant | Rank ($r$) | Prompt Structure |
| :--- | :---: | :--- |
| **Model A** | 32[cite: 4] | Movie Genres only[cite: 4] |
| **Model B** | 128[cite: 4] | Movie Genres only[cite: 4] |
| **Model C** | 128[cite: 4] | Movie Genres + Plot Overview[cite: 4] |

---

## 🧪 Evaluation Experiments

1. **Weight Sweeping ($w \in [0.0, 1.8]$)**: Varying LoRA strength to find the sweet spot between poster layout adoption and anatomical/visual fidelity[cite: 4].
2. **Inference Steps Convergence ($t \in \{5, 15, 30, 50, 100\}$)**: Evaluating quality vs. computational cost under fixed weight $w=1.0$[cite: 4].
3. **Genre Generalization**: Testing cross-genre versatility (Dark/slasher horror, Ghibli/Disney animation, emotional drama/romance)[cite: 4].

---

## 📈 Key Results & Analysis

### 1️⃣ Impact of Rank ($r=32$ vs $r=128$)
* **Model A ($r=32$)**: Captures color palettes and genre atmosphere, but **fails to learn structural poster composition** (missing typography blocks/titles)[cite: 4].
* **Model B ($r=128$)**: Successfully generates poster layout structure, simulated title placement, and character composition (at the cost of larger weight files and training time)[cite: 4].

### 2️⃣ Impact of Semantic Context (Model B vs Model C)
* Including the plot overview (**Model C**) significantly enhances **atmospheric fidelity**[cite: 4].
* While Model B produces standard genre interpretations, Model C leverages plot details to fine-tune lighting, contrast, and color desaturation (e.g., darker, moodier lighting for horror)[cite: 4].

### 3️⃣ Loss Curve Behavior & Epoch Duration
* Training loss drops sharply in early steps and levels off quickly, as the robust pre-trained base model (SD v1.5) rapidly aligns prior knowledge to the new style[cite: 4].
* However, early stopping at Epoch 1 produces suboptimal posters lacking spatial structure[cite: 4]. The full **4 epochs** are strictly required to learn structural poster layout despite minimal loss changes[cite: 4].

### 4️⃣ Sensitivity Analysis (Adapter Weight & Steps)
* **LoRA Weight ($w$)**: At $w=0.0$, output is generic SD v1.5[cite: 4]. As $w$ increases toward $1.0$, poster attributes and pseudo-typography blocks appear[cite: 4]. Over-scaling ($w > 1.4$) introduces severe saturation artifacts[cite: 4].
* **Inference Steps ($t$)**: Results at $t < 15$ are blurry/abstract[cite: 4]. Visual stability, edge sharpness, and contrast peak at **$t = 50$**[cite: 4]. Pushing beyond $t=50$ adds compute cost without perceptual gain[cite: 4].

---

## 💡 Key Conclusions

1. **Rank Controls Structure**: Lower ranks ($r=32$) only learn textures, whereas higher ranks ($r=128$) are necessary to learn complex spatial layouts like movie posters and typography placement[cite: 4].
2. **Prompts Drive Atmosphere**: Rich semantic prompts combining genres and plot overviews (Model C) provide superior visual lighting and thematic alignment compared to genre labels alone[cite: 4].
3. **Optimal Generation Settings**: The ideal balance between quality and compute is **$w = 1.0$** and **$t = 50$ sampling steps**[cite: 4].
4. **Current Limitations & Future Work**: While SD v1.5 + LoRA replicates poster layouts and pseudo-text blocks, producing crisp, fully legible text remains an architectural limit of SD v1.5[cite: 4]. Future work includes integrating **ControlNet** for structural layouts or migrating to **SDXL** for high-res typography rendering[cite: 4].

---

## 🛠️ Requirements and Quick Start
# Clone repository
git clone [https://github.com/your-username/sd-lora-movie-posters.git](https://github.com/your-username/sd-lora-movie-posters.git)
cd sd-lora-movie-posters

# Install dependencies
pip install torch torchvision diffusers transformers accelerate peft xformers