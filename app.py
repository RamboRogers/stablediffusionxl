from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch
import gradio as gr
import random
import socket
import re
import cpuinfo
import gpustat
import pathlib

def print_system_info():
    # print CPU information
    cpu_info = cpuinfo.get_cpu_info()
    cpu_str = f"CPU:\n  Model: {cpu_info['brand_raw']}\n  Cores: {cpu_info['count']}\n"

    # print GPU information
    gpu_stats = gpustat.GPUStatCollection.new_query()
    gpu_str = "GPU:\n"
    for i, gpu in enumerate(gpu_stats.gpus):
        gpu_str += f"  GPU {i}: {gpu.name}\n"

    return cpu_str + gpu_str

#Preload model/llm
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16",add_watermarker=False
).to("cuda")

def to_filename(text, max_length=200):
    # replace any non-alphanumeric characters with an empty string
    filename = re.sub(r'\W+', '', text)
    # replace any spaces with an underscore
    filename = re.sub(r'\s+', '_', filename)
    # truncate the filename to the desired length
    filename = filename[:max_length]
    return filename

#Description in Markdown
description = """
This is a demo ⚡ of Stable Diffusion XL, a new diffusion model that can generate high quality images. [Stable Diffusion XL](https://huggingface.co/docs/diffusers/using-diffusers/sdxl)

It is being presented 📦 using Gradio, a library for quickly creating UIs for machine learning models. [Gradio](https://www.gradio.app/docs/interface)

"""

article = """
Coded 🧾 by [Matthew Rogers](https://matthewrogers.org)

Running on:
"""

# get system information
system_info = print_system_info()
# append system information to article
article += system_info

# create a directory to save the generated images
saved_dir = pathlib.Path("saved")
saved_dir.mkdir(parents=True, exist_ok=True)

# Function to generate the output and the image
def generateImage(prompt, prompt_2,num_inference_steps=50,seed=0):
    
    #create image, use default 50 for num_inference_steps
    # set a random seed if seed is not defined
    if seed == 0 or seed == None:
        seed = random.randint(0, 2**32 - 1)
        print("seed:",seed)
    # set the seed for the PyTorch random number generator
    torch.manual_seed(seed)

    if num_inference_steps == 0:
        num_inference_steps = 50

    image = pipe(prompt=prompt,prompt_2=prompt_2, width=1024, height=1024, num_inference_steps=num_inference_steps, ).images[0]
    used_seed = torch.initial_seed()
    print("used_seed:",used_seed)

    # generate a filename based on the input prompts and the number of inference steps
    filename = to_filename(f"{prompt}_{prompt_2}_{num_inference_steps}_{used_seed}", max_length=200)
    # add the file extension
    filename = f"saved/{filename}.png"

    image.save( filename, "PNG")

    #return values
    return image, used_seed, num_inference_steps, (sorted(saved_dir.glob("*.png"), key=lambda f: f.stat().st_ctime, reverse=True))

gui = gr.Interface(
    fn=generateImage,
    inputs=[gr.Textbox(lines=2,label="Prompt"), gr.Textbox(lines=2,label="Prompt2"), gr.Slider(0, 150, value=80, label="Steps"),gr.Number(label="Seed")],
    outputs=[gr.Image(label="Generated Image"), gr.Number(label="Seed"), gr.Number(label="Steps"), gr.Gallery(label="Gallery")],
    title="Stable Diffusion X on Gradio",
    description=description,
    article=article,
    allow_flagging="never",
)

gui.launch()