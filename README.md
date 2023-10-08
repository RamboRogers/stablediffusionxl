# Stable Diffusion XL on Gradio

This is an interface using Gradio to iteract with Stable Diffusion XL.  I wrote this to learn Gradio and wanted to share for 💪 others.  I have about four hours of dev time building this and learning Gradio, Pytorch and Python.  I believe the next version of this could be built much ⚡ faster.

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) 

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

---

## Installation

This application is simple to install. You need python installed, pytorch, and an Nvidia GPU to use this interface. [Anaconda](https://www.anaconda.com/download) is a great distribution.


```bash
# Example:
git clone https://github.com/ramborogers/stablediffusionxl.git
cd stablediffusionxl
pip install -r requirements.txt
```
## Usage

Just launch the application using python.  You may want to activate conda.

```bash
# Example:
python app.py
```

Open up your browser to [http://127.0.0.1:7860/](http://127.0.0.1:7860/) and start creating!  The gallery will update after the first render.

![usage](usage.png)

## Acknowledgements

[Gradio](https://www.gradio.app/) is amazing, and so is [Stable Diffusion XL](https://huggingface.co/docs/diffusers/using-diffusers/sdxl)!

[matthewrogers.org](https://matthewrogers.org)
