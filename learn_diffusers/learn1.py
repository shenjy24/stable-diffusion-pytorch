import torch
from diffusers import DDPMPipeline

from utils.utils import generate_random_str


def deconstruct_pipeline():
    ddpm = DDPMPipeline.from_pretrained("google/ddpm-cat-256").to("cuda")
    image = ddpm(num_inference_steps=20).images[0]
    image.save(f"../image/learn1/{generate_random_str() + 'cat'}.png")


if __name__ == '__main__':
    print(torch.cuda.is_available())
    deconstruct_pipeline()
