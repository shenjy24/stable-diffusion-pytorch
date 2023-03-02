# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
from diffusers import StableDiffusionPipeline


def stable_diffusion(prompt):
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    image = pipe(prompt).images[0]
    image.save(f"image/astronaut_rides_horse.png")
    image


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(torch.cuda.is_available())
    stable_diffusion("a photograph of an astronaut riding a horse")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
