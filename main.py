# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import time
import torch
from huggingface_hub import snapshot_download
from diffusers import StableDiffusionPipeline

from utils.utils import generate_random_str


def local_model(prompt):
    """
    加载本地模型
    :param prompt: 提示词
    """
    start_time = time.time()
    # 直接从缓存目录.cache中获取，将随机数目录名改为majicmix
    pipe = StableDiffusionPipeline.from_pretrained("./model/majicmix", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    image = pipe(prompt).images[0]
    image.save(f"image/{generate_random_str()}.png")

    print(f"函数 {local_model.__name__} 的运行时间为: {time.time() - start_time} 秒")


def from_pretrained(prompt):
    """
    从 hugging face 获取预训练模型
    :param prompt: 提示词
    """
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    image = pipe(prompt).images[0]
    image.save(f"image/astronaut_rides_horse.png")


def download_model(repo):
    """
    下载模型
    :param repo: 模型名
    """
    snapshot_download(repo_id=repo)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(torch.cuda.is_available())
    p = "Very handsome boy, detailed textures, photorealistic 3d, quantum fractals, art by artgerm and Epic Game Art, trending on artstation"
    local_model(p)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
