import time

import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DiffusionPipeline
from huggingface_hub import snapshot_download

from utils.utils import generate_random_str


def local_model(positive_prompt, negative_prompt):
    """
    加载本地模型优化版本
    :param positive_prompt: 正面提示词
    :param negative_prompt: 负面提示词
    """
    start_time = time.time()

    # 直接从缓存目录.cache中获取，将随机数目录名改为majicmix
    pipe = StableDiffusionPipeline.from_pretrained("./model/majicmix", torch_dtype=torch.float16,
                                                   safety_checker=None)
    # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    generator = torch.Generator("cuda").manual_seed(0)
    image = pipe(prompt=positive_prompt, negative_prompt=negative_prompt, generator=generator,
                 num_inference_steps=20).images[0]
    image.save(f"image/main/{generate_random_str() + local_model.__name__}.png")

    print(f"函数 {local_model.__name__} 的运行时间为: {time.time() - start_time} 秒")


def remote_model(positive_prompt, negative_prompt):
    """
    从 hugging face 获取预训练模型
    :param negative_prompt: 正面提示词
    :param positive_prompt: 负面提示词
    """
    pipe = DiffusionPipeline.from_pretrained("SG161222/Realistic_Vision_V2.0", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    image = pipe(prompt=positive_prompt, negative_prompt=negative_prompt, num_inference_steps=20).images[0]
    image.save(f"image/main/{generate_random_str() + '_' + remote_model.__name__}.png")


def download_model(repo):
    """
    下载模型
    :param repo: 模型名
    """
    snapshot_download(repo_id=repo)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(torch.cuda.is_available())
    # p = "beautiful dog"
    pp = "Best quality, masterpiece, ultra high res, (photorealistic:1.4), 1girl"
    np = "badhandv4"
    local_model(pp, np)
    # remote_model(pp, np)
