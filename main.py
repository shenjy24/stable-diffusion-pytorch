import time
from functools import partial

import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline, \
    EulerAncestralDiscreteScheduler
from huggingface_hub import snapshot_download

from utils.utils import generate_random_str


def local_model(rid, positive_prompt, negative_prompt):
    """
    加载本地模型优化版本
    :param positive_prompt: 正面提示词
    :param negative_prompt: 负面提示词
    """
    start_time = time.time()

    # 直接从缓存目录.cache中获取，将随机数目录名改为majicmix
    pipe = StableDiffusionPipeline.from_pretrained("./model/majicmix", torch_dtype=torch.float16, safety_checker=None)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    # generator = torch.Generator("cuda").manual_seed(-1)
    image = pipe(prompt=positive_prompt, negative_prompt=negative_prompt,
                 num_inference_steps=20, callback=partial(progress, rid)).images[0]
    image.save(f"image/main/{generate_random_str() + local_model.__name__}.png")

    print(f"函数 {local_model.__name__} 的运行时间为: {time.time() - start_time} 秒")


def remote_model(positive_prompt, negative_prompt):
    """
    从 hugging face 获取预训练模型
    :param negative_prompt: 正面提示词
    :param positive_prompt: 负面提示词
    """
    pipe = DiffusionPipeline.from_pretrained("digiplay/majicMIX_realistic_v6", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    image = pipe(prompt=positive_prompt, negative_prompt=negative_prompt, num_inference_steps=20).images[0]
    image.save(f"image/main/{generate_random_str() + '_' + remote_model.__name__}.png")


def download_model(repo):
    """
    下载模型
    :param repo: 模型名
    """
    snapshot_download(repo_id=repo)


def progress(rid, step, timestep, latents):
    """
    绘图过程的回调方法，用于展示进度
    :param step:
    :param timestep:
    :param latents:
    :return:
    """
    print(f"RID: {rid}, Step: {step}, Timestep: {timestep}")
    # print(f"Latents: {latents}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(torch.cuda.is_available())
    # p = "beautiful dog"
    rid = generate_random_str()
    pp = "非常英俊的男人,超写实风格,细致纹理,逼真的3D效果,量子分形,由Artgerm和Epic Game Art创作,ArtStation上的热门作品"
    np = "ng_deepnegative_v1_75t, badhandv4"
    local_model(rid, pp, np)
    # remote_model(pp, np)
