import time
from functools import partial
from io import BytesIO

import requests
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DiffusionPipeline, \
    EulerAncestralDiscreteScheduler, StableDiffusionUpscalePipeline, StableDiffusionLatentUpscalePipeline, \
    EulerDiscreteScheduler
from huggingface_hub import snapshot_download

from utils.utils import generate_random_str


def load_safetensors(positive_prompt, negative_prompt):
    # pipeline = StableDiffusionPipeline.from_single_file(
    #     "./model/twingShadow_v12.safetensors"
    # )
    pipeline = StableDiffusionPipeline.from_single_file(
        "https://huggingface.co/JonasSim/TWingshadow_v1.4/blob/main/twing_shadow_v1.4.safetensors",
        torch_dtype=torch.float16, use_safetensors=True
    )
    pipeline = pipeline.to("cuda")
    image = pipeline(prompt=positive_prompt, negative_prompt=negative_prompt, num_inference_steps=20).images[0]
    return image


def upscale_image(request_id, positive_prompt, negative_prompt, low_res_img):
    # load model and scheduler
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        model_id, revision="fp16", torch_dtype=torch.float16
    )
    pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to("cuda")
    upscale_img = pipeline(prompt=positive_prompt, negative_prompt=negative_prompt, image=low_res_img,
                           num_inference_steps=20, callback=partial(progress, request_id)).images[0]
    return upscale_img


def latent_upscale_image(request_id, positive_prompt, negative_prompt, low_res_img):
    # load model and scheduler
    model_id = "stabilityai/sd-x2-latent-upscaler"
    pipeline = StableDiffusionLatentUpscalePipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    )
    pipeline = pipeline.to("cuda")
    generator = torch.manual_seed(33)
    upscale_img = pipeline(prompt=positive_prompt, negative_prompt=negative_prompt, image=low_res_img,
                           num_inference_steps=20, callback=partial(progress, request_id),
                           guidance_scale=0, generator=generator).images[0]
    return upscale_img


def upscale(request_id, model_id, positive_prompt, negative_prompt):
    img = get_image(request_id, model_id, positive_prompt, negative_prompt)
    img.save(f"image/main/img.png")
    upscale_img = upscale_image(request_id, positive_prompt, negative_prompt, img)
    upscale_img.save(f"image/main/upscale_img.png")
    return upscale_img


def upscale_local_image(request_id, image_url, positive_prompt, negative_prompt):
    low_res_img = Image.open(image_url)
    img = upscale_image(request_id, positive_prompt, negative_prompt, low_res_img)
    return img


def latent_upscale_local_image(request_id, image_url, positive_prompt, negative_prompt):
    low_res_img = Image.open(image_url)
    img = latent_upscale_image(request_id, positive_prompt, negative_prompt, low_res_img)
    return img


def upscale_remote_image(request_id, image_url, positive_prompt, negative_prompt):
    response = requests.get(image_url)
    low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
    img = upscale_image(request_id, positive_prompt, negative_prompt, low_res_img)
    return img


def latent_upscale(request_id, model_id, positive_prompt, negative_prompt):
    img = get_image(request_id, model_id, positive_prompt, negative_prompt)
    img.save(f"image/main/img.png")
    latent_upscale_img = latent_upscale_image(request_id, positive_prompt, negative_prompt, img)
    latent_upscale_img.save(f"image/main/latent_upscale_img.png")
    return latent_upscale_img


def get_image(request_id, model_id, positive_prompt, negative_prompt):
    """
    加载本地模型优化版本
    :param request_id:
    :param model_id:
    :param positive_prompt: 正面提示词
    :param negative_prompt: 负面提示词
    """
    # 直接从缓存目录.cache中获取，将随机数目录名改为majicmix
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None,
                                                   force_download=True, resume_download=False)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    image = pipe(prompt=positive_prompt, negative_prompt=negative_prompt, num_inference_steps=20,
                 callback=partial(progress, request_id)).images[0]
    return image


def local_model(request_id, positive_prompt, negative_prompt):
    """
    加载本地模型优化版本
    :param request_id:
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
                 num_inference_steps=20, callback=partial(progress, request_id)).images[0]
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


def progress(request_id, step, timestep, latents):
    """
    绘图过程的回调方法，用于展示进度
    :param request_id:
    :param step:
    :param timestep:
    :param latents:
    :return:
    """
    torch.cuda.empty_cache()
    print(f"RID: {request_id}, Step: {step}, Timestep: {timestep}")
    # print(f"Latents: {latents}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(torch.cuda.is_available())
    pp = "young,1girl"
    np = "ng_deepnegative_v1_75t, badhandv4"
    request_id = generate_random_str()
    # upscale(request_id, "./model/majicmix", pp, np)
    latent_upscale(request_id, "./model/majicmix", pp, np)
    # img = latent_upscale_local_image(request_id, "image/main/img.png", pp, np)
    # img.save(f"image/main/upscale_local_image.png")
