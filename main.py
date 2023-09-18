import time
from functools import partial

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DiffusionPipeline, \
    EulerAncestralDiscreteScheduler, StableDiffusionUpscalePipeline
from huggingface_hub import snapshot_download

from utils.utils import generate_random_str


def load_safetensors(positive_prompt, negative_prompt):
    pipeline = StableDiffusionPipeline.from_single_file(
        "https://huggingface.co/JonasSim/TWingshadow_v1.4/blob/main/twing_shadow_v1.4.safetensors",
        torch_dtype=torch.float16, use_safetensors=True
    )
    pipeline = pipeline.to("cuda")
    image = pipeline(prompt=positive_prompt, negative_prompt=negative_prompt, num_inference_steps=20).images[0]
    return image


def upscale_image(positive_prompt, negative_prompt, low_res_img):
    # load model and scheduler
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        model_id, revision="fp16", torch_dtype=torch.float16
    )
    pipeline = pipeline.to("cuda")

    # let's download an  image
    # url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
    # response = requests.get(url)
    # low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
    # low_res_img = low_res_img.resize((128, 128))
    # prompt = "a white cat"

    upscale_img = pipeline(prompt=positive_prompt, negative_prompt=negative_prompt, image=low_res_img).images[0]
    return upscale_img


def get_image(model, positive_prompt, negative_prompt):
    """
    加载本地模型优化版本
    :param positive_prompt: 正面提示词
    :param negative_prompt: 负面提示词
    """
    # 直接从缓存目录.cache中获取，将随机数目录名改为majicmix
    pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16, safety_checker=None,
                                                   force_download=True, resume_download=False)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    image = pipe(prompt=positive_prompt, negative_prompt=negative_prompt, num_inference_steps=20).images[0]
    return image


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
    # rid = generate_random_str()
    pp = "beautiful,young,1girl"
    np = "ng_deepnegative_v1_75t, badhandv4"
    # local_model(rid, pp, np)
    # remote_model(pp, np)
    image1 = get_image("./model/twing_shadow_v1.2", pp, np)
    image1.save(f"image/main/{generate_random_str() + '_' + get_image.__name__}.png")

    # image2 = Image.open('image/main/87c0b48ba96c4db8a6c2d1ace2fa4110_get_image.png')
    # img = upscale_image(pp, np, image2)
    # img.save(f"image/main/{generate_random_str() + '_' + upscale_image.__name__}.png")
    #
    # img = load_safetensors(pp, np)
    # img.save(f"image/main/{generate_random_str() + '_' + load_safetensors().__name__}.png")
