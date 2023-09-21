import torch
from diffusers import StableDiffusionPipeline, StableDiffusionUpscalePipeline, EulerAncestralDiscreteScheduler


def load_safetensors(positive_prompt, negative_prompt):
    """
    加载safetensors格式模型，目前官方API还有BUG
    :param positive_prompt:
    :param negative_prompt:
    :return:
    """
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


def upscale_image(positive_prompt, negative_prompt, low_res_img):
    """
    放大图片，提高图片分辨率
    :param positive_prompt:
    :param negative_prompt:
    :param low_res_img:
    :return:
    """
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


def get_image(model_id, positive_prompt, negative_prompt):
    """
    加载本地模型优化版本
    :param model_id: 模型
    :param positive_prompt: 正面提示词
    :param negative_prompt: 负面提示词
    """
    # 直接从缓存目录.cache中获取，将随机数目录名改为majicmix
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None,
                                                   force_download=True, resume_download=False)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    image = pipe(prompt=positive_prompt, negative_prompt=negative_prompt, num_inference_steps=20).images[0]
    return image
