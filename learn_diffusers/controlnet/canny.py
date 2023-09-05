import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from diffusers.utils import load_image


def canny_image(image):
    """
    边缘检测，返回预处理后的图片
    :param image: Image图片
    :return: 预处理后的轮廓图片
    """
    image = np.array(image)

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_img = Image.fromarray(image)
    return canny_img


def use_controlnet(canny_img):
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )
    # scheduler can drastically reduce inference time
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # When running enable_model_cpu_offload, do not manually move the pipeline to GPU with .to("cuda")
    # once CPU offloading is enabled, the pipeline automatically takes care of GPU memory management
    pipe.enable_model_cpu_offload()
    # pipe.enable_xformers_memory_efficient_attention()

    prompt = ", best quality, extremely detailed"
    prompt = [t + prompt for t in ["Sandra Oh", "Kim Kardashian", "rihanna", "taylor swift"]]
    generator = [torch.Generator(device="cpu").manual_seed(2) for i in range(len(prompt))]

    output = pipe(
        prompt,
        canny_img,
        negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * 4,
        num_inference_steps=20,
        generator=generator,
    )

    image_grid(output.images, 2, 2)


def image_grid(imgs, rows, cols):
    """
    图片显示工具
    :param imgs: 图片列表
    :param rows: 行
    :param cols: 列
    :return:
    """
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


if __name__ == '__main__':
    img_url = "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
    reference_img = load_image(img_url)
    canny_img = canny_image(reference_img)
    use_controlnet(canny_img)
