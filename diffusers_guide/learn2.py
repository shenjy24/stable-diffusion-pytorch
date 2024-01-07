import torch
from PIL import Image
from diffusers import DDPMScheduler, UNet2DModel
from utils.utils import generate_random_str


def deno_process():
    """
    去噪流程
    :return:
    """
    scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
    model = UNet2DModel.from_pretrained("google/ddpm-cat-256").to("cuda")
    # Set the number of timesteps to run the denoising process
    scheduler.set_timesteps(50)

    sample_size = model.config.sample_size
    noise = torch.randn((1, 3, sample_size, sample_size)).to("cuda")

    input_noise = noise
    for t in scheduler.timesteps:
        # 自动禁用梯度计算，从而在执行这段代码时不会为张量计算梯度
        with torch.no_grad():
            noisy_residual = model(input_noise, t).sample
        previous_noisy_sample = scheduler.step(noisy_residual, t, input_noise).prev_sample
        input_noise = previous_noisy_sample

    image = (input_noise / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    image = Image.fromarray((image * 255).round().astype("uint8"))
    image.save(f"../image/learn2/{generate_random_str()}.png")


if __name__ == '__main__':
    print(torch.cuda.is_available())
    deno_process()
