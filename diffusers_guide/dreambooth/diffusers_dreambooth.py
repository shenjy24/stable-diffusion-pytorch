import numpy as np
import torch
import torchvision.utils
from PIL import Image
from datasets import load_dataset
from diffusers import StableDiffusionPipeline
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def show_images(x):
    """给定一批图像，创建一个网格并将其转换为PIL"""
    x = x * 0.5 + 0.5
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im


def make_grid(images, size=64):
    """给定一个PIL图像列表，将它们叠加成一行以便查看"""
    output_im = Image.new("RGB", (size * len(images), size))
    for i, im in enumerate(images):
        output_im.paste(im.resize((size, size)), (i * size, 0))
    return output_im


def load_diffuser_dataset():
    dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")
    image_size = 32
    batch_size = 64
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),  # 调整大小
            transforms.RandomHorizontalFlip(),  # 随机翻转
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
    )

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}

    dataset.set_transform(transform)
    # 创建数据加载器
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    xb = next(iter(train_dataloader))["images"].to(device)[:8]
    print("X shape:", xb.shape)
    show_images(xb).resize((8 * 64, 64), resample=Image.NEAREST)


def dreambooth_library():
    model_id = "sd-dreambooth-library/mr-potato-head"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

    prompt = "an abstract oil painting of sks mr potato head by picasso"
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    image.save(f"../../image/main/dreambooth.png")


if __name__ == '__main__':
    load_diffuser_dataset()
