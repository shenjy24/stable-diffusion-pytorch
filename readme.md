## 使用Pytorch进行AI绘图

### 系统要求

显卡需要支持 `CUDA`，提前安装好 [CUDA工具](https://developer.nvidia.com/cuda-toolkit-archive)，`Pytorch` 和 `CUDA` 的版本要求可以参考 https://pytorch.org/get-started/locally/ 。

### 配置环境

#### 1. 创建conda虚拟环境
```
conda create -n python3 python=3
conda activate python3
```

#### 2. 安装`PyTorch`依赖
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

#### 3. 安装`StableDiffusion`依赖

```
conda install diffusers=0.11.1
conda install transformers scipy ftfy accelerate
```

### 代码示例

```python
import torch
from diffusers import StableDiffusionPipeline


def stable_diffusion(prompt):
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    image = pipe(prompt).images[0]
    image.save(f"image/astronaut_rides_horse.png")
    image

    
if __name__ == '__main__':
    // 校验系统CUDA是否可用
    print(torch.cuda.is_available())
    stable_diffusion("a photograph of an astronaut riding a horse")
```

该示例会拉取`CompVis/stable-diffusion-v1-4`模型，然后根据 `prompt` 生成图片，保存到 `image` 目录下，图片名为 `astronaut_rides_horse.png` 。



### 参考文档

[colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb#scrollTo=yEErJFjlrSWS)

[CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)
