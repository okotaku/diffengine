# DiffEngine

[![build](https://github.com/okotaku/diffengine/actions/workflows/build.yml/badge.svg)](https://github.com/okotaku/diffengine/actions/workflows/build.yml)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://diffengine.readthedocs.io/en/latest/)
[![license](https://img.shields.io/github/license/okotaku/diffengine.svg)](https://github.com/okotaku/diffengine/blob/main/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/okotaku/diffengine.svg)](https://github.com/okotaku/diffengine/issues)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/okotaku/cfca8baf57938f193c212dcb43c02fac/diffengine-example.ipynb)
[![Linting: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

[📘 Documentation](https://diffengine.readthedocs.io/en/latest/) |
[🤔 Reporting Issues](https://github.com/okotaku/diffengine/issues/new/choose)

## 📄 Table of Contents

- [📖 Introduction](#-introduction-)
- [🛠️ Installation](#-installation-)
- [👨‍🏫 Get Started](#-get-started-)
- [🖋 Example Notebook](#-example-notebook-)
- [📘 Documentation](#-documentation-)
- [📊 Model Zoo](#-model-zoo-)
- [🙌 Contributing](#-contributing-)
- [🎫 License](#-license-)
- [🖊️ Citation](#-citation-)
- [💻 Sponsors](#-sponsors-)
- [🤝 Acknowledgement](#-acknowledgement-)

## 📖 Introduction [🔝](#-table-of-contents)

DiffEngine is the open-source toolbox for training state-of-the-art Diffusion Models. Packed with advanced features including diffusers and MMEngine, DiffEngine empowers both seasoned experts and newcomers in the field to efficiently create and enhance diffusion models. Stay at the forefront of innovation with our cutting-edge platform, accelerating your journey in Diffusion Models training.

1. **Training state-of-the-art Diffusion Models**: Empower your projects with state-of-the-art Diffusion Models. We can use Stable Diffusion, Stable Diffusion XL, DreamBooth, LoRA etc.
2. **Unified Config System and Module Designs**: Thanks to MMEngine, our platform boasts a unified configuration system and modular designs that streamline your workflow. Easily customize hyperparameters, loss functions, and other crucial settings while maintaining a structured and organized project environment.
3. **Inference with diffusers.pipeline**: Seamlessly transition from training to real-world application using the diffusers.pipeline module. Effortlessly deploy your trained Diffusion Models for inference tasks, enabling quick and efficient decision-making based on the insights derived from your models.

## 🛠️ Installation [🔝](#-table-of-contents)

Before installing DiffEngine, please ensure that PyTorch >= v2.0 has been successfully installed following the [official guide](https://pytorch.org/get-started/locally/).

Install DiffEngine

```
pip install openmim
pip install git+https://github.com/okotaku/diffengine.git
```

## 👨‍🏫 Get Started [🔝](#-table-of-contents)

DiffEngine makes training easy through its pre-defined configs. These configs provide a streamlined way to start your training process. Here's how you can get started using one of the pre-defined configs:

1. **Choose a config**: You can find various pre-defined configs in the [`configs`](configs/) directory of the DiffEngine repository. For example, if you wish to train a DreamBooth model using the Stable Diffusion algorithm, you can use the [`configs/stable_diffusion_dreambooth/stable_diffusion_v15_dreambooth_lora_dog.py`](configs/stable_diffusion_dreambooth/stable_diffusion_v15_dreambooth_lora_dog.py).

2. **Start Training**: Open a terminal and run the following command to start training with the selected config:

```bash
mim train diffengine stable_diffusion_v15_dreambooth_lora_dog.py
```

3. **Monitor Progress and get results**: The training process will begin, and you can track its progress. The outputs of the training will be located in the `work_dirs/stable_diffusion_v15_dreambooth_lora_dog` directory, specifically when using the `stable_diffusion_v15_dreambooth_lora_dog` config.

```
work_dirs/stable_diffusion_v15_dreambooth_lora_dog
├── 20230802_033741
|   ├── 20230802_033741.log  # log file
|   └── vis_data
|         ├── 20230802_033741.json  # log json file
|         ├── config.py  # config file for each experiment
|         └── vis_image  # visualized image from each step
├── step999
|   └── pytorch_lora_weights.bin  # weight for inferencing with diffusers.pipeline
├── iter_1000.pth  # checkpoint from each step
├── last_checkpoint  # last checkpoint, it can be used for resuming
└── stable_diffusion_v15_dreambooth_lora_dog.py  # latest config file
```

An illustrative output example is provided below:

![img](https://github.com/okotaku/diffengine/assets/24734142/e4576779-e05f-42d0-a709-d6481eea87a9)

4. **Inference with diffusers.pipeline**: Once you have trained a model, simply specify the path to the saved model and inference by the `diffusers.pipeline` module.

```py
import torch
from diffusers import DiffusionPipeline

checkpoint = 'work_dirs/stable_diffusion_v15_dreambooth_lora_dog/step999'
prompt = 'A photo of sks dog in a bucket'

pipe = DiffusionPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16)
pipe.to('cuda')
pipe.load_lora_weights(checkpoint)

image = pipe(
    prompt,
    num_inference_steps=50,
).images[0]
image.save('demo.png')
```

## 🖋 Example Notebook [🔝](#-table-of-contents)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/okotaku/cfca8baf57938f193c212dcb43c02fac/diffengine-example.ipynb)

For a more hands-on introduction to DiffEngine, you can run the Example Notebook on Colaboratory. This notebook demonstrates the process of training using SDV1.5 and SDV2.1 DreamBooth configurations.

## 📘 Documentation [🔝](#-table-of-contents)

For detailed user guides and advanced guides, please refer to our [Documentation](https://diffengine.readthedocs.io/en/latest/):

- [Get Started](https://diffengine.readthedocs.io/en/latest/get_started.html) for get started.

<details>
<summary>Run Guides</summary>

- [Run Stable Diffusion](https://diffengine.readthedocs.io/en/latest/run_guides/run_sd.html)
- [Run Stable Diffusion XL](https://diffengine.readthedocs.io/en/latest/run_guides/run_sdxl.html)
- [Run Stable Diffusion DreamBooth](https://diffengine.readthedocs.io/en/latest/run_guides/run_dreambooth.html)
- [Run Stable Diffusion XL DreamBooth](https://diffengine.readthedocs.io/en/latest/run_guides/run_dreambooth_xl.html)
- [Run Stable Diffusion LoRA](https://diffengine.readthedocs.io/en/latest/run_guides/run_lora.html)
- [Run Stable Diffusion XL LoRA](https://diffengine.readthedocs.io/en/latest/run_guides/run_lora_xl.html)
- [Run Stable Diffusion ControlNet](https://diffengine.readthedocs.io/en/latest/run_guides/run_controlnet.html)
- [Run Stable Diffusion XL ControlNet](https://diffengine.readthedocs.io/en/latest/run_guides/run_controlnet_xl.html)
- [Run IP Adapter](https://diffengine.readthedocs.io/en/latest/run_guides/run_ip_adapter.html)
- [Run T2I Adapter](https://diffengine.readthedocs.io/en/latest/run_guides/run_t2i_adapter.html)
- [Inference](https://diffengine.readthedocs.io/en/latest/run_guides/inference.html)

</details>

<details>
<summary>User Guides</summary>

- [Learn About Config](https://diffengine.readthedocs.io/en/latest/user_guides/config.html)
- [Prepare Dataset](https://diffengine.readthedocs.io/en/latest/user_guides/dataset_prepare.html)

</details>

<details>
<summary>Blog Posts</summary>

- [Introduction to DiffEngine](https://medium.com/@to78314910/introduction-to-diffengine-cad272e900c4)
- [Train ControlNet with DiffEngine](https://medium.com/@to78314910/train-controlnet-with-diffengine-727ef42bc38)
- [On Architectural Compression of Text-to-Image Diffusion Models](https://medium.com/@to78314910/on-architectural-compression-of-text-to-image-diffusion-models-ce8c9cba512a)

</details>

## 📊 Model Zoo [🔝](#-table-of-contents)

<details open>

<div align="center">
  <b>Supported algorithms</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Stable Diffusions</b>
      </td>
      <td>
        <b>Stable Diffusion XLs</b>
      </td>
      <td>
        <b>DeepFloyd IFs</b>
      </td>
      <td>
        <b>Others</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
            <li><a href="configs/stable_diffusion/README.md">Stable Diffusion (2022)</a></li>
            <li><a href="configs/stable_diffusion_controlnet/README.md">ControlNet (ICCV'2023)</a></li>
            <li><a href="configs/stable_diffusion_dreambooth/README.md">DreamBooth (CVPR'2023)</a></li>
            <li><a href="configs/stable_diffusion_lora/README.md">LoRA (ICLR'2022)</a></li>
            <li><a href="configs/distill_sd_dreambooth/README.md">Distill SD DreamBooth (2023)</a></li>
      </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/stable_diffusion_xl/README.md">Stable Diffusion XL (2023)</a></li>
          <li><a href="configs/stable_diffusion_xl_controlnet/README.md">ControlNet (ICCV'2023)</a></li>
          <li><a href="configs/stable_diffusion_xl_dreambooth/README.md">DreamBooth (CVPR'2023)</a></li>
          <li><a href="configs/stable_diffusion_xl_lora/README.md">LoRA (ICLR'2022)</a></li>
          <li><a href="configs/stable_diffusion_xl_controlnet_small/README.md">ControlNet Small (2023)</a></li>
          <li><a href="configs/t2i_adapter/README.md">T2I-Adapter (2023)</a></li>
          <li><a href="configs/ip_adapter/README.md">IP-Adapter (2023)</a></li>
          <li><a href="configs/esd/README.md">Erasing Concepts from Diffusion Models (2023)</a></li>
        </ul>
      </td>
      <td>
        <ul>
            <li><a href="configs/deepfloyd_if/README.md">DeepFloyd IF (2023)</a></li>
            <li><a href="configs/deepfloyd_if_dreambooth/README.md">DreamBooth (CVPR'2023)</a></li>
      </ul>
      </td>
      <td>
        <ul>
            <li><a href="configs/min_snr_loss/README.md">Min-SNR Loss (ICCV'2023)</a></li>
            <li><a href="configs/debias_estimation_loss/README.md">DeBias Estimation Loss (2023)</a></li>
      </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

</details>

## 🙌 Contributing [🔝](#-table-of-contents)

We appreciate all contributions to improve clshub. Please refer to [CONTRIBUTING.md](https://github.com/open-mmlab/mmpretrain/blob/main/CONTRIBUTING.md) for the contributing guideline.

## 🎫 License [🔝](#-table-of-contents)

This project is released under the [Apache 2.0 license](LICENSE).

## 🖊️ Citation [🔝](#-table-of-contents)

If DiffEngine is helpful to your research, please cite it as below.

```
@misc{diffengine2023,
    title = {{DiffEngine}: diffusers training toolbox with mmengine},
    author = {{DiffEngine Contributors}},
    howpublished = {\url{https://github.com/okotaku/diffengine}},
    year = {2023}
}
```

## 💻 Sponsors

[takuoko](https://github.com/okotaku) is a member of Z by HP Data Science Global Ambassadors. Special Thanks to Z by HP for sponsoring me a Z8G4 Workstation with dual A6000 GPU and a ZBook with RTX5000 GPU.

## 🤝 Acknowledgement [🔝](#-table-of-contents)

This repo borrows the architecture design and part of the code from [mmengine](https://github.com/open-mmlab/mmengine), [mmagic](https://github.com/open-mmlab/mmagic) and [diffusers](https://github.com/huggingface/diffusers).

Also, please check the following openmmlab and huggingface projects and the corresponding Documentation.

- [OpenMMLab](https://openmmlab.com/)
- [HuggingFace](https://huggingface.co/)
- [MIM](https://github.com/open-mmlab/mim): MIM Installs OpenMMLab Packages.

```
@article{mmengine2022,
  title   = {{MMEngine}: OpenMMLab Foundational Library for Training Deep Learning Models},
  author  = {MMEngine Contributors},
  howpublished = {\url{https://github.com/open-mmlab/mmengine}},
  year={2022}
}
```

```
@misc{mmagic2023,
    title = {{MMagic}: {OpenMMLab} Multimodal Advanced, Generative, and Intelligent Creation Toolbox},
    author = {{MMagic Contributors}},
    howpublished = {\url{https://github.com/open-mmlab/mmagic}},
    year = {2023}
}
```

```
@misc{von-platen-etal-2022-diffusers,
  author = {Patrick von Platen and Suraj Patil and Anton Lozhkov and Pedro Cuenca and Nathan Lambert and Kashif Rasul and Mishig Davaadorj and Thomas Wolf},
  title = {Diffusers: State-of-the-art diffusion models},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/diffusers}}
}
```
