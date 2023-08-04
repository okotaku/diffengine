# DiffEngine

[![build](https://github.com/okotaku/diffengine/actions/workflows/build.yml/badge.svg)](https://github.com/okotaku/diffengine/actions/workflows/build.yml)
[![license](https://img.shields.io/github/license/okotaku/diffengine.svg)](https://github.com/okotaku/diffengine/blob/main/LICENSE)

## Table of Contents

- [Introduction](#introduction)
- [Get Started](#get-started)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Acknowledgement](#acknowledgement)

## Introduction

DiffEngine is an open source diffusers training toolbox with mmengine.

Documentation: [docs](docs)

## Get Started

3 commands quick run for DreamBooth Training Example.

```
pip install openmim
pip install git+https://github.com/okotaku/diffengine.git
mim train diffengine stable_diffusion_v15_dreambooth_lora_dog.py
```

Output example is:

![img](https://github.com/okotaku/diffengine/assets/24734142/e4576779-e05f-42d0-a709-d6481eea87a9)

You can also run [Example Notebook](examples/example-dreambooth.ipynb) quickly. It has an example of SDV1.5 & SDV2.1 DreamBooth Training.

Please refer to [get_started.md](docs/source/get_started.md) for get started.
Other tutorials for:

- [Run Stable Diffusion](docs/source/run_guides/run_sd.md)
- [Run Stable Diffusion XL](docs/source/run_guides/run_sdxl.md)
- [Run Stable Diffusion DreamBooth](docs/source/run_guides/run_dreambooth.md)
- [Run Stable Diffusion XL DreamBooth](docs/source/run_guides/run_dreambooth_xl.md)
- [Run Stable Diffusion LoRA](docs/source/run_guides/run_lora.md)
- [Run Stable Diffusion XL LoRA](docs/source/run_guides/run_lora_xl.md)
- [Inference](docs/source/run_guides/inference.md)
- [Learn About Config](docs/source/user_guides/config.md)
- [Prepare Dataset](docs/source/user_guides/dataset_prepare.md)

## Contributing

### CONTRIBUTING

We appreciate all contributions to improve clshub. Please refer to [CONTRIBUTING.md](https://github.com/open-mmlab/mmpretrain/blob/main/CONTRIBUTING.md) for the contributing guideline.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Citation

If DiffEngine is helpful to your research, please cite it as below.

```
@misc{diffengine2023,
    title = {{DiffEngine}: diffusers training toolbox with mmengine},
    author = {{DiffEngine Contributors}},
    howpublished = {\url{https://github.com/okotaku/diffengine}},
    year = {2023}
}
```

## Acknowledgement

This repo borrows the architecture design and part of the code from [mmengine](https://github.com/open-mmlab/mmengine), [mmagic](https://github.com/open-mmlab/mmagic) and [diffusers](https://github.com/huggingface/diffusers).

Also, please check the following openmmlab projects and the corresponding Documentation.

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
