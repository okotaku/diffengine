#### Single GPU

Environment:

- A6000 Single GPU
- nvcr.io/nvidia/pytorch:23.10-py3

Settings:

- 1epoch training.

|                    Model                     | total time |
| :------------------------------------------: | :--------: |
| stable_diffusion_xl_pokemon_blip_fast (fp16) |  9 m 47 s  |
| stable_diffusion_xl_pokemon_blip_dali (bf16) |  9 m 44 s  |
