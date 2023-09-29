from argparse import ArgumentParser

from mmengine import Config
from mmengine.registry import init_default_scope
from mmengine.runner.checkpoint import (_load_checkpoint,
                                        _load_checkpoint_to_model)
from PIL import Image

from diffengine.registry import MODELS

init_default_scope('diffengine')


def main():
    parser = ArgumentParser()
    parser.add_argument('prompt', help='Prompt text.')
    parser.add_argument('config', help='Path to config file.')
    parser.add_argument('checkpoint', help='Path to weight file.')
    parser.add_argument('--out', help='Output path', default='demo.png')
    parser.add_argument(
        '--height',
        help='The height for output images.',
        default=None,
        type=int)
    parser.add_argument(
        '--width', help='The width for output images.', default=None, type=int)
    parser.add_argument(
        '--example-image',
        help='Path to example image for generation.',
        type=str,
        default=None)
    parser.add_argument(
        '--device', help='Device used for inference', default='cuda')
    args = parser.parse_args()

    config = Config.fromfile(args.config).copy()

    StableDiffuser = MODELS.build(config.model)
    StableDiffuser = StableDiffuser.to(args.device)

    checkpoint = _load_checkpoint(args.checkpoint, map_location='cpu')
    _load_checkpoint_to_model(
        StableDiffuser, checkpoint['state_dict'], strict=False)

    kwargs = {}
    if args.example_image is not None:
        kwargs['example_image'] = [args.example_image]
    image = StableDiffuser.infer([args.prompt], **kwargs)[0]
    Image.fromarray(image).save(args.out)


if __name__ == '__main__':
    main()
