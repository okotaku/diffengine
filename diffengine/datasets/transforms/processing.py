# flake8: noqa: S311
import html
import inspect
import random
import re
import urllib.parse as ul
from collections.abc import Sequence
from enum import EnumMeta

import numpy as np
import torch
import torchvision
from diffusers.utils import is_bs4_available, is_ftfy_available
from mmengine.dataset.base_dataset import Compose
from torchvision.transforms.functional import crop
from torchvision.transforms.transforms import InterpolationMode
from transformers import CLIPImageProcessor as HFCLIPImageProcessor

from diffengine.datasets.transforms.base import BaseTransform
from diffengine.registry import TRANSFORMS

if is_bs4_available():
    from bs4 import BeautifulSoup

if is_ftfy_available():
    import ftfy


def _str_to_torch_dtype(t: str):
    """Map to torch.dtype."""
    import torch  # noqa: F401
    return eval(f"torch.{t}")  # noqa


def _interpolation_modes_from_str(t: str):
    """Map to Interpolation."""
    t = t.lower()
    inverse_modes_mapping = {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "box": InterpolationMode.BOX,
        "hammimg": InterpolationMode.HAMMING,
        "lanczos": InterpolationMode.LANCZOS,
    }
    return inverse_modes_mapping[t]


class TorchVisonTransformWrapper:
    """TorchVisonTransformWrapper.

    We can use torchvision.transforms like `dict(type='torchvision/Resize',
    size=512)`

    Args:
    ----
        transform (str): The name of transform. For example
            `torchvision/Resize`.
        keys (List[str]): `keys` to apply augmentation from results.
    """

    def __init__(self,
                 transform,
                 *args,
                 keys: list[str] | None = None,
                 **kwargs) -> None:
        if keys is None:
            keys = ["img"]
        self.keys = keys
        if "interpolation" in kwargs and isinstance(kwargs["interpolation"],
                                                    str):
            kwargs["interpolation"] = _interpolation_modes_from_str(
                kwargs["interpolation"])
        if "dtype" in kwargs and isinstance(kwargs["dtype"], str):
            kwargs["dtype"] = _str_to_torch_dtype(kwargs["dtype"])
        self.t = transform(*args, **kwargs)

    def __call__(self, results) -> dict:
        """Call transform."""
        for k in self.keys:
            results[k] = self.t(results[k])
        return results

    def __repr__(self) -> str:
        """Repr."""
        return f"TorchVision{self.t!r}"


def register_vision_transforms() -> list[str]:
    """Register vision transforms.

    Register transforms in ``torchvision.transforms`` to the ``TRANSFORMS``
    registry.

    Returns
    -------
        List[str]: A list of registered transforms' name.
    """
    vision_transforms = []
    for module_name in dir(torchvision.transforms):
        if not re.match("[A-Z]", module_name):
            # must startswith a capital letter
            continue
        _transform = getattr(torchvision.transforms, module_name)
        if inspect.isclass(_transform) and callable(
                _transform) and not isinstance(_transform, (EnumMeta)):
            from functools import partial
            TRANSFORMS.register_module(
                module=partial(
                    TorchVisonTransformWrapper, transform=_transform),
                name=f"torchvision/{module_name}")
            vision_transforms.append(f"torchvision/{module_name}")
    return vision_transforms


# register all the transforms in torchvision by using a transform wrapper
VISION_TRANSFORMS = register_vision_transforms()


@TRANSFORMS.register_module()
class SaveImageShape(BaseTransform):
    """Save image shape as 'ori_img_shape' in results."""

    def transform(self, results: dict) -> dict | tuple[list, list] | None:
        """Transform.

        Args:
        ----
            results (dict): The result dict.

        Returns:
        -------
            dict: 'ori_img_shape' key is added as original image shape.
        """
        results["ori_img_shape"] = [
            results["img"].height,
            results["img"].width,
        ]
        return results


@TRANSFORMS.register_module()
class RandomCrop(BaseTransform):
    """RandomCrop.

    The difference from torchvision/RandomCrop is
        1. save crop top left as 'crop_top_left' and `crop_bottom_right` in
        results
        2. apply same random parameters to multiple `keys` like ['img',
        'condition_img'].

    Args:
    ----
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted
            as (size[0], size[0])
        keys (List[str]): `keys` to apply augmentation from results.
        force_same_size (bool): Force same size for all keys. Defaults to True.
    """

    def __init__(self,
                 *args,
                 size: Sequence[int] | int,
                 keys: list[str] | None = None,
                 force_same_size: bool = True,
                 **kwargs) -> None:
        if keys is None:
            keys = ["img"]
        if not isinstance(size, Sequence):
            size = (size, size)
        self.size = size
        self.keys = keys
        self.force_same_size = force_same_size
        self.pipeline = torchvision.transforms.RandomCrop(
            *args, size, **kwargs)

    def transform(self, results: dict) -> dict | tuple[list, list] | None:
        """Transform.

        Args:
        ----
            results (dict): The result dict.

        Returns:
        -------
            dict: 'crop_top_left' and  `crop_bottom_right` key is added as crop
                point.
        """
        if self.force_same_size:
            assert all(
                results["img"].size == results[k].size for k in self.keys), (
                "Size mismatch. {k: results[k].size for k in self.keys}"
            )
        y1, x1, h, w = self.pipeline.get_params(results["img"], self.size)
        for k in self.keys:
            results[k] = crop(results[k], y1, x1, h, w)
        results["crop_top_left"] = [y1, x1]
        results["crop_bottom_right"] = [y1 + h, x1 + w]
        return results


@TRANSFORMS.register_module()
class CenterCrop(BaseTransform):
    """CenterCrop.

    The difference from torchvision/CenterCrop is
        1. save crop top left as 'crop_top_left' and `crop_bottom_right` in
        results

    Args:
    ----
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted
            as (size[0], size[0])
        keys (List[str]): `keys` to apply augmentation from results.
    """

    def __init__(self,
                 *args,
                 size: Sequence[int] | int,
                 keys: list[str] | None = None,
                 **kwargs) -> None:
        if keys is None:
            keys = ["img"]
        if not isinstance(size, Sequence):
            size = (size, size)
        self.size = size
        self.keys = keys
        self.pipeline = torchvision.transforms.CenterCrop(
            *args, size, **kwargs)

    def transform(self, results: dict) -> dict | tuple[list, list] | None:
        """Transform.

        Args:
        ----
            results (dict): The result dict.

        Returns:
        -------
            dict: 'crop_top_left' key is added as crop points.
        """
        assert all(results["img"].size == results[k].size for k in self.keys)
        y1 = max(0, int(round((results["img"].height - self.size[0]) / 2.0)))
        x1 = max(0, int(round((results["img"].width - self.size[1]) / 2.0)))
        y2 = max(0, int(round((results["img"].height + self.size[0]) / 2.0)))
        x2 = max(0, int(round((results["img"].width + self.size[1]) / 2.0)))
        for k in self.keys:
            results[k] = self.pipeline(results[k])
        results["crop_top_left"] = [y1, x1]
        results["crop_bottom_right"] = [y2, x2]
        return results


@TRANSFORMS.register_module()
class MultiAspectRatioResizeCenterCrop(BaseTransform):
    """Multi Aspect Ratio Resize and Center Crop.

    Args:
    ----
        sizes (List[sequence]): List of desired output size of the crop.
            Sequence like (h, w).
        keys (List[str]): `keys` to apply augmentation from results.
        interpolation (str): Desired interpolation enum defined by
            torchvision.transforms.InterpolationMode.
            Defaults to 'bilinear'.
    """

    def __init__(
            self,
            *args,  # noqa
            sizes: list[Sequence[int]],
            keys: list[str] | None = None,
            interpolation: str = "bilinear",
            **kwargs) -> None:  # noqa
        if keys is None:
            keys = ["img"]
        self.sizes = sizes
        self.aspect_ratios = np.array([s[0] / s[1] for s in sizes])
        self.pipelines = []
        for s in self.sizes:
            self.pipelines.append(
                Compose([
                    TorchVisonTransformWrapper(
                        torchvision.transforms.Resize,
                        size=min(s),
                        interpolation=interpolation,
                        keys=keys),
                    CenterCrop(size=s, keys=keys),
                ]))

    def transform(self, results: dict) -> dict | tuple[list, list] | None:
        """Transform.

        Args:
        ----
            results (dict): The result dict.
        """
        aspect_ratio = results["img"].height / results["img"].width
        bucked_id = np.argmin(np.abs(aspect_ratio - self.aspect_ratios))
        return self.pipelines[bucked_id](results)


@TRANSFORMS.register_module()
class RandomHorizontalFlip(BaseTransform):
    """RandomHorizontalFlip.

    The difference from torchvision/RandomHorizontalFlip is
        1. update 'crop_top_left' and `crop_bottom_right` if exists.
        2. apply same random parameters to multiple `keys` like ['img',
        'condition_img'].

    Args:
    ----
        p (float): probability of the image being flipped.
            Default value is 0.5.
        keys (List[str]): `keys` to apply augmentation from results.
    """

    def __init__(self, *args, p: float = 0.5, keys=None, **kwargs) -> None:
        if keys is None:
            keys = ["img"]
        self.p = p
        self.keys = keys
        self.pipeline = torchvision.transforms.RandomHorizontalFlip(
            *args, p=1.0, **kwargs)

    def transform(self, results: dict) -> dict | tuple[list, list] | None:
        """Transform.

        Args:
        ----
            results (dict): The result dict.

        Returns:
        -------
            dict: 'crop_top_left' key is fixed.
        """
        if random.random() < self.p:
            assert all(results["img"].size == results[k].size
                       for k in self.keys)
            for k in self.keys:
                results[k] = self.pipeline(results[k])
            if "crop_top_left" in results:
                y1 = results["crop_top_left"][0]
                x1 = results["img"].width - results["crop_bottom_right"][1]
                results["crop_top_left"] = [y1, x1]
        return results


@TRANSFORMS.register_module()
class ComputeTimeIds(BaseTransform):
    """Compute time ids as 'time_ids' in results."""

    def transform(self, results: dict) -> dict | tuple[list, list] | None:
        """Transform.

        Args:
        ----
            results (dict): The result dict.

        Returns:
        -------
            dict: 'time_ids' key is added as original image shape.
        """
        assert "ori_img_shape" in results
        assert "crop_top_left" in results
        target_size = [results["img"].height, results["img"].width]
        time_ids = results["ori_img_shape"] + results[
            "crop_top_left"] + target_size
        results["time_ids"] = time_ids
        return results


@TRANSFORMS.register_module()
class ComputePixArtImgInfo(BaseTransform):
    """Compute Orig Height and Widh + Aspect Ratio.

    Return 'resolution', 'aspect_ratio' in results
    """

    def transform(self, results: dict) -> dict | tuple[list, list] | None:
        """Transform.

        Args:
        ----
            results (dict): The result dict.

        Returns:
        -------
            dict: 'time_ids' key is added as original image shape.
        """
        assert "ori_img_shape" in results
        results["resolution"] = [float(s) for s in results["ori_img_shape"]]
        results["aspect_ratio"] = results["img"].height / results["img"].width
        return results


@TRANSFORMS.register_module()
class CLIPImageProcessor(BaseTransform):
    """CLIPImageProcessor.

    Args:
    ----
        key (str): `key` to apply augmentation from results. Defaults to 'img'.
        output_key (str): `output_key` after applying augmentation from
            results. Defaults to 'clip_img'.
    """

    def __init__(self, key: str = "img", output_key: str = "clip_img",
                 pretrained: str | None = None) -> None:
        self.key = key
        self.output_key = output_key
        if pretrained is None:
            self.pipeline = HFCLIPImageProcessor()
        else:
            self.pipeline = HFCLIPImageProcessor.from_pretrained(
                pretrained, subfolder="image_processor")

    def transform(self, results: dict) -> dict | tuple[list, list] | None:
        """Transform.

        Args:
        ----
            results (dict): The result dict.
        """
        # (1, 3, 224, 224) -> (3, 224, 224)
        results[self.output_key] = self.pipeline(
            images=results[self.key], return_tensors="pt").pixel_values[0]
        return results


@TRANSFORMS.register_module()
class RandomTextDrop(BaseTransform):
    """RandomTextDrop. Replace text to empty.

    Args:
    ----
        p (float): probability of the image being flipped.
            Default value is 0.5.
        keys (List[str]): `keys` to apply augmentation from results.
    """

    def __init__(self, p: float = 0.1, keys=None) -> None:
        if keys is None:
            keys = ["text"]
        self.p = p
        self.keys = keys

    def transform(self, results: dict) -> dict | tuple[list, list] | None:
        """Transform.

        Args:
        ----
            results (dict): The result dict.
        """
        if random.random() < self.p:
            for k in self.keys:
                results[k] = ""
        return results



@TRANSFORMS.register_module()
class T5TextPreprocess(BaseTransform):
    """T5 Text Preprocess.

    Args:
    ----
        keys (List[str]): `keys` to apply augmentation from results.
        clean_caption (bool): clean caption. Defaults to False.
    """

    def __init__(self, keys=None,
                 *,
                 clean_caption: bool = True) -> None:
        if clean_caption:
            assert is_ftfy_available(), "Please install ftfy."
            assert is_bs4_available(), "Please install bs4."

        if keys is None:
            keys = ["text"]
        self.keys = keys
        self.clean_caption = clean_caption
        self.bad_punct_regex = re.compile(
            r"["  # noqa
            + "#®•©™&@·º½¾¿¡§~"
            + r"\)"
            + r"\("
            + r"\]"
            + r"\["
            + r"\}"
            + r"\{"
            + r"\|"
            + "\\"
            + r"\/"
            + r"\*"
            + r"]{1,}",
        )

    def _clean_caption(self, caption: str) -> str:  # noqa
        """Clean caption.

        Copied from
        diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._clean_caption
        """
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub("<person>", "person", caption)
        # urls:
        caption = re.sub(
            r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",
            "",
            caption,
        )  # regex for urls
        caption = re.sub(
            r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",
            "",
            caption,
        )  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features="html.parser").text

        # @<nickname>
        caption = re.sub(r"@[\w\d]+\b", "", caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
        caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
        caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
        caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
        caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
        caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
        caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
        #######################################################

        # все виды тире / all types of dash --> "-"
        caption = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",
            "-",
            caption,
        )

        # кавычки к одному стандарту
        caption = re.sub(r"[`´«»“”¨]", '"', caption)  # noqa
        caption = re.sub(r"[‘’]", "'", caption)  # noqa

        # &quot;
        caption = re.sub(r"&quot;?", "", caption)
        # &amp
        caption = re.sub(r"&amp", "", caption)

        # ip addresses:
        caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

        # article ids:
        caption = re.sub(r"\d:\d\d\s+$", "", caption)

        # \n
        caption = re.sub(r"\\n", " ", caption)

        # "#123"
        caption = re.sub(r"#\d{1,3}\b", "", caption)
        # "#12345.."
        caption = re.sub(r"#\d{5,}\b", "", caption)
        # "123456.."
        caption = re.sub(r"\b\d{6,}\b", "", caption)
        # filenames:
        caption = re.sub(
            r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption)

        #
        caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

        # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(self.bad_punct_regex, r" ", caption)
        caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, caption)) > 3:  # noqa
            caption = re.sub(regex2, " ", caption)

        caption = ftfy.fix_text(caption)
        caption = html.unescape(html.unescape(caption))

        caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
        caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
        caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

        caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
        caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
        caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
        caption = re.sub(
            r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "",
            caption)
        caption = re.sub(r"\bpage\s+\d+\b", "", caption)

        # j2d1a2a...
        caption = re.sub(
            r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption)

        caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)  # noqa

        caption = re.sub(r"\b\s+\:\s+", r": ", caption)
        caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
        caption = re.sub(r"\s+", " ", caption)

        caption.strip()

        caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
        caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
        caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
        caption = re.sub(r"^\.\S+$", "", caption)

        return caption.strip()

    def transform(self, results: dict) -> dict | tuple[list, list] | None:
        """Transform.

        Args:
        ----
            results (dict): The result dict.
        """
        for k in self.keys:
            text = results[k]
            if self.clean_caption:
                text = self._clean_caption(text)
            else:
                text = text.lower().strip()
            results[k] = text
        return results


@TRANSFORMS.register_module()
class MaskToTensor(BaseTransform):
    """MaskToTensor.

    1. Convert mask to tensor.
    2. Transpose mask from (H, W, 1) to (1, H, W)

    Args:
    ----
        key (str): `key` to apply augmentation from results.
            Defaults to 'mask'.
    """

    def __init__(self, key: str = "mask") -> None:
        self.key = key

    def transform(self, results: dict) -> dict | tuple[list, list] | None:
        """Transform.

        Args:
        ----
            results (dict): The result dict.
        """
        # (1, 3, 224, 224) -> (3, 224, 224)
        results[self.key] = torch.Tensor(results[self.key]).permute(2, 0, 1)
        return results


@TRANSFORMS.register_module()
class GetMaskedImage(BaseTransform):
    """GetMaskedImage.

    Args:
    ----
        key (str): `key` to outputs.
            Defaults to 'masked_image'.
    """

    def __init__(self, key: str = "masked_image") -> None:
        self.key = key

    def transform(self, results: dict) -> dict | tuple[list, list] | None:
        """Transform.

        Args:
        ----
            results (dict): The result dict.
        """
        mask_threahold = 0.5
        results[self.key] = results["img"] * (results["mask"] < mask_threahold)
        return results


@TRANSFORMS.register_module()
class AddConstantCaption(BaseTransform):
    """AddConstantCaption.

    Example. "a dog." * constant_caption="in szn style"
        -> "a dog. in szn style"

    Args:
    ----
        keys (List[str]): `keys` to apply augmentation from results.
    """

    def __init__(self, constant_caption: str, keys=None) -> None:
        if keys is None:
            keys = ["text"]
        self.constant_caption: str = constant_caption
        self.keys = keys

    def transform(self,
                  results: dict) -> dict | tuple[list, list] | None:
        """Transform.

        Args:
        ----
            results (dict): The result dict.
        """
        for k in self.keys:
            results[k] = results[k] + " " + self.constant_caption
        return results
