import torch
from torchvision.transforms import v2
import random
from math import floor
from PIL import Image

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, annotation, metas):
        for transform in self.transforms:
            image, annotation, metas = transform(image, annotation, metas)
        return image, annotation, metas
    
class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, annotation, metas):
        image = image.to(torch.float32).div(255)
        image = v2.functional.normalize(image, mean=self.mean, std=self.std)
        if "bbox" in annotation:
            h, w = image.shape[-2:]
            annotation["bbox"] = annotation["bbox"] / torch.tensor([w, h, w, h])
        return image, annotation, metas
    
class RandomResize:
    def __init__(self, sizes: list, max_size: int | None = None, keep_aspect_ratio: bool = True):
        self.sizes = sizes
        self.max_size = max_size
        self.keep_aspect_ratio = keep_aspect_ratio

    def __call__(self, image, annotation, metas):
        new_size = random.choice(self.sizes)  # choose the size for images

        def get_new_hw(_curr_hw: list, _new_size) -> tuple[int, int]:
            _curr_h, _curr_w = _curr_hw
            if self.keep_aspect_ratio:
                if self.max_size is not None:  # need to restrict the longer side length
                    _min_hw, _max_hw = float(min(_curr_h, _curr_w)), float(max(_curr_h, _curr_w))
                    if _max_hw / _min_hw * _new_size > self.max_size:  # need to restrict the resize size
                        _new_size = int(floor(self.max_size * _min_hw / _max_hw))
                # Calculate the new height and width while maintaining aspect ratio:
                if _curr_w < _curr_h:
                    _new_w = _new_size
                    _new_h = int(round(_new_size * _curr_h / _curr_w))
                else:
                    _new_h = _new_size
                    _new_w = int(round(_new_size * _curr_w / _curr_h))
                return _new_h, _new_w
            else:
                # When not keeping aspect ratio, just use the same size for both dimensions
                return _new_size, _new_size

        new_hw = get_new_hw(get_image_hw(image), _new_size=new_size)    # new yx
        scale_ratio_x = new_hw[1] / get_image_hw(image)[1]
        scale_ratio_y = new_hw[0] / get_image_hw(image)[0]
        
        metas['original_image_size'] = get_image_hw(image)
        metas['image_size'] = new_hw
        metas['scale_ratio_x'] = scale_ratio_x
        metas['scale_ratio_y'] = scale_ratio_y
        
        # Resize images:
        if isinstance(image, torch.Tensor):
            image = v2.functional.resize(image, new_hw, interpolation=v2.InterpolationMode.BICUBIC)
        else:
            raise NotImplementedError(f"The input image type {type(image)} is not supported.")
        # Resize annotations:
        if "bbox" in annotation:
            annotation["bbox"] = annotation["bbox"] * torch.as_tensor([scale_ratio_x, scale_ratio_y] * 2)
        if "intrinsic" in annotation and annotation["valid_camera"]:
            annotation["intrinsic"][0, :] = annotation["intrinsic"][0, :] * scale_ratio_x
            annotation["intrinsic"][1, :] = annotation["intrinsic"][1, :] * scale_ratio_y
            
        return image, annotation, metas    

class ToTensor:
    def __init__(self):
        return

    def __call__(self, image, annotation, metas):
        image = v2.functional.to_image(image)
        return image, annotation, metas
    
def get_image_hw(image: torch.Tensor | list | Image.Image):
    if isinstance(image, torch.Tensor):
        return image.shape[-2], image.shape[-1]
    elif isinstance(image, list):
        return get_image_hw(image[0])
    elif isinstance(image, Image.Image):
        return image.height, image.width
    else:
        raise NotImplementedError("The input image type is not supported.")