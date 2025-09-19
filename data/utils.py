import torch
from torchvision.transforms import v2
import random
from math import floor
from PIL import Image
import numpy as np
import cv2
import copy
from sn_calibration.src.evaluate_extremities import mirror_labels

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, annotation, metas):
        for transform in self.transforms:
            if transform is None:
                continue
            image, annotation, metas = transform(image, annotation, metas)
        return image, annotation, metas
    
class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, annotation, metas):
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
            image = torch.clamp(image, 0, 1)
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
        image = v2.functional.to_dtype(image, torch.float32, scale=True)
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


class ColorJitter:
    """Apply color jitter transformation with configurable parameters"""
    def __init__(self, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0, p=1.0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, image, annotation, metas):
        # Store color jitter parameters and apply decision for consistent application across frames
        if 'color_jitter_params' not in metas:
            # Make the apply decision once and store it
            metas['color_jitter_apply'] = random.random() <= self.p
            if not metas['color_jitter_apply']:
                return image, annotation, metas
            # Generate random parameters
            brightness_factor = None
            if self.brightness > 0:
                brightness_factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            
            contrast_factor = None
            if self.contrast > 0:
                contrast_factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            
            saturation_factor = None
            if self.saturation > 0:
                saturation_factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            
            hue_factor = None
            if self.hue > 0:
                hue_factor = random.uniform(-self.hue, self.hue)
            
            metas['color_jitter_params'] = {
                'brightness': brightness_factor,
                'contrast': contrast_factor,
                'saturation': saturation_factor,
                'hue': hue_factor
            }
        else:
            # Check if we should apply color jitter based on the stored decision
            if not metas['color_jitter_apply']:
                return image, annotation, metas
        
        # Apply color jitter using stored parameters
        params = metas['color_jitter_params']
        
        if isinstance(image, torch.Tensor):
            if params['brightness'] is not None:
                image = v2.functional.adjust_brightness(image, params['brightness'])
            if params['contrast'] is not None:
                image = v2.functional.adjust_contrast(image, params['contrast'])
            if params['saturation'] is not None:
                image = v2.functional.adjust_saturation(image, params['saturation'])
            if params['hue'] is not None:
                image = v2.functional.adjust_hue(image, params['hue'])
        elif isinstance(image, list):
            # Apply to all frames in the list
            for i in range(len(image)):
                if params['brightness'] is not None:
                    image[i] = v2.functional.adjust_brightness(image[i], params['brightness'])
                if params['contrast'] is not None:
                    image[i] = v2.functional.adjust_contrast(image[i], params['contrast'])
                if params['saturation'] is not None:
                    image[i] = v2.functional.adjust_saturation(image[i], params['saturation'])
                if params['hue'] is not None:
                    image[i] = v2.functional.adjust_hue(image[i], params['hue'])
        else:
            raise NotImplementedError(f"Color jitter not implemented for image type: {type(image)}")
        
        return image, annotation, metas


class RandomHorizontalFlip:
    """Apply random horizontal flip with annotation adjustment"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, annotation, metas):
        # Store flip decision for consistent application across frames
        if 'horizontal_flip' not in metas:
            metas['horizontal_flip'] = random.random() <= self.p
        
        if not metas['horizontal_flip']:
            return image, annotation, metas
        
        # Apply horizontal flip to image(s)
        if isinstance(image, torch.Tensor):
            image = v2.functional.hflip(image)
            image_width = image.shape[-1]
        elif isinstance(image, list):
            # Apply to all frames in the list
            for i in range(len(image)):
                image[i] = v2.functional.hflip(image[i])
            image_width = image[0].shape[-1]
        else:
            raise NotImplementedError(f"Horizontal flip not implemented for image type: {type(image)}")
        
        # Adjust annotations
        if "bbox" in annotation:
            # Flip bounding boxes (format: x, y, w, h)
            # bbox = annotation["bbox"].clone()
            bbox = annotation["bbox"]
            bbox[:, 0] = image_width - bbox[:, 0] - bbox[:, 2]  # x_new = width - x_old - w
            annotation["bbox"] = bbox
        
        if "lines" in annotation and len(annotation["lines"]) > 0:
            lines = annotation["lines"]
            for _, line in lines.items():
                for point in line:
                    point["x"] = 1 - point["x"]
            # lines = flip_annot_names(lines, swap_top_bottom=False, swap_posts=False)
            lines = correct_lines_labels_reverse(lines)
            lines = {swap_left_right_names(k): v for k, v in lines.items()}
            lines = correct_lines_labels(lines)
            annotation["lines"] = lines
        
        # Flip lines_target if it exists (horizontal flip along x-axis)
        if "lines_target" in annotation:
            lines_target = annotation["lines_target"]
            annotation["lines_target"] = torch.flip(lines_target, dims=[-1])
        
        # Flip keypoints_target if it exists (horizontal flip along x-axis)
        if "keypoints_target" in annotation:
            keypoints_target = annotation["keypoints_target"]
            annotation["keypoints_target"] = torch.flip(keypoints_target, dims=[-1])
        
        return image, annotation, metas


class GaussianNoise:
    """Add Gaussian noise to images"""
    def __init__(self, mean=0.0, std=0.02, p=1.0):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, image, annotation, metas):
        # Store noise parameters and apply decision for consistent application across frames
        if 'gaussian_noise_params' not in metas:
            metas['gaussian_noise_apply'] = random.random() <= self.p
            if not metas['gaussian_noise_apply']:
                return image, annotation, metas
            metas['gaussian_noise_params'] = {
                'mean': self.mean,
                'std': random.uniform(0, self.std)  # Random std up to max
            }
        else:
            # Check if we should apply gaussian noise based on the stored decision
            if not metas['gaussian_noise_apply']:
                return image, annotation, metas
        
        params = metas['gaussian_noise_params']
        
        if isinstance(image, torch.Tensor):
            noise = torch.randn_like(image) * params['std'] + params['mean']
            image = torch.clamp(image + noise, 0, 1) if image.max() <= 1 else torch.clamp(image + noise * 255, 0, 255)
        elif isinstance(image, list):
            # Apply to all frames in the list
            for i in range(len(image)):
                noise = torch.randn_like(image[i]) * params['std'] + params['mean']
                image[i] = torch.clamp(image[i] + noise, 0, 1) if image[i].max() <= 1 else torch.clamp(image[i] + noise * 255, 0, 255)
        else:
            raise NotImplementedError(f"Gaussian noise not implemented for image type: {type(image)}")
        
        return image, annotation, metas


class GaussianBlur:
    """Apply Gaussian blur to images"""
    def __init__(self, kernel_size_range=(3, 7), sigma_range=(0.1, 2.0), p=1.0):
        self.kernel_size_range = kernel_size_range
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, image, annotation, metas):
        # Store blur parameters and apply decision for consistent application across frames
        if 'gaussian_blur_params' not in metas:
            metas['gaussian_blur_apply'] = random.random() <= self.p
            if not metas['gaussian_blur_apply']:
                return image, annotation, metas
            kernel_size = random.randint(self.kernel_size_range[0], self.kernel_size_range[1])
            if kernel_size % 2 == 0:  # Ensure kernel size is odd
                kernel_size += 1
            sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
            metas['gaussian_blur_params'] = {
                'kernel_size': kernel_size,
                'sigma': sigma
            }
        else:
            # Check if we should apply gaussian blur based on the stored decision
            if not metas['gaussian_blur_apply']:
                return image, annotation, metas
        
        params = metas['gaussian_blur_params']
        
        if isinstance(image, torch.Tensor):
            image = v2.functional.gaussian_blur(image, [params['kernel_size'], params['kernel_size']], [params['sigma'], params['sigma']])
        elif isinstance(image, list):
            # Apply to all frames in the list
            for i in range(len(image)):
                image[i] = v2.functional.gaussian_blur(image[i], [params['kernel_size'], params['kernel_size']], [params['sigma'], params['sigma']])
        else:
            raise NotImplementedError(f"Gaussian blur not implemented for image type: {type(image)}")
        
        return image, annotation, metas


class ClearAugmentationMetas:
    """Clear augmentation metadata to ensure independence between samples"""
    def __call__(self, image, annotation, metas):
        # Remove augmentation-specific metadata
        keys_to_remove = ['color_jitter_params', 'color_jitter_apply', 'horizontal_flip', 'gaussian_noise_params', 'gaussian_noise_apply', 'gaussian_blur_params', 'gaussian_blur_apply']
        for key in keys_to_remove:
            if key in metas:
                del metas[key]
        return image, annotation, metas
    
FLIP_POSTS = {
    'Goal left post right': 'Goal left post left ',
    'Goal left post left ': 'Goal left post right',
    'Goal right post right': 'Goal right post left',
    'Goal right post left': 'Goal right post right'
}

h_lines = ['Goal left crossbar', 'Side line left', 'Small rect. left main', 'Big rect. left main', 'Middle line',
                   'Big rect. right main', 'Small rect. right main', 'Side line right', 'Goal right crossbar']

v_lines = ['Side line top', 'Big rect. left top', 'Small rect. left top', 'Small rect. left bottom',
                   'Big rect. left bottom', 'Big rect. right top', 'Small rect. right top', 'Small rect. right bottom',
                              'Big rect. right bottom', 'Side line bottom']

def swap_top_bottom_names(line_name: str) -> str:
    x: str = 'top'
    y: str = 'bottom'
    if x in line_name or y in line_name:
        return y.join(part.replace(y, x) for part in line_name.split(x))
    return line_name

def swap_left_right_names(line_name: str) -> str:
    x: str = 'left'
    y: str = 'right'
    if x in line_name or y in line_name:
        return y.join(part.replace(y, x) for part in line_name.split(x))
    return line_name

def swap_posts_names(line_name: str) -> str:
    if line_name in FLIP_POSTS:
        return FLIP_POSTS[line_name]
    return line_name

def flip_annot_names(annot, swap_top_bottom: bool = True, swap_posts: bool = True):
    annot = mirror_labels(annot)
    if swap_top_bottom:
        annot = {swap_top_bottom_names(k): v for k, v in annot.items()}
    if swap_posts:
        annot = {swap_posts_names(k): v for k, v in annot.items()}
    return annot

def correct_lines_labels(data):
    if 'Goal left post left' in data.keys():
        data['Goal left post left '] = copy.deepcopy(data['Goal left post left'])
        del data['Goal left post left']

    return data

def correct_lines_labels_reverse(data):
    if 'Goal left post left ' in data.keys():
        data['Goal left post left'] = copy.deepcopy(data['Goal left post left '])
        del data['Goal left post left ']

    return data