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


class RandomAffine:
    """Apply random affine transformation with annotation adjustment"""
    def __init__(self, degrees=0, translate=None, scale=None, shear=None, p=1.0):
        """
        Args:
            degrees: Range of degrees to select from for rotation. If degrees is a number
                    instead of sequence like (min, max), the range of degrees will be (-degrees, +degrees)
            translate: Tuple of maximum absolute fraction for horizontal and vertical translations
                      For example translate=(a, b), then horizontal shift is randomly sampled 
                      in the range -img_width * a < dx < img_width * a and vertical shift is 
                      randomly sampled in the range -img_height * b < dy < img_height * b
            scale: Scaling factor interval, e.g (a, b), then scale is randomly sampled from the range a <= scale <= b
            shear: Range of degrees to select from for shearing. If shear is a number,
                   a shear parallel to the x axis in the range (-shear, +shear) will be applied
            p: Probability of applying affine transformation
        """
        self.degrees = degrees if isinstance(degrees, (tuple, list)) else (-degrees, degrees)
        self.translate = translate
        self.scale = scale
        self.shear = shear if isinstance(shear, (tuple, list)) else (-shear, shear) if shear is not None else None
        self.p = p

    def __call__(self, image, annotation, metas):
        # Store affine parameters for consistent application across frames
        if 'random_affine_params' not in metas:
            metas['random_affine_apply'] = random.random() <= self.p
            if not metas['random_affine_apply']:
                return image, annotation, metas
                
            # Get original image dimensions
            if isinstance(image, torch.Tensor):
                orig_h, orig_w = image.shape[-2], image.shape[-1]
            elif isinstance(image, list):
                orig_h, orig_w = image[0].shape[-2], image[0].shape[-1]
            else:
                raise NotImplementedError(f"Random affine not implemented for image type: {type(image)}")
            
            # Generate random affine parameters
            angle = random.uniform(self.degrees[0], self.degrees[1])
            
            translate_x = 0
            translate_y = 0
            if self.translate is not None:
                max_dx = self.translate[0] * orig_w
                max_dy = self.translate[1] * orig_h
                translate_x = random.uniform(-max_dx, max_dx)
                translate_y = random.uniform(-max_dy, max_dy)
            
            scale_factor = 1.0
            if self.scale is not None:
                scale_factor = random.uniform(self.scale[0], self.scale[1])
            
            shear_x = 0
            if self.shear is not None:
                shear_x = random.uniform(self.shear[0], self.shear[1])
            
            metas['random_affine_params'] = {
                'angle': angle,
                'translate': (translate_x, translate_y),
                'scale': scale_factor,
                'shear': shear_x,
                'orig_w': orig_w,
                'orig_h': orig_h
            }
        else:
            # Check if we should apply random affine based on the stored decision
            if not metas['random_affine_apply']:
                return image, annotation, metas
        
        params = metas['random_affine_params']
        angle = params['angle']
        translate_x, translate_y = params['translate']
        scale_factor = params['scale']
        shear_x = params['shear']
        orig_w, orig_h = params['orig_w'], params['orig_h']
        
        # Apply affine transformation to image(s)
        if isinstance(image, torch.Tensor):
            image = v2.functional.affine(
                image, 
                angle=angle, 
                translate=[translate_x, translate_y], 
                scale=scale_factor, 
                shear=[shear_x, 0]
            )
        elif isinstance(image, list):
            # Apply to all frames in the list
            for i in range(len(image)):
                image[i] = v2.functional.affine(
                    image[i], 
                    angle=angle, 
                    translate=[translate_x, translate_y], 
                    scale=scale_factor, 
                    shear=[shear_x, 0]
                )
        
        # Compute affine transformation matrix
        # Center of rotation is the center of the image
        center_x, center_y = orig_w / 2, orig_h / 2
        
        # Convert angle to radians
        angle_rad = np.radians(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Convert shear to radians
        shear_rad = np.radians(shear_x)
        shear_tan = np.tan(shear_rad)
        
        # Create transformation matrix (from destination to source coordinates)
        # This matches PyTorch's convention
        # First apply shear, then scale, then rotation, then translation
        
        # Combined transformation matrix
        a = scale_factor * cos_a
        b = scale_factor * (-sin_a + shear_tan * cos_a)
        c = scale_factor * sin_a
        d = scale_factor * (cos_a + shear_tan * sin_a)
        
        # Translation part (includes centering)
        tx = -a * center_x - b * center_y + center_x + translate_x
        ty = -c * center_x - d * center_y + center_y + translate_y
        
        # Store transformation matrix for coordinate transformation
        transform_matrix = np.array([[a, b, tx], [c, d, ty], [0, 0, 1]])
        
        # Adjust bounding boxes
        if "bbox" in annotation and len(annotation["bbox"]) > 0:
            bbox = annotation["bbox"].clone()
            # bbox format: [x, y, w, h]
            
            new_bboxes = []
            valid_indices = []
            
            for i, box in enumerate(bbox):
                x, y, w, h = box
                
                # Get the four corners of the bounding box
                corners = np.array([
                    [x, y, 1],          # top-left
                    [x + w, y, 1],      # top-right  
                    [x + w, y + h, 1],  # bottom-right
                    [x, y + h, 1]       # bottom-left
                ]).T
                
                # Transform corners
                transformed_corners = transform_matrix @ corners
                transformed_corners = transformed_corners[:2, :]  # Remove homogeneous coordinate
                
                # Get new bounding box from transformed corners
                min_x = np.min(transformed_corners[0, :])
                max_x = np.max(transformed_corners[0, :])
                min_y = np.min(transformed_corners[1, :])
                max_y = np.max(transformed_corners[1, :])
                
                new_w = max_x - min_x
                new_h = max_y - min_y
                
                # Check if the transformed box is still valid (intersects with image)
                if (max_x > 0 and max_y > 0 and min_x < orig_w and min_y < orig_h and 
                    new_w > 0 and new_h > 0):
                    # Clip to image boundaries
                    clipped_min_x = max(0, min_x)
                    clipped_min_y = max(0, min_y)
                    clipped_max_x = min(orig_w, max_x)
                    clipped_max_y = min(orig_h, max_y)
                    
                    clipped_w = clipped_max_x - clipped_min_x
                    clipped_h = clipped_max_y - clipped_min_y
                    
                    if clipped_w > 0 and clipped_h > 0:
                        new_bboxes.append([clipped_min_x, clipped_min_y, clipped_w, clipped_h])
                        valid_indices.append(i)
            
            if len(new_bboxes) > 0:
                annotation["bbox"] = torch.tensor(new_bboxes, dtype=torch.float32)
                valid_mask = torch.tensor(valid_indices, dtype=torch.long)
                
                # Filter all related annotations
                if "id" in annotation:
                    annotation["id"] = annotation["id"][valid_mask]
                if "category" in annotation:
                    annotation["category"] = annotation["category"][valid_mask]
                if "visibility" in annotation:
                    annotation["visibility"] = annotation["visibility"][valid_mask]
                if "role" in annotation:
                    annotation["role"] = annotation["role"][valid_mask]
                if "jersey" in annotation:
                    annotation["jersey"] = annotation["jersey"][valid_mask]
                if "digit_head" in annotation:
                    annotation["digit_head"] = annotation["digit_head"][valid_mask]
                if "digit_tail" in annotation:
                    annotation["digit_tail"] = annotation["digit_tail"][valid_mask]
                if "legibility_score" in annotation:
                    annotation["legibility_score"] = annotation["legibility_score"][valid_mask]
            else:
                # No valid boxes, create empty tensors
                annotation["bbox"] = torch.zeros((0, 4), dtype=torch.float32)
                if "id" in annotation:
                    annotation["id"] = torch.zeros((0,), dtype=torch.int64)
                if "category" in annotation:
                    annotation["category"] = torch.zeros((0,), dtype=torch.int64)
                if "visibility" in annotation:
                    annotation["visibility"] = torch.zeros((0,), dtype=torch.float32)
                if "role" in annotation:
                    annotation["role"] = torch.zeros((0,), dtype=torch.int64)
                if "jersey" in annotation:
                    annotation["jersey"] = torch.zeros((0,), dtype=torch.int64)
                if "digit_head" in annotation:
                    annotation["digit_head"] = torch.zeros((0,), dtype=torch.int64)
                if "digit_tail" in annotation:
                    annotation["digit_tail"] = torch.zeros((0,), dtype=torch.int64)
                if "legibility_score" in annotation:
                    annotation["legibility_score"] = torch.zeros((0,), dtype=torch.float32)
        
        # Adjust lines annotations
        if "lines" in annotation and len(annotation["lines"]) > 0:
            lines = annotation["lines"]
            adjusted_lines = {}
            
            for line_name, points in lines.items():
                adjusted_points = []
                for point in points:
                    # Convert normalized coordinates to absolute coordinates
                    abs_x = point["x"] * orig_w
                    abs_y = point["y"] * orig_h
                    
                    # Apply affine transformation
                    point_homogeneous = np.array([abs_x, abs_y, 1])
                    transformed_point = transform_matrix @ point_homogeneous
                    
                    # Convert back to normalized coordinates
                    adjusted_points.append({
                        "x": transformed_point[0] / orig_w,
                        "y": transformed_point[1] / orig_h
                    })
                
                adjusted_lines[line_name] = adjusted_points
            
            annotation["lines"] = adjusted_lines
        
        # Transform lines_target if it exists
        if "lines_target" in annotation:
            # Note: This is a heatmap transformation which is more complex
            # For simplicity, we'll apply the same affine transformation
            lines_target = annotation["lines_target"]
            h, w = lines_target.shape[-2:]
            # Scale transformation matrix to heatmap size
            scale_h = h / orig_h
            scale_w = w / orig_w
            heatmap_transform = transform_matrix.copy()
            heatmap_transform[0, 0] *= scale_w  # a
            heatmap_transform[0, 1] *= scale_h  # b  
            heatmap_transform[0, 2] *= scale_w  # tx
            heatmap_transform[1, 0] *= scale_w  # c
            heatmap_transform[1, 1] *= scale_h  # d
            heatmap_transform[1, 2] *= scale_h  # ty
            
            # Apply affine transformation to each heatmap channel
            transformed_heatmaps = []
            for i in range(lines_target.shape[0]):
                heatmap = lines_target[i].unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
                transformed_heatmap = v2.functional.affine(
                    heatmap,
                    angle=angle,
                    translate=[translate_x * scale_w, translate_y * scale_h],
                    scale=scale_factor,
                    shear=[shear_x, 0]
                )
                transformed_heatmaps.append(transformed_heatmap.squeeze(0).squeeze(0))
            annotation["lines_target"] = torch.stack(transformed_heatmaps, dim=0)
        
        # Transform keypoints_target if it exists
        if "keypoints_target" in annotation:
            keypoints_target = annotation["keypoints_target"]
            h, w = keypoints_target.shape[-2:]
            # Apply affine transformation to each keypoint heatmap channel
            transformed_keypoints = []
            for i in range(keypoints_target.shape[0]):
                heatmap = keypoints_target[i].unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
                transformed_heatmap = v2.functional.affine(
                    heatmap,
                    angle=angle,
                    translate=[translate_x * (w / orig_w), translate_y * (h / orig_h)],
                    scale=scale_factor,
                    shear=[shear_x, 0]
                )
                transformed_keypoints.append(transformed_heatmap.squeeze(0).squeeze(0))
            annotation["keypoints_target"] = torch.stack(transformed_keypoints, dim=0)
        
        return image, annotation, metas

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(degrees={self.degrees}, translate={self.translate}, scale={self.scale}, shear={self.shear}, p={self.p})"


class RandomCrop:
    """Apply random crop to images with annotation adjustment"""
    def __init__(self, crop_size_ratio_range=(0.6, 1.0), p=1.0):
        """
        Args:
            crop_size_ratio_range: Range of crop size as ratio of original image
            p: Probability of applying crop
        """
        self.crop_size_ratio_range = crop_size_ratio_range
        self.p = p

    def __call__(self, image, annotation, metas):
        # Store crop parameters for consistent application across frames
        if 'random_crop_params' not in metas:
            metas['random_crop_apply'] = random.random() <= self.p
            if not metas['random_crop_apply']:
                return image, annotation, metas
                
            # Get original image dimensions
            if isinstance(image, torch.Tensor):
                orig_h, orig_w = image.shape[-2], image.shape[-1]
            elif isinstance(image, list):
                orig_h, orig_w = image[0].shape[-2], image[0].shape[-1]
            else:
                raise NotImplementedError(f"Random crop not implemented for image type: {type(image)}")
            
            # Generate random crop parameters
            crop_ratio_h = random.uniform(self.crop_size_ratio_range[0], self.crop_size_ratio_range[1])
            crop_ratio_w = random.uniform(self.crop_size_ratio_range[0], self.crop_size_ratio_range[1])
            crop_h = int(orig_h * crop_ratio_h)
            crop_w = int(orig_w * crop_ratio_w)
            
            # Random crop position
            max_x = orig_w - crop_w
            max_y = orig_h - crop_h
            crop_x = random.randint(0, max_x) if max_x > 0 else 0
            crop_y = random.randint(0, max_y) if max_y > 0 else 0
            
            metas['random_crop_params'] = {
                'crop_x': crop_x,
                'crop_y': crop_y,
                'crop_w': crop_w,
                'crop_h': crop_h,
                'orig_w': orig_w,
                'orig_h': orig_h
            }
        else:
            # Check if we should apply random crop based on the stored decision
            if not metas['random_crop_apply']:
                return image, annotation, metas
        
        params = metas['random_crop_params']
        crop_x, crop_y, crop_w, crop_h = params['crop_x'], params['crop_y'], params['crop_w'], params['crop_h']
        orig_w, orig_h = params['orig_w'], params['orig_h']
        
        # Apply crop to image(s)
        if isinstance(image, torch.Tensor):
            image = image[..., crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        elif isinstance(image, list):
            # Apply to all frames in the list
            for i in range(len(image)):
                image[i] = image[i][..., crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        
        # Adjust bounding boxes
        if "bbox" in annotation and len(annotation["bbox"]) > 0:
            bbox = annotation["bbox"].clone()
            # bbox format: [x, y, w, h]
            
            # Adjust box coordinates relative to crop
            bbox[:, 0] = bbox[:, 0] - crop_x  # x
            bbox[:, 1] = bbox[:, 1] - crop_y  # y
            
            # Filter out boxes that are completely outside the crop
            x1 = bbox[:, 0]
            y1 = bbox[:, 1]
            x2 = bbox[:, 0] + bbox[:, 2]
            y2 = bbox[:, 1] + bbox[:, 3]
            
            # Check which boxes intersect with the crop area
            valid_mask = (x2 > 0) & (y2 > 0) & (x1 < crop_w) & (y1 < crop_h)
            
            if valid_mask.any():
                # Clip boxes to crop boundaries
                bbox[:, 0] = torch.clamp(bbox[:, 0], 0, crop_w)  # x
                bbox[:, 1] = torch.clamp(bbox[:, 1], 0, crop_h)  # y
                bbox[:, 2] = torch.clamp(x2, 0, crop_w) - bbox[:, 0]  # w
                bbox[:, 3] = torch.clamp(y2, 0, crop_h) - bbox[:, 1]  # h
                
                # Keep only valid boxes (with positive width and height)
                valid_size_mask = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
                final_mask = valid_mask & valid_size_mask
                
                # Filter all related annotations
                annotation["bbox"] = bbox[final_mask]
                if "id" in annotation:
                    annotation["id"] = annotation["id"][final_mask]
                if "category" in annotation:
                    annotation["category"] = annotation["category"][final_mask]
                if "visibility" in annotation:
                    annotation["visibility"] = annotation["visibility"][final_mask]
                if "role" in annotation:
                    annotation["role"] = annotation["role"][final_mask]
                if "jersey" in annotation:
                    annotation["jersey"] = annotation["jersey"][final_mask]
                if "digit_head" in annotation:
                    annotation["digit_head"] = annotation["digit_head"][final_mask]
                if "digit_tail" in annotation:
                    annotation["digit_tail"] = annotation["digit_tail"][final_mask]
                if "legibility_score" in annotation:
                    annotation["legibility_score"] = annotation["legibility_score"][final_mask]
            else:
                # No valid boxes, create empty tensors
                annotation["bbox"] = torch.zeros((0, 4), dtype=torch.float32)
                if "id" in annotation:
                    annotation["id"] = torch.zeros((0,), dtype=torch.int64)
                if "category" in annotation:
                    annotation["category"] = torch.zeros((0,), dtype=torch.int64)
                if "visibility" in annotation:
                    annotation["visibility"] = torch.zeros((0,), dtype=torch.float32)
                if "role" in annotation:
                    annotation["role"] = torch.zeros((0,), dtype=torch.int64)
                if "jersey" in annotation:
                    annotation["jersey"] = torch.zeros((0,), dtype=torch.int64)
                if "digit_head" in annotation:
                    annotation["digit_head"] = torch.zeros((0,), dtype=torch.int64)
                if "digit_tail" in annotation:
                    annotation["digit_tail"] = torch.zeros((0,), dtype=torch.int64)
                if "legibility_score" in annotation:
                    annotation["legibility_score"] = torch.zeros((0,), dtype=torch.float32)
        
        # Adjust lines annotations
        if "lines" in annotation and len(annotation["lines"]) > 0:
            lines = annotation["lines"]
            adjusted_lines = {}
            
            for line_name, points in lines.items():
                adjusted_points = []
                for point in points:
                    # Convert normalized coordinates to absolute coordinates
                    abs_x = point["x"] * orig_w
                    abs_y = point["y"] * orig_h
                    
                    # Adjust relative to crop
                    crop_abs_x = abs_x - crop_x
                    crop_abs_y = abs_y - crop_y
                    
                    # Check if point is within crop area
                    # if 0 <= crop_abs_x <= crop_w and 0 <= crop_abs_y <= crop_h:
                        # Convert back to normalized coordinates relative to crop
                    adjusted_points.append({
                        "x": crop_abs_x / crop_w,
                        "y": crop_abs_y / crop_h
                    })
                
                # Only keep lines that have at least one point within the crop
                # if len(adjusted_points) > 0:
                adjusted_lines[line_name] = adjusted_points
            
            annotation["lines"] = adjusted_lines
        
        # Store crop info for later use
        metas['crop_applied'] = True
        metas['crop_info'] = params
        
        return image, annotation, metas


class ClearAugmentationMetas:
    """Clear augmentation metadata to ensure independence between samples"""
    def __call__(self, image, annotation, metas):
        # Remove augmentation-specific metadata
        keys_to_remove = ['color_jitter_params', 'color_jitter_apply', 'horizontal_flip', 'gaussian_noise_params', 'gaussian_noise_apply', 'gaussian_blur_params', 'gaussian_blur_apply', 'random_affine_params', 'random_affine_apply', 'random_crop_params', 'random_crop_apply', 'crop_applied', 'crop_info']
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