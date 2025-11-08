"""
nnInteractive Transforms Module
==============================

This module provides transform classes for data augmentation and preprocessing.
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple, List
import torch.nn.functional as F


class nnInteractiveTransforms:
    """
    Transform pipeline for nnInteractive data
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize transforms
        
        Args:
            config: Transform configuration dictionary
        """
        self.config = config
        self.setup_transforms()
    
    def setup_transforms(self) -> None:
        """Setup transform pipeline"""
        self.transforms = []
        
        # Add transforms based on configuration
        if self.config.get('random_flip', False):
            self.transforms.append(RandomFlip3D())
        
        if self.config.get('random_rotation', 0) > 0:
            self.transforms.append(RandomRotation3D(
                max_angle=self.config['random_rotation']
            ))
        
        if self.config.get('random_scale', None):
            self.transforms.append(RandomScale3D(
                scale_range=self.config['random_scale']
            ))
        
        if self.config.get('target_size', None):
            self.transforms.append(Resize3D(
                target_size=self.config['target_size']
            ))
    
    def __call__(self, image: torch.Tensor, mask: torch.Tensor, 
                 interactions: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply transforms to image, mask, and interactions"""
        for transform in self.transforms:
            image, mask, interactions = transform(image, mask, interactions)
        
        return image, mask, interactions


class RandomFlip3D:
    """Random 3D flipping transform"""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: torch.Tensor, mask: torch.Tensor, 
                 interactions: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # Random flip along each axis
        for axis in [0, 1, 2]:
            if np.random.random() < self.p:
                image = torch.flip(image, [axis])
                mask = torch.flip(mask, [axis])
                # Update interactions accordingly
                interactions = self._update_interactions_for_flip(interactions, axis)
        
        return image, mask, interactions
    
    def _update_interactions_for_flip(self, interactions: Dict[str, torch.Tensor], axis: int) -> Dict[str, torch.Tensor]:
        """Update interaction coordinates after flipping"""
        # This is a simplified update - in practice, you'd need more sophisticated logic
        return interactions


class RandomRotation3D:
    """Random 3D rotation transform"""
    
    def __init__(self, max_angle: float = 15.0):
        self.max_angle = max_angle
    
    def __call__(self, image: torch.Tensor, mask: torch.Tensor, 
                 interactions: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # Simplified 3D rotation - in practice, you'd use proper 3D rotation matrices
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        
        # Apply rotation (simplified)
        # This is a placeholder - implement proper 3D rotation
        return image, mask, interactions


class RandomScale3D:
    """Random 3D scaling transform"""
    
    def __init__(self, scale_range: List[float]):
        self.scale_range = scale_range
    
    def __call__(self, image: torch.Tensor, mask: torch.Tensor, 
                 interactions: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        
        # Apply scaling (simplified)
        # This is a placeholder - implement proper 3D scaling
        return image, mask, interactions


class Resize3D:
    """3D resize transform"""
    
    def __init__(self, target_size: List[int]):
        self.target_size = target_size
    
    def __call__(self, image: torch.Tensor, mask: torch.Tensor, 
                 interactions: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # Resize image and mask to target size
        # This is a placeholder - implement proper 3D resizing
        return image, mask, interactions


class Normalize3D:
    """3D normalization transform"""
    
    def __init__(self, mean: List[float], std: List[float]):
        self.mean = torch.tensor(mean).view(1, 1, 1, 1)
        self.std = torch.tensor(std).view(1, 1, 1, 1)
    
    def __call__(self, image: torch.Tensor, mask: torch.Tensor, 
                 interactions: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        image = (image - self.mean) / self.std
        return image, mask, interactions


class RandomCrop3D:
    """Random 3D crop transform"""
    
    def __init__(self, crop_size: List[int]):
        self.crop_size = crop_size
    
    def __call__(self, image: torch.Tensor, mask: torch.Tensor, 
                 interactions: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # Get current dimensions
        _, h, w, d = image.shape
        
        # Calculate crop boundaries
        h_start = np.random.randint(0, max(1, h - self.crop_size[0]))
        w_start = np.random.randint(0, max(1, w - self.crop_size[1]))
        d_start = np.random.randint(0, max(1, d - self.crop_size[2]))
        
        h_end = h_start + self.crop_size[0]
        w_end = w_start + self.crop_size[1]
        d_end = d_start + self.crop_size[2]
        
        # Crop image and mask
        image = image[:, h_start:h_end, w_start:w_end, d_start:d_end]
        mask = mask[:, h_start:h_end, w_start:w_end, d_start:d_end]
        
        # Update interactions for crop
        interactions = self._update_interactions_for_crop(interactions, h_start, w_start, d_start)
        
        return image, mask, interactions
    
    def _update_interactions_for_crop(self, interactions: Dict[str, torch.Tensor], 
                                    h_start: int, w_start: int, d_start: int) -> Dict[str, torch.Tensor]:
        """Update interaction coordinates after cropping"""
        # This is a simplified update - in practice, you'd need more sophisticated logic
        return interactions


class RandomNoise3D:
    """Random 3D noise transform"""
    
    def __init__(self, noise_factor: float = 0.05):
        self.noise_factor = noise_factor
    
    def __call__(self, image: torch.Tensor, mask: torch.Tensor, 
                 interactions: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # Add random noise to image
        noise = torch.randn_like(image) * self.noise_factor
        image = image + noise
        
        return image, mask, interactions


class RandomBrightnessContrast3D:
    """Random brightness and contrast transform"""
    
    def __init__(self, brightness_factor: float = 0.1, contrast_factor: float = 0.1):
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
    
    def __call__(self, image: torch.Tensor, mask: torch.Tensor, 
                 interactions: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # Random brightness
        brightness = 1.0 + np.random.uniform(-self.brightness_factor, self.brightness_factor)
        image = image * brightness
        
        # Random contrast
        contrast = 1.0 + np.random.uniform(-self.contrast_factor, self.contrast_factor)
        mean = image.mean()
        image = (image - mean) * contrast + mean
        
        return image, mask, interactions
