"""
nnInteractive Dataset Class
===========================

This module provides dataset classes for fine-tuning nnInteractive models.
It supports various data formats and interaction types.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Any, Optional, Tuple, List
import logging
from pathlib import Path

# Try to import medical image libraries
try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False
    print("Warning: SimpleITK not available. NIfTI support limited.")

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    print("Warning: h5py not available. HDF5 support limited.")

logger = logging.getLogger(__name__)


class nnInteractiveDataset(Dataset):
    """
    Dataset class for nnInteractive fine-tuning
    
    This dataset supports:
    - Multiple data formats (NIfTI, NumPy, HDF5)
    - Various interaction types (points, boxes, scribbles, lassos)
    - Data augmentation and preprocessing
    - Flexible data organization
    """
    
    def __init__(
        self,
        data_path: str,
        transforms: Optional[Dict[str, Any]] = None,
        split: str = "train",
        interaction_types: Optional[List[str]] = None,
        max_interactions: int = 10,
        random_seed: int = 42
    ):
        """
        Initialize the dataset
        
        Args:
            data_path: Path to the data directory
            transforms: Dictionary of transform configurations
            split: Dataset split ("train", "validation", "test")
            interaction_types: Types of interactions to generate
            max_interactions: Maximum number of interactions per sample
            random_seed: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.transforms = transforms or {}
        self.split = split
        self.interaction_types = interaction_types or ["point", "box", "scribble"]
        self.max_interactions = max_interactions
        self.random_seed = random_seed
        
        # Set random seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Validate data path
        if not self.data_path.exists():
            raise ValueError(f"Data path does not exist: {self.data_path}")
        
        # Setup data organization
        self._setup_data_organization()
        
        # Load data samples
        self.samples = self._load_samples()
        
        # Setup transforms
        self.transform_pipeline = nnInteractiveTransforms(transforms)
        
        logger.info(f"Initialized {split} dataset with {len(self.samples)} samples")
    
    def _setup_data_organization(self) -> None:
        """Setup data organization structure"""
        # Expected directory structure:
        # data_path/
        # ├── images/          # Input images
        # ├── masks/           # Ground truth masks
        # ├── interactions/    # Pre-computed interactions (optional)
        # └── metadata.json    # Dataset metadata
        
        self.images_dir = self.data_path / "images"
        self.masks_dir = self.data_path / "masks"
        self.interactions_dir = self.data_path / "interactions"
        self.metadata_file = self.data_path / "metadata.json"
        
        # Validate required directories
        if not self.images_dir.exists():
            raise ValueError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise ValueError(f"Masks directory not found: {self.masks_dir}")
        
        # Check if interactions directory exists
        self.has_precomputed_interactions = self.interactions_dir.exists()
        
        # Load metadata if available
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load dataset metadata"""
        if self.metadata_file.exists():
            try:
                import json
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
                return {}
        return {}
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load data samples"""
        samples = []
        
        # Get all image files
        image_files = list(self.images_dir.glob("*"))
        image_files = [f for f in image_files if self._is_valid_image_file(f)]
        
        for image_file in image_files:
            # Find corresponding mask file
            mask_file = self._find_corresponding_mask(image_file)
            if mask_file is None:
                logger.warning(f"No mask found for image: {image_file}")
                continue
            
            # Check if pre-computed interactions exist
            interaction_file = None
            if self.has_precomputed_interactions:
                interaction_file = self._find_corresponding_interaction(image_file)
            
            sample = {
                'image_path': str(image_file),
                'mask_path': str(mask_file),
                'interaction_path': interaction_file,
                'sample_id': image_file.stem
            }
            
            # Add metadata if available
            if self.metadata and image_file.stem in self.metadata:
                sample.update(self.metadata[image_file.stem])
            
            samples.append(sample)
        
        # Sort samples for reproducibility
        samples.sort(key=lambda x: x['sample_id'])
        
        return samples
    
    def _is_valid_image_file(self, file_path: Path) -> bool:
        """Check if file is a valid image file"""
        valid_extensions = {'.nii.gz', '.nii', '.npz', '.h5', '.hdf5'}
        return file_path.suffix in valid_extensions or file_path.suffixes[-2:] == ['.nii', '.gz']
    
    def _find_corresponding_mask(self, image_file: Path) -> Optional[Path]:
        """Find corresponding mask file for an image"""
        # Try different naming conventions
        possible_names = [
            image_file.stem + "_mask" + image_file.suffix,
            image_file.stem + "_gt" + image_file.suffix,
            image_file.stem + "_label" + image_file.suffix,
            image_file.stem + ".nii.gz",  # Assume same name with .nii.gz
            image_file.stem + ".nii",
        ]
        
        for name in possible_names:
            mask_path = self.masks_dir / name
            if mask_path.exists():
                return mask_path
        
        # Try to find any file with similar name
        for mask_file in self.masks_dir.glob(f"{image_file.stem}*"):
            if mask_file.exists():
                return mask_file
        
        return None
    
    def _find_corresponding_interaction(self, image_file: Path) -> Optional[Path]:
        """Find corresponding interaction file"""
        interaction_path = self.interactions_dir / f"{image_file.stem}_interactions.json"
        if interaction_path.exists():
            return interaction_path
        return None
    
    def __len__(self) -> int:
        """Return the number of samples"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample by index"""
        sample = self.samples[idx]
        
        # Load image and mask
        image = self._load_image(sample['image_path'])
        mask = self._load_mask(sample['mask_path'])
        
        # Generate or load interactions
        interactions = self._get_interactions(sample)
        
        # Apply transforms
        if self.transform_pipeline:
            image, mask, interactions = self.transform_pipeline(image, mask, interactions)
        
        # Prepare output
        output = {
            'image': image,
            'target': mask,
            'prompts': interactions,
            'sample_id': sample['sample_id']
        }
        
        return output
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load image from file"""
        image_path = Path(image_path)
        
        if image_path.suffix == '.npz':
            data = np.load(image_path)
            # Assume first array is the image
            image = data[data.files[0]]
        elif image_path.suffix in ['.h5', '.hdf5'] and H5PY_AVAILABLE:
            with h5py.File(image_path, 'r') as f:
                # Assume first dataset is the image
                key = list(f.keys())[0]
                image = f[key][:]
        elif image_path.suffix in ['.nii', '.nii.gz'] and SITK_AVAILABLE:
            sitk_image = sitk.ReadImage(str(image_path))
            image = sitk.GetArrayFromImage(sitk_image)
        else:
            # Try to load as numpy array
            try:
                image = np.load(image_path)
            except:
                raise ValueError(f"Unsupported image format: {image_path}")
        
        # Ensure 3D format
        if image.ndim == 2:
            image = image[None, ...]  # Add channel dimension
        elif image.ndim == 4:
            image = image[0]  # Take first channel
        
        # Convert to torch tensor
        image = torch.from_numpy(image).float()
        
        # Normalize if needed
        if self.transforms.get('normalize', False):
            mean = self.transforms.get('normalize_mean', [0.5])
            std = self.transforms.get('normalize_std', [0.5])
            image = (image - mean[0]) / std[0]
        
        return image
    
    def _load_mask(self, mask_path: str) -> torch.Tensor:
        """Load mask from file"""
        mask_path = Path(mask_path)
        
        if mask_path.suffix == '.npz':
            data = np.load(mask_path)
            mask = data[data.files[0]]
        elif mask_path.suffix in ['.h5', '.hdf5'] and H5PY_AVAILABLE:
            with h5py.File(mask_path, 'r') as f:
                key = list(f.keys())[0]
                mask = f[key][:]
        elif mask_path.suffix in ['.nii', '.nii.gz'] and SITK_AVAILABLE:
            sitk_mask = sitk.ReadImage(str(mask_path))
            mask = sitk.GetArrayFromImage(sitk_mask)
        else:
            try:
                mask = np.load(mask_path)
            except:
                raise ValueError(f"Unsupported mask format: {mask_path}")
        
        # Ensure 3D format
        if mask.ndim == 2:
            mask = mask[None, ...]
        elif mask.ndim == 4:
            mask = mask[0]
        
        # Convert to torch tensor
        mask = torch.from_numpy(mask).float()
        
        # Binarize mask if needed
        if mask.max() > 1:
            mask = (mask > 0).float()
        
        return mask
    
    def _get_interactions(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Generate or load interactions for a sample"""
        if sample.get('interaction_path') and self.has_precomputed_interactions:
            return self._load_precomputed_interactions(sample['interaction_path'])
        else:
            return self._generate_interactions(sample)
    
    def _load_precomputed_interactions(self, interaction_path: str) -> Dict[str, torch.Tensor]:
        """Load pre-computed interactions"""
        try:
            import json
            with open(interaction_path, 'r') as f:
                interactions = json.load(f)
            
            # Convert to torch tensors
            torch_interactions = {}
            for key, value in interactions.items():
                if isinstance(value, list):
                    torch_interactions[key] = torch.tensor(value, dtype=torch.float32)
                else:
                    torch_interactions[key] = torch.tensor([value], dtype=torch.float32)
            
            return torch_interactions
        except Exception as e:
            logger.warning(f"Failed to load interactions from {interaction_path}: {e}")
            return self._generate_interactions({})
    
    def _generate_interactions(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Generate random interactions for training"""
        # This is a simplified interaction generation
        # In practice, you might want more sophisticated interaction simulation
        
        interactions = {}
        
        # Generate random points
        if "point" in self.interaction_types:
            num_points = np.random.randint(1, 5)
            points = []
            for _ in range(num_points):
                # Random 3D coordinates
                x = np.random.uniform(0, 1)
                y = np.random.uniform(0, 1)
                z = np.random.uniform(0, 1)
                points.append([x, y, z])
            interactions['points'] = torch.tensor(points, dtype=torch.float32)
        
        # Generate random boxes
        if "box" in self.interaction_types:
            num_boxes = np.random.randint(1, 3)
            boxes = []
            for _ in range(num_boxes):
                # Random 2D box (one dimension fixed for 2D box)
                x1, x2 = sorted(np.random.uniform(0, 1, 2))
                y1, y2 = sorted(np.random.uniform(0, 1, 2))
                z = np.random.randint(0, 10)  # Fixed slice
                boxes.append([[x1, x2], [y1, y2], [z, z+1]])
            interactions['boxes'] = torch.tensor(boxes, dtype=torch.float32)
        
        # Generate random scribbles
        if "scribble" in self.interaction_types:
            # Create a simple random scribble pattern
            scribble = torch.zeros((1, 64, 64, 64), dtype=torch.float32)
            # Add random scribble lines
            for _ in range(np.random.randint(1, 4)):
                start = np.random.randint(0, 64, 3)
                end = np.random.randint(0, 64, 3)
                # Simple line drawing (simplified)
                scribble[0, start[0], start[1], start[2]] = 1.0
                scribble[0, end[0], end[1], end[2]] = 1.0
            interactions['scribbles'] = scribble
        
        return interactions
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get information about a sample without loading the data"""
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range")
        return self.samples[idx].copy()


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
