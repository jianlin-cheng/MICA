from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn.functional as F
import random
from typing import Tuple

class CryoEMAugmentation:
    """Medium-level augmentation for CryoEM data."""
    
    def __init__(self):
        # Medium augmentation parameters
        self.gaussian_noise_std = 0.03
        self.brightness_range = 0.05
        self.contrast_range = (0.9, 1.1)
        self.rotation_prob = 0.5
        self.flip_prob = 0.3
        self.translation_pixels = 2
        self.blur_prob = 0.2
        self.augment_prob = 0.4
    
    def __call__(self, density_map: torch.Tensor, af3_features: torch.Tensor, 
                 targets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple:
        """Apply augmentation to single sample."""
        if random.random() > self.augment_prob:
            return density_map, af3_features, targets
        
        backbone_targets, ca_targets, aa_targets = targets
        # Noise augmentation
        if random.random() < 0.7:
            density_map = density_map + torch.randn_like(density_map) * self.gaussian_noise_std
        
        # Intensity augmentation
        if random.random() < 0.5:
            brightness = random.uniform(-self.brightness_range, self.brightness_range)
            density_map = density_map + brightness
        
        if random.random() < 0.5:
            contrast = random.uniform(*self.contrast_range)
            mean_val = density_map.mean()
            density_map = (density_map - mean_val) * contrast + mean_val
        
        # Spatial augmentation (applied consistently)
        if random.random() < 0.6:
            all_inputs = torch.cat([density_map, af3_features], dim=0)  # [25, 64, 64, 64]
            all_targets = torch.stack([backbone_targets, ca_targets, aa_targets], dim=0)  # [3, 64, 64, 64]
            
            # Random 90-degree rotation
            if random.random() < self.rotation_prob:
                k = random.randint(1, 3)
                axis = random.choice([(1, 2), (1, 3), (2, 3)])
                all_inputs = torch.rot90(all_inputs, k=k, dims=axis)
                all_targets = torch.rot90(all_targets, k=k, dims=axis)
            
            # Random flip
            if random.random() < self.flip_prob:
                flip_axis = random.choice([1, 2, 3])
                all_inputs = torch.flip(all_inputs, dims=[flip_axis])
                all_targets = torch.flip(all_targets, dims=[flip_axis])
            
            # Small translation
            if random.random() < 0.4:
                for i in range(3):
                    shift = random.randint(-self.translation_pixels, self.translation_pixels)
                    if shift != 0:
                        axis = i + 1
                        all_inputs = torch.roll(all_inputs, shifts=shift, dims=axis)
                        all_targets = torch.roll(all_targets, shifts=shift, dims=axis)
            
            # Split back
            density_map = all_inputs[:1]
            af3_features = all_inputs[1:]
            backbone_targets = all_targets[0]
            ca_targets = all_targets[1]
            aa_targets = all_targets[2]
        
        # Blur density map
        if random.random() < self.blur_prob:
            density_map = self._apply_blur(density_map)
        
        return density_map, af3_features, (backbone_targets, ca_targets, aa_targets)
    
    def _apply_blur(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply simple 3D Gaussian blur."""
        kernel_size = 3
        sigma = random.uniform(0.5, 1.0)
        
        # Create 1D Gaussian kernel
        x = torch.arange(kernel_size, dtype=tensor.dtype, device=tensor.device)
        x = x - kernel_size // 2
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()
        
        # Apply separable convolution
        padding = kernel_size // 2
        C = tensor.shape[0]
        tensor = tensor.unsqueeze(0)
        
        # Blur along each spatial dimension
        for dim in range(1, 4):  # D, H, W dimensions
            kernel_shape = [1, 1, 1, 1, 1]
            kernel_shape[dim + 1] = kernel_size
            kernel_3d = kernel.view(kernel_shape).expand([C, 1] + [-1] * 3)
            
            pad_3d = [0, 0, 0, 0, 0, 0]
            pad_3d[2 * (3 - dim) : 2 * (3 - dim) + 2] = [padding, padding]
            
            tensor = F.conv3d(tensor, kernel_3d, padding=tuple(pad_3d[:6:2]), groups=C)
        
        return tensor.squeeze(0)


class CryoEMDataset(Dataset):
    def __init__(self, data_dir, exp_only_prob=0.4, use_augmentation=True):
        super().__init__()
        self.data_dir = data_dir
        self.exp_only_prob = exp_only_prob
        self.use_augmentation = use_augmentation
        self.training = True
        
        self.types = [
            'CA', 'N', 'C', 'O', 'ALA', 'CYS', 'ASP', 'GLU', 'PHE',
            'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO',
            'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR'
        ]
        
        # Setup augmentation
        if self.use_augmentation:
            self.augment = CryoEMAugmentation()
        else:
            self.augment = None
    
    def __len__(self):
        return len(self.data_dir)
    
    def __getitem__(self, idx):
        map_path = self.data_dir[idx]
        input_map_ = np.load(map_path)['grid']
        
        backbone_mask_path = map_path.replace('normalized_maps', 'BB_masks')
        backbone_mask_ = np.load(backbone_mask_path)['grid']
        
        carbon_alpha_mask_path = map_path.replace('normalized_maps', 'CA_masks')
        carbon_alpha_mask_ = np.load(carbon_alpha_mask_path)['grid']
        
        amino_acid_mask_path = map_path.replace('normalized_maps', 'AA_masks')
        amino_acid_mask_ = np.load(amino_acid_mask_path)['grid']
        
        feature_list = []
        for feat_type in self.types:
            feature_path = map_path.replace('normalized_maps', f'{feat_type}_encodings')
            feature = np.load(feature_path)['grid']
            feature_list.append(feature)
        
        af3_features_ = np.stack(feature_list, axis=0)  # Shape: (24, D, H, W)
        
        # Random AF3 feature usage
        if np.random.random() < self.exp_only_prob:
            af3_features_ = np.zeros_like(af3_features_)
        
        # Convert to tensors
        input_map = torch.from_numpy(input_map_).unsqueeze(0).float()
        backbone_mask = torch.from_numpy(backbone_mask_).long()
        carbon_alpha_mask = torch.from_numpy(carbon_alpha_mask_).long()
        amino_acid_mask = torch.from_numpy(amino_acid_mask_).long()
        af3_features = torch.from_numpy(af3_features_).float()
        
        # Apply augmentation during training
        if self.use_augmentation and self.augment is not None:
            input_map, af3_features, targets = self.augment(
                input_map, af3_features, 
                (backbone_mask, carbon_alpha_mask, amino_acid_mask)
            )
            backbone_mask, carbon_alpha_mask, amino_acid_mask = targets
        
        return input_map, af3_features, backbone_mask, carbon_alpha_mask, amino_acid_mask


class CryoEMTestDataset(Dataset):
    def __init__(self, data_dir, transform):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.types = [
            'CA', 'N', 'C', 'O', 'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 
            'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 
            'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR'
        ]

    def __len__(self):
        return len(self.data_dir)
    

    def __getitem__(self, idx):
        map_path = self.data_dir[idx]
        data = np.load(map_path)
        map = data['grid']
        metadata = {
            'i': data['i'],
            'j': data['j'],
            'k': data['k'],
            'di': data['di'],
            'dj': data['dj'],
            'dk': data['dk'],
            'orig_shape': data['orig_shape'],
            'filename': self.data_dir[idx].split("/")[-1].split(".")[0]
        }
        
        try:
            # Load and concatenate features
            feature_list = []
            for feat_type in self.types:
                feature_path = map_path.replace('normalized_map_grids', f"AF3_encoding_grids/{feat_type}_grids")
                feature_path = feature_path.replace('normalized_map', f"{feat_type}")
                feature = np.load(feature_path)['grid']
                feature_list.append(feature)
            af3_features_ = np.stack(feature_list, axis=0)  # Shape: (24, D, H, W)_     
        except:
            af3_features_ = np.zeros((24, 64, 64, 64))  
            
        input_map = torch.from_numpy(map).unsqueeze(0).float()      
        af3_features = torch.from_numpy(af3_features_).float()
        
        return input_map, af3_features, metadata







