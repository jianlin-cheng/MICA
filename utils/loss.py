import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class WeightedMultiTaskLoss(nn.Module):
    def __init__(self, 
                    backbone_weights=[0.03, 0.001, 0.3, 1], 
                    carbon_alpha_weights=[0.01, 0.001, 0.1, 1.0],
                    amino_acid_weights=[
                    0.001,   # background+masked 
                    1.0,   # ALA - most common baseline
                    1.8,   # CYS - rare but not extreme
                    1.1,   # ASP 
                    1.1,   # GLU
                    1.3,   # PHE
                    1.0,   # GLY - very common
                    1.6,   # HIS - rare
                    1.1,   # ILE
                    1.1,   # LYS  
                    0.9,   # LEU - most common, slight reduction
                    1.7,   # MET - rare
                    1.2,   # ASN
                    1.2,   # PRO
                    1.3,   # GLN
                    1.1,   # ARG
                    1.0,   # SER - common
                    1.1,   # THR
                    1.0,   # VAL - common
                    2.2,   # TRP - rarest, but much more reasonable
                    1.4    # TYR
                    ],
                    label_smoothing=0.1
                 ):

        super().__init__()
        self.backbone_weights = torch.tensor(backbone_weights, dtype=torch.float32).cuda()
        self.carbon_alpha_weights = torch.tensor(carbon_alpha_weights, dtype=torch.float32).cuda()
        self.amino_acid_weights = torch.tensor(amino_acid_weights, dtype=torch.float32).cuda()
        self.label_smoothing = label_smoothing

    def cosine_transition(self, epoch, start_epoch, end_epoch):
        """Smooth cosine transition between 0 and 1"""
        if epoch <= start_epoch:
            return 0.0
        elif epoch >= end_epoch:
            return 1.0

        progress = (epoch - start_epoch) / (end_epoch - start_epoch)
        return 0.5 * (1 - math.cos(math.pi * progress))

    def forward(self, outputs, targets, epoch, num_epochs):
        backbone_output, carbon_alpha_output, amino_acid_output = outputs
        backbone_target, carbon_alpha_target, amino_acid_target = targets

        # Starting weights 
        start_lambda_b = 0.6  
        start_lambda_c = 0.25 
        start_lambda_a = 0.15 

        # Target weights 
        target_lambda_b = 0.25 
        target_lambda_c = 0.4 
        target_lambda_a = 0.35  

        # Transition epoch
        transition_epoch = 25

        # Calculate progress with cosine annealing (smooth transition)
        progress = self.cosine_transition(epoch, 0, transition_epoch)

        # Cosine interpolation between starting and target weights
        lambda_b = start_lambda_b + (target_lambda_b - start_lambda_b) * progress
        lambda_c = start_lambda_c + (target_lambda_c - start_lambda_c) * progress
        lambda_a = start_lambda_a + (target_lambda_a - start_lambda_a) * progress

        # Normalize weights to sum to 1.0 (ensures exact balance)
        total_lambda = lambda_b + lambda_c + lambda_a
        lambda_b /= total_lambda
        lambda_c /= total_lambda
        lambda_a /= total_lambda

        # Include all classes with appropriate weighting
        backbone_loss = F.cross_entropy(
            backbone_output, backbone_target,
            weight=self.backbone_weights,
            reduction='mean'
        )

        carbon_alpha_loss = F.cross_entropy(
            carbon_alpha_output, carbon_alpha_target,
            weight=self.carbon_alpha_weights,
            reduction='mean'
        )

        amino_acid_loss = F.cross_entropy(
            amino_acid_output, amino_acid_target,
            weight=self.amino_acid_weights,
            reduction='mean'
        )

        # Apply weighted loss
        total_loss = (lambda_b * backbone_loss +
                        lambda_c * carbon_alpha_loss +
                        lambda_a * amino_acid_loss)


        return total_loss, {
            'total_loss': total_loss.item(),
            'backbone_loss': backbone_loss.item(),
            'carbon_alpha_loss': carbon_alpha_loss.item(),
            'amino_acid_loss': amino_acid_loss.item(),
            'lambda_b': lambda_b,
            'lambda_c': lambda_c,
            'lambda_a': lambda_a,
        }

