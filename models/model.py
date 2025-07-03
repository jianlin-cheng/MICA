import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleInput(nn.Module):
    def __init__(self, base_filters=64, dropout_rate=0.1):
        super().__init__()
        # Multi-scale convolutions for experimental map
        self.exp_convs = nn.ModuleList([
            nn.Conv3d(1, base_filters//2, 3, padding=1),
            nn.Conv3d(1, base_filters//2, 5, padding=2),
            nn.Conv3d(1, base_filters//2, 7, padding=3),
            nn.Conv3d(1, base_filters//2, 9, padding=4)
        ])
        
        # Support path and feature path
        self.feat_conv = nn.Conv3d(24, base_filters, 3, padding=1)
        
        # Feature attention
        self.exp_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(base_filters*2, base_filters, 1),
            nn.ReLU(),
            nn.Conv3d(base_filters, base_filters*2, 1),
            nn.Sigmoid()
        )
        
        self.exp_downsizing = nn.Conv3d(base_filters*2, base_filters, 1)
        
        # Conditional feature integration
        self.feat_gate = nn.Sequential(
            nn.Conv3d(base_filters, base_filters//4, 1),
            nn.ReLU(),
            nn.Conv3d(base_filters//4, 1, 1),
            nn.Sigmoid()
        )
        
        self.fusion = nn.Conv3d(base_filters*3, base_filters, 1)
        
        # REGULARIZATION 1: Input-level dropout
        self.input_dropout = nn.Dropout3d(dropout_rate)
        
    def forward(self, exp_map, af_features=None):
        # Apply input dropout during training
        if self.training:
            exp_map = self.input_dropout(exp_map)
            
        multi_scale_features = []
        for conv in self.exp_convs:
            multi_scale_features.append(conv(exp_map))
        x_exp = torch.cat(multi_scale_features, dim=1)
        
        # Apply self-attention to enhance experimental map features
        x_exp_enhanced = x_exp * self.exp_attention(x_exp)
        
        if af_features is None:
            return self.exp_downsizing(x_exp_enhanced)
        
        # Check if af_features are reliable (non-zero)
        is_af_zero = af_features.abs().sum() < 1e-6
        
        if is_af_zero:
            return self.exp_downsizing(x_exp_enhanced)
        
        # Process AF features with dropout
        if self.training and af_features is not None:
            af_features = self.input_dropout(af_features)
            
        x_feat = self.feat_conv(af_features)
        feat_importance = self.feat_gate(x_feat)
        x_feat_weighted = x_feat * feat_importance
        
        x = torch.cat([x_exp_enhanced, x_feat_weighted], dim=1)
        return self.fusion(x)

class DualAttention(nn.Module):
    def __init__(self, channels, dropout_rate=0.1):
        super().__init__()
        self.local_attn = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, groups=channels),
            nn.InstanceNorm3d(channels),
            nn.ReLU(),
            # REGULARIZATION 2: Attention dropout
            nn.Dropout3d(dropout_rate)
        )
        
        self.global_attn = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels//4, 1),
            nn.ReLU(),
            nn.Dropout3d(dropout_rate),  # Attention bottleneck dropout
            nn.Conv3d(channels//4, channels, 1),
            nn.Sigmoid()
        )
        
        self.fusion = nn.Conv3d(channels*2, channels, 1)
        
    def forward(self, x):
        local_feat = self.local_attn(x)
        global_feat = self.global_attn(x) * x
        return self.fusion(torch.cat([local_feat, global_feat], dim=1))

class ResidualDenseBlock(nn.Module):
    def __init__(self, channels, dropout_rate=0.15):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(channels, channels//2, 3, padding=1),
            nn.InstanceNorm3d(channels//2),
            nn.ReLU(),
            # REGULARIZATION 3: Dense block dropout (higher rate)
            nn.Dropout3d(dropout_rate)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(channels*3//2, channels//2, 3, padding=1),
            nn.InstanceNorm3d(channels//2),
            nn.ReLU(),
            nn.Dropout3d(dropout_rate)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(channels*2, channels, 3, padding=1),
            nn.InstanceNorm3d(channels),
            nn.ReLU(),
            nn.Dropout3d(dropout_rate)
        )
        
        self.se = SEBlock(channels, dropout_rate=dropout_rate)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat([x, x1], dim=1))
        x3 = self.conv3(torch.cat([x, x1, x2], dim=1))
        return self.se(x3)

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.dense_block = ResidualDenseBlock(in_channels, dropout_rate)
        self.dual_attn = DualAttention(in_channels, dropout_rate)
        self.transition = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(),
            # REGULARIZATION 4: Transition dropout
            nn.Dropout3d(dropout_rate * 0.5)  # Lower rate for transitions
        )
        
    def forward(self, x):
        x = self.dense_block(x)
        x = self.dual_attn(x)
        return self.transition(x)

class FPN(nn.Module):
    def __init__(self, base_filters, dropout_rate=0.1):
        super().__init__()
        self.lateral = nn.ModuleList([
            nn.Conv3d(base_filters*2, base_filters, 1),
            nn.Conv3d(base_filters*4, base_filters, 1),
            nn.Conv3d(base_filters*8, base_filters, 1)
        ])
        
        self.smooth = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(base_filters, base_filters, 3, padding=1),
                # REGULARIZATION 5: FPN feature dropout
                nn.Dropout3d(dropout_rate * 0.5)
            ),
            nn.Sequential(
                nn.Conv3d(base_filters, base_filters, 3, padding=1),
                nn.Dropout3d(dropout_rate * 0.5)
            ),
            nn.Sequential(
                nn.Conv3d(base_filters, base_filters, 3, padding=1),
                nn.Dropout3d(dropout_rate * 0.5)
            )
        ])
        
        self.weights = nn.Parameter(torch.ones(3)/3)
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, features):
        weights = self.softmax(self.weights)
        
        # Lateral connections
        c3, c4, c5 = features
        p5 = self.lateral[2](c5)
        p4 = self.lateral[1](c4)
        p3 = self.lateral[0](c3)
        
        # Top-down pathway
        p5_up = F.interpolate(p5, size=p3.shape[2:], mode='trilinear', align_corners=True)
        p4_up = F.interpolate(p4, size=p3.shape[2:], mode='trilinear', align_corners=True)
        
        # Smooth with dropout
        p5_smooth = self.smooth[2](p5_up)
        p4_smooth = self.smooth[1](p4_up)
        p3_smooth = self.smooth[0](p3)
        
        # Weighted fusion
        return torch.cat([
            weights[0] * p3_smooth,
            weights[1] * p4_smooth,
            weights[2] * p5_smooth
        ], dim=1)

class TaskSpecificDecoderHead(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_rate=0.2):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, 3, padding=1)
        self.norm1 = nn.InstanceNorm3d(64)
        self.conv2 = nn.Conv3d(64, 32, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(32)
        
        self.calibration = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(32, 8, 1),
            nn.ReLU(),
            # REGULARIZATION 6: Head-specific dropout (highest rate)
            nn.Dropout3d(dropout_rate),
            nn.Conv3d(8, 32, 1),
            nn.Sigmoid()
        )
        
        self.final = nn.Conv3d(32, num_classes, 1)
        
        # REGULARIZATION 7: Feature dropout before final prediction
        self.feature_dropout = nn.Dropout3d(dropout_rate)
        
    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        
        # Apply feature dropout before calibration
        if self.training:
            x = self.feature_dropout(x)
            
        x = x * self.calibration(x)
        return self.final(x)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16, dropout_rate=0.1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            # REGULARIZATION 8: SE block dropout
            nn.Dropout(dropout_rate),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y

class MICA(nn.Module):
    def __init__(self, base_filters=64, dropout_schedule=None):
        super().__init__()
        
        # ADAPTIVE DROPOUT: Adjust dropout rates based on training epoch
        if dropout_schedule is None:
            # Default progressive dropout schedule
            self.dropout_schedule = {
                'early': 0, 
                'mid': 0,  
                'late': 0 
            }
        else:
            self.dropout_schedule = dropout_schedule
            
        self.current_dropout = self.dropout_schedule['early']
        
        # Multi-scale input processing
        self.input_processing = MultiScaleInput(base_filters, self.current_dropout)
        
        # Enhanced encoder with residual dense connections
        self.encoder = nn.ModuleList([
            Encoder(base_filters, base_filters*2, self.current_dropout),
            Encoder(base_filters*2, base_filters*4, self.current_dropout),
            Encoder(base_filters*4, base_filters*8, self.current_dropout)
        ])
        
        # Adaptive FPN
        self.fpn = FPN(base_filters, self.current_dropout)
        
        # Task-specific heads with progressive dropout
        self.backbone_head = TaskSpecificDecoderHead(192, 4, self.current_dropout * 2)
        self.ca_head = TaskSpecificDecoderHead(196, 4, self.current_dropout * 2)
        self.aa_head = TaskSpecificDecoderHead(200, 21, self.current_dropout * 2)
        
    def update_dropout_rate(self, epoch):
        """Update dropout rates based on training epoch"""
        if epoch < 35:
            new_rate = self.dropout_schedule['early']
        elif epoch < 50:
            new_rate = self.dropout_schedule['mid']
        else:
            new_rate = self.dropout_schedule['late']
            
        if new_rate != self.current_dropout:
            self.current_dropout = new_rate
            self._update_all_dropout_rates(new_rate)
    
    def _update_all_dropout_rates(self, rate):
        """Recursively update dropout rates in all modules"""
        def update_dropout(module):
            for child in module.children():
                if isinstance(child, (nn.Dropout, nn.Dropout3d)):
                    child.p = rate
                elif hasattr(child, 'children'):
                    update_dropout(child)
        
        update_dropout(self)
        
        # Update head dropout rates (2x base rate)
        def update_head_dropout(module, head_rate):
            for child in module.children():
                if isinstance(child, (nn.Dropout, nn.Dropout3d)):
                    child.p = head_rate
                elif hasattr(child, 'children'):
                    update_head_dropout(child, head_rate)
        
        update_head_dropout(self.backbone_head, rate * 2)
        update_head_dropout(self.ca_head, rate * 2)
        update_head_dropout(self.aa_head, rate * 2)
        
    def forward(self, exp_map, af_features=None):
        # Input processing
        x = self.input_processing(exp_map, af_features)
        
        # Encoder pathway
        features = []
        for encoder in self.encoder:
            x = encoder(x)
            features.append(x)
        
        # FPN
        fpn_features = self.fpn(features)
        # Task-specific predictions with progressive feature concatenation
        backbone = self.backbone_head(fpn_features)
        ca = self.ca_head(torch.cat([fpn_features, backbone], dim=1))
        aa = self.aa_head(torch.cat([fpn_features, backbone, ca], dim=1))

        return backbone, ca, aa

# Enhanced weight initialization with regularization considerations
def init_weights_with_regularization(m):
    if isinstance(m, nn.Conv3d):
        # Xavier initialization for better gradient flow with dropout
        nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def create_regularized_model():
    # Custom dropout schedule for regularization
    dropout_schedule = {
        'early': 0.01,   # Light regularization early
        'mid': 0.05,     # Moderate regularization mid-training  
        'late': 0.1     # Heavy regularization late
    }
    
    model = MICA(base_filters=64, dropout_schedule=dropout_schedule)
    model.apply(init_weights_with_regularization)
    return model

