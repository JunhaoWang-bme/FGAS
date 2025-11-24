import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_rate=0.1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, dropout_rate=0.1):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout_rate)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DirectionEmbedding(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.direction_embeddings = nn.Parameter(torch.randn(3, embed_dim))
        nn.init.normal_(self.direction_embeddings, std=0.02)

    def forward(self, direction_idx):
        return self.direction_embeddings[direction_idx]


class CrossViewAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)

        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout)
        )

    def forward(self, query_feat, key_feat, value_feat):
        B, N, C = query_feat.shape

        q = self.q_proj(query_feat).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_feat).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value_feat).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, C)

        attn_output = self.norm1(query_feat + self.out_proj(attn_output))
        ffn_output = self.ffn(attn_output)
        output = self.norm2(attn_output + ffn_output)

        return output


class ContrastiveHead(nn.Module):
    def __init__(self, feature_dim, projection_dim=128, temperature=0.07):
        super().__init__()
        self.temperature = temperature

        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, projection_dim)
        )

        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim)
        )

    def forward(self, features):
        return self.projector(features)

    def get_predictions(self, features):
        projected = self.projector(features)
        predicted = self.predictor(projected)
        return projected, predicted


class MultiViewUNet3D(nn.Module):
    def __init__(self, n_channels=1, n_classes=4, bilinear=False, dropout_rate=0.1,
                 base_channels=32, use_contrastive=True, feature_dim=512):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_contrastive = use_contrastive
        self.feature_dim = feature_dim
        self.base_channels = base_channels

        self.direction_embedding = DirectionEmbedding(embed_dim=n_channels)

        self.inc = DoubleConv(n_channels, base_channels, dropout_rate=dropout_rate)
        self.down1 = Down(base_channels, base_channels * 2, dropout_rate=dropout_rate)
        self.down2 = Down(base_channels * 2, base_channels * 4, dropout_rate=dropout_rate)
        self.down3 = Down(base_channels * 4, base_channels * 8, dropout_rate=dropout_rate)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor, dropout_rate=dropout_rate)

        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear, dropout_rate=dropout_rate)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear, dropout_rate=dropout_rate)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear, dropout_rate=dropout_rate)
        self.up4 = Up(base_channels * 2, base_channels, bilinear, dropout_rate=dropout_rate)
        self.outc = OutConv(base_channels, n_classes)

        if use_contrastive:
            self.feature_extractor = nn.AdaptiveAvgPool3d(1)
            self.feature_projection = nn.Linear(base_channels * 16 // factor, feature_dim)
            self.cross_attention = CrossViewAttention(feature_dim, num_heads=8, dropout=dropout_rate)
            self.contrastive_head = ContrastiveHead(feature_dim, projection_dim=128)

    def encode(self, x, direction_idx=None):
        if direction_idx is not None:
            direction_emb = self.direction_embedding(direction_idx)
            B, C, D, H, W = x.shape
            direction_emb = direction_emb.view(1, C, 1, 1, 1).expand(B, C, D, H, W)
            x = x + direction_emb

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return x1, x2, x3, x4, x5

    def decode(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def extract_features(self, x5):
        features = self.feature_extractor(x5).squeeze(-1).squeeze(-1).squeeze(-1)
        features = self.feature_projection(features)
        return features

    def forward(self, x, direction_idx=None, return_features=False):
        x1, x2, x3, x4, x5 = self.encode(x, direction_idx)
        logits = self.decode(x1, x2, x3, x4, x5)

        if return_features and self.use_contrastive:
            features = self.extract_features(x5)
            return logits, features

        return logits

    def forward_multi_view(self, sag_x, cor_x, tra_x):
        sag_logits, sag_features = self.forward(sag_x, direction_idx=0, return_features=True)
        cor_logits, cor_features = self.forward(cor_x, direction_idx=1, return_features=True)
        tra_logits, tra_features = self.forward(tra_x, direction_idx=2, return_features=True)

        B, C = sag_features.shape
        sag_features = sag_features.view(B, 1, C)
        cor_features = cor_features.view(B, 1, C)
        tra_features = tra_features.view(B, 1, C)

        sag_attended = self.cross_attention(sag_features, cor_features, tra_features)
        cor_attended = self.cross_attention(cor_features, sag_features, tra_features)
        tra_attended = self.cross_attention(tra_features, sag_features, cor_features)

        sag_features = sag_attended.view(B, C)
        cor_features = cor_attended.view(B, C)
        tra_features = tra_attended.view(B, C)

        return (sag_logits, cor_logits, tra_logits), (sag_features, cor_features, tra_features)


class UNet3D(nn.Module):
    def __init__(self, n_channels=1, n_classes=4, bilinear=False, dropout_rate=0.1, base_channels=32):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, base_channels, dropout_rate=dropout_rate)
        self.down1 = Down(base_channels, base_channels * 2, dropout_rate=dropout_rate)
        self.down2 = Down(base_channels * 2, base_channels * 4, dropout_rate=dropout_rate)
        self.down3 = Down(base_channels * 4, base_channels * 8, dropout_rate=dropout_rate)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor, dropout_rate=dropout_rate)
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear, dropout_rate=dropout_rate)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear, dropout_rate=dropout_rate)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear, dropout_rate=dropout_rate)
        self.up4 = Up(base_channels * 2, base_channels, bilinear, dropout_rate=dropout_rate)
        self.outc = OutConv(base_channels, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


def get_model(config):
    if config['model'].get('use_contrastive', False):
        model = MultiViewUNet3D(
            n_channels=config['model']['n_channels'],
            n_classes=config['model']['n_classes'],
            bilinear=config['model']['bilinear'],
            dropout_rate=config['model']['dropout_rate'],
            base_channels=config['model'].get('base_channels', 32),
            use_contrastive=True,
            feature_dim=config['model'].get('feature_dim', 512)
        )
    else:
        model = UNet3D(
            n_channels=config['model']['n_channels'],
            n_classes=config['model']['n_classes'],
            bilinear=config['model']['bilinear'],
            dropout_rate=config['model']['dropout_rate'],
            base_channels=config['model'].get('base_channels', 32)
        )
    return model