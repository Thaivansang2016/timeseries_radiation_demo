import torch
import torch.nn as nn

class MITransformerModel(nn.Module):
    def __init__(self, img_dim=512, rad_dim=1, d_model=64, nhead=4, num_layers=2):
        super(MITransformerModel, self).__init__()

        # Embedding layers
        self.img_linear = nn.Linear(img_dim, d_model)
        self.rad_linear = nn.Linear(rad_dim, d_model)

        # Positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, 100, d_model))  # max seq_len = 100

        # Transformer encoders
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.img_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.rad_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Normalization & dropout
        self.norm_img = nn.LayerNorm(d_model)
        self.norm_rad = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.3)

        # Fusion & classification
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    def forward(self, img_feat_seq, rad_seq):
        """
        img_feat_seq: [B, T, 512]  -- image feature sequence
        rad_seq:      [B, T] or [B, T, 1]  -- radar/ts sequence
        """
        B, T, _ = img_feat_seq.size()
    
        # Fix rad_seq shape if needed
        if rad_seq.dim() == 2:
            rad_seq = rad_seq.unsqueeze(-1)  # [B, T, 1]
    
        # Project input features to d_model
        img_embed = self.img_linear(img_feat_seq) + self.pos_enc[:, :T]
        rad_embed = self.rad_linear(rad_seq) + self.pos_enc[:, :T]
    
        # Transformer encoding
        img_encoded = self.img_transformer(img_embed)
        rad_encoded = self.rad_transformer(rad_embed)
    
        # Last timestep features
        img_feat = self.norm_img(img_encoded[:, -1, :])
        rad_feat = self.norm_rad(rad_encoded[:, -1, :])
    
        img_feat = self.dropout(img_feat)
        rad_feat = self.dropout(rad_feat)
    
        combined = torch.cat([img_feat, rad_feat], dim=1)
        output = self.fc(combined)
        return output.squeeze()
    
